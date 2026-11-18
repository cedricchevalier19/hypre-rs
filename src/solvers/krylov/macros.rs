//! The `krylov_solver!` macro, which generates the hypre Krylov solvers.

use crate::HypreResult;
use hypre_sys::{HYPRE_Int, HYPRE_Real};

/// Converts a parameter read from hypre to its configuration type.
pub(crate) trait FromHypre<Raw>: Sized {
    fn from_hypre(raw: Raw) -> HypreResult<Self>;
}

impl FromHypre<HYPRE_Real> for f64 {
    fn from_hypre(raw: HYPRE_Real) -> HypreResult<Self> {
        Ok(raw)
    }
}

impl FromHypre<HYPRE_Int> for usize {
    fn from_hypre(raw: HYPRE_Int) -> HypreResult<Self> {
        Ok(raw.try_into()?)
    }
}

impl FromHypre<HYPRE_Int> for u32 {
    fn from_hypre(raw: HYPRE_Int) -> HypreResult<Self> {
        Ok(raw.try_into()?)
    }
}

/// hypre's boolean parameters are ints, nonzero meaning `true`.
impl FromHypre<HYPRE_Int> for bool {
    fn from_hypre(raw: HYPRE_Int) -> HypreResult<Self> {
        Ok(raw != 0)
    }
}

/// Generates a solver wrapping a ParCSR Krylov solver of hypre:
///
/// - the solver type, owning its `HYPRE_Solver` handle (`new`, `config`,
///   `Drop`), implementing the solver trait given by
///   `solves Trait for MatrixType` ([LinearSolver](crate::solvers::LinearSolver)
///   for `Matrix`, or
///   [SymmetricLinearSolver](crate::solvers::SymmetricLinearSolver) for
///   `SymmetricMatrix`),
/// - its configuration type, with a builder that rejects negative
///   tolerances,
/// - `current_config` when hypre has getters for the parameters.
///
/// Each parameter `field: Type => Setter` is an `Option<Type>` field of the
/// configuration, applied with `Setter` when `Some`. The optional
/// `getters { field: RawType => Getter }` read them back, converting with
/// [FromHypre]; parameters without a getter are `None` in `current_config`.
macro_rules! krylov_solver {
    (
        $(#[$solver_doc:meta])*
        solver $Solver:ident;
        $(#[$config_doc:meta])*
        config $Config:ident, $Builder:ident;
        create $create:path, destroy $destroy:path;
        setup $setup:path, solve $solve:path;
        solves $Trait:ident for $Mat:ident;
        status $num_iters:path, $res_norm:path $(, converged $converged:path)?;
        positive [$($positive:ident),*];
        params {
            $( $(#[$field_doc:meta])* $field:ident: $ty:ty => $setter:path, )*
        }
        $( getters {
            $( $gfield:ident: $raw:ty => $getter:path, )*
        } )?
    ) => {
        $(#[$solver_doc])*
        #[derive(Debug)]
        pub struct $Solver<'h> {
            internal_solver: HYPRE_Solver,
            _hypre: PhantomData<&'h ()>,
        }

        $(#[$config_doc])*
        ///
        /// Fields set to `None` keep hypre's default (or, for
        #[doc = concat!("[", stringify!($Solver), "::config], the current value).")]
        #[derive(Default, Debug, Clone, Builder, Copy)]
        #[builder(setter(into, strip_option), default, derive(Debug, Copy))]
        #[builder(build_fn(validate = "Self::validate", error = "HypreError"))]
        pub struct $Config {
            $( $(#[$field_doc])* pub $field: Option<$ty>, )*
        }

        impl $Builder {
            /// Rejects negative tolerances.
            fn validate(&self) -> HypreResult<()> {
                $( check_positive_parameter![self, $positive]; )*
                Ok(())
            }
        }

        impl<'h> $Solver<'h> {
            /// Creates a solver with the given configuration.
            ///
            /// # Errors
            ///
            #[doc = concat!("See [", stringify!($Solver), "::config].")]
            pub fn new(
                _hypre: &'h HypreContext<'_>,
                comm: &'h impl mpi::topology::Communicator,
                config: $Config,
            ) -> HypreResult<Self> {
                let mut solver = Self {
                    internal_solver: null_mut(),
                    _hypre: PhantomData,
                };
                unsafe {
                    check_hypre!($create(comm.as_raw(), &mut solver.internal_solver));
                }
                // On error, `solver` is dropped and its handle destroyed.
                solver.config(config)
            }

            /// Changes the configuration of the solver. Fields set to `None`
            /// keep their current value.
            ///
            /// # Errors
            ///
            /// [HypreError::ConversionError] if a value does not fit hypre's
            /// type, or the error returned by hypre. The solver is dropped
            /// then, and fields set before the failing one were already
            /// applied.
            pub fn config(self, config: $Config) -> HypreResult<Self> {
                $( set_parameter![$setter, self.internal_solver, config.$field]; )*
                Ok(self)
            }

            $(
            /// Returns the configuration of the solver, as read from hypre.
            pub fn current_config(&self) -> HypreResult<$Config> {
                let mut config = $Config::default();
                $(
                    let raw: $raw = get_parameter![$getter, self.internal_solver, $raw]?;
                    config.$gfield = Some(
                        $crate::solvers::krylov::macros::FromHypre::from_hypre(raw)?,
                    );
                )*
                Ok(config)
            }
            )?
        }

        impl $Trait for $Solver<'_> {
            fn solve(
                &self,
                mat: &$Mat<'_>,
                rhs: &Vector<'_>,
                x: &mut Vector<'_>,
            ) -> HypreResult<IterativeSolverStatus> {
                let a = mat.get_internal()? as HYPRE_ParCSRMatrix;
                let b = rhs.get_internal()? as HYPRE_ParVector;
                let x = x.get_internal()? as HYPRE_ParVector;
                unsafe {
                    check_hypre!($setup(self.internal_solver, a, b, x));
                    // A solve that does not converge returns HYPRE_ERROR_CONV.
                    check_hypre!($solve(self.internal_solver, a, b, x));
                    // The getters cannot fail on a set-up solver.
                    let mut num_iters: HYPRE_Int = 0;
                    $num_iters(self.internal_solver, &mut num_iters);
                    let mut res_norm: HYPRE_Real = 0.0;
                    $res_norm(self.internal_solver, &mut res_norm);
                    #[allow(unused_mut)]
                    let mut converged: HYPRE_Int = 1;
                    $( $converged(self.internal_solver, &mut converged); )?
                    Ok(IterativeSolverStatus {
                        num_iters: num_iters.try_into()?,
                        res_num: res_norm,
                        converged: converged != 0,
                    })
                }
            }
        }

        impl_hypre_drop!($Solver<'_>, internal_solver, $destroy);
    };
}
