//! The `hypre_solver!` macro, which generates the hypre solvers and
//! preconditioners.

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

/// Generates a type wrapping a ParCSR solver or preconditioner of hypre:
///
/// - the type, owning its `HYPRE_Solver` handle (`new`, `config`, `Drop`),
/// - its configuration type, with a builder that rejects negative values of
///   the `positive` parameters,
/// - `current_config` when hypre has getters for the parameters,
/// - with `setup`, `solve`, `solves Trait for MatrixType` and `status`, the
///   solver trait; `MatrixType` is written in terms of the generic matrix
///   `M` of `solve`: `M` for [LinearSolver](crate::solvers::LinearSolver),
///   `SymmetricMatrix<M>` for
///   [SymmetricLinearSolver](crate::solvers::SymmetricLinearSolver).
///   Preconditioners leave it out,
/// - with `precond SetPrecond, PreconditionerType`, an owned preconditioner
///   (`with_preconditioner`, `preconditioner`), set with hypre's
///   `SetPrecond` function.
///
/// `create` is called as `create(comm, &mut handle)`.
///
/// Each parameter `field: Type => Setter` is an `Option<Type>` field of the
/// configuration, applied with `Setter` when `Some`. The optional
/// `getters { field: RawType => Getter }` read them back, converting with
/// [FromHypre]; parameters without a getter are `None` in `current_config`.
macro_rules! hypre_solver {
    (
        $(#[$solver_doc:meta])*
        solver $Solver:ident;
        $(#[$config_doc:meta])*
        config $Config:ident, $Builder:ident;
        create $create:path, destroy $destroy:path;
        $(
            setup $setup:path, solve $solve:path;
            solves $Trait:ident for $MatTy:ty;
            status $num_iters:path, $res_norm:path $(, converged $converged:path)?;
        )?
        $( precond $set_precond:path, $Precond:ident; )?
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
            $(
            /// hypre does not copy the preconditioner: keep it alive while
            /// the solver is.
            preconditioner: Option<$Precond<'h>>,
            )?
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
            #[doc = concat!("Creates a [", stringify!($Solver), "] with the given configuration.")]
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
                    $( preconditioner: None::<$Precond<'h>>, )?
                    _hypre: PhantomData,
                };
                unsafe {
                    check_hypre!($create(comm.as_raw(), &mut solver.internal_solver));
                }
                // On error, `solver` is dropped and its handle destroyed.
                solver.config(config)
            }

            /// Changes the configuration. Fields set to `None` keep their
            /// current value.
            ///
            /// # Errors
            ///
            /// [HypreError::ConversionError] if a value does not fit hypre's
            /// type, or the error returned by hypre. `self` is dropped then,
            /// and fields set before the failing one were already applied.
            pub fn config(self, config: $Config) -> HypreResult<Self> {
                $( set_parameter![$setter, self.internal_solver, config.$field]; )*
                Ok(self)
            }

            $(
            /// Returns the configuration, as read from hypre.
            pub fn current_config(&self) -> HypreResult<$Config> {
                let mut config = $Config::default();
                $(
                    let raw: $raw = get_parameter![$getter, self.internal_solver, $raw]?;
                    config.$gfield = Some(
                        $crate::solvers::solver_macro::FromHypre::from_hypre(raw)?,
                    );
                )*
                Ok(config)
            }
            )?

            $(
            /// Sets the preconditioner, replacing (and dropping) the previous
            /// one. The solver owns it from now on.
            ///
            /// # Errors
            ///
            /// The error returned by hypre; `self` is dropped then.
            pub fn with_preconditioner(
                mut self,
                preconditioner: impl Into<$Precond<'h>>,
            ) -> HypreResult<Self> {
                let preconditioner = preconditioner.into();
                let raw = preconditioner.raw();
                unsafe {
                    check_hypre!($set_precond(
                        self.internal_solver,
                        raw.solve,
                        raw.setup,
                        raw.solver
                    ));
                }
                self.preconditioner = Some(preconditioner);
                Ok(self)
            }

            /// Returns the preconditioner, if one is set.
            pub fn preconditioner(&self) -> Option<&$Precond<'h>> {
                self.preconditioner.as_ref()
            }
            )?
        }

        $(
        impl $Trait for $Solver<'_> {
            fn solve<M: $crate::Matrix + ?Sized>(
                &self,
                mat: &$MatTy,
                rhs: &$crate::vector::ParVector<'_>,
                x: &mut $crate::vector::ParVector<'_>,
            ) -> HypreResult<IterativeSolverStatus> {
                let a = $crate::matrix::sealed::AsParCSR::as_parcsr(mat)?;
                let b = rhs.as_raw();
                let x = x.as_raw();
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
        )?

        impl_hypre_drop!($Solver<'_>, internal_solver, $destroy);
    };
}
