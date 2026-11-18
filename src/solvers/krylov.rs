//! Krylov solvers for general (not necessarily symmetric) matrices:
//! GMRES and its variants, and BiCGSTAB.
//!
//! They only differ by their hypre functions and parameters, so
//! [krylov_solver] generates each of them.

use derive_builder::Builder;
use std::marker::PhantomData;
use std::ptr::null_mut;

use crate::error::HypreError;
use crate::matrix::Matrix;
use crate::solvers::{IterativeSolverStatus, LinearSolver};
use crate::{HypreContext, HypreResult, Vector};
use hypre_sys::*;

/// Generates a solver wrapping a ParCSR Krylov solver of hypre:
///
/// - the solver type, owning its `HYPRE_Solver` handle (`new`, `config`,
///   [LinearSolver], `Drop`),
/// - its configuration type, with a builder that rejects negative
///   tolerances,
/// - `current_config` when hypre has getters for the parameters.
///
/// Each parameter `field: Type => Setter` is an `Option<Type>` field of the
/// configuration, applied with `Setter` when `Some`. The optional
/// `getters { field: RawType => Getter }` read them back; parameters without
/// a getter are `None` in `current_config`.
macro_rules! krylov_solver {
    (
        $(#[$solver_doc:meta])*
        solver $Solver:ident;
        $(#[$config_doc:meta])*
        config $Config:ident, $Builder:ident;
        create $create:path, destroy $destroy:path;
        setup $setup:path, solve $solve:path;
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
                    let value = get_parameter![$getter, self.internal_solver, $raw]?;
                    config.$gfield = Some(value);
                )*
                Ok(config)
            }
            )?
        }

        impl LinearSolver for $Solver<'_> {
            fn solve(
                &self,
                mat: &Matrix<'_>,
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

krylov_solver! {
    /// Restarted GMRES (Generalized Minimal RESidual) solver, for general
    /// matrices.
    ///
    /// # Example
    ///
    /// ```
    /// use hypre_rs::solvers::{GMRESSolver, GMRESSolverConfigBuilder};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// let config = GMRESSolverConfigBuilder::default()
    ///     .tol(1e-8)
    ///     .k_dim(30usize)
    ///     .build()?;
    /// let solver = GMRESSolver::new(&ctx, &comm, config)?;
    /// assert_eq!(solver.current_config()?.k_dim, Some(30));
    /// # drop(solver);
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    solver GMRESSolver;
    /// [GMRESSolver] configuration.
    config GMRESSolverConfig, GMRESSolverConfigBuilder;
    create HYPRE_ParCSRGMRESCreate, destroy HYPRE_ParCSRGMRESDestroy;
    setup HYPRE_ParCSRGMRESSetup, solve HYPRE_ParCSRGMRESSolve;
    status HYPRE_ParCSRGMRESGetNumIterations, HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm,
        converged HYPRE_GMRESGetConverged;
    positive [tol, abs_tol];
    params {
        /// Relative convergence tolerance
        tol: f64 => HYPRE_ParCSRGMRESSetTol,
        /// Absolute convergence tolerance
        abs_tol: f64 => HYPRE_ParCSRGMRESSetAbsoluteTol,
        /// Maximum number of iterations
        max_iters: usize => HYPRE_ParCSRGMRESSetMaxIter,
        /// Minimum number of iterations
        min_iters: usize => HYPRE_ParCSRGMRESSetMinIter,
        /// Dimension of the Krylov space: GMRES restarts after `k_dim`
        /// iterations
        k_dim: usize => HYPRE_ParCSRGMRESSetKDim,
        /// Logging level (1 stores the residual norms)
        logging: u32 => HYPRE_ParCSRGMRESSetLogging,
        /// Print level (0: nothing, 2: residual norm of every iteration)
        print_level: u32 => HYPRE_ParCSRGMRESSetPrintLevel,
    }
    getters {
        tol: HYPRE_Real => HYPRE_GMRESGetTol,
        abs_tol: HYPRE_Real => HYPRE_GMRESGetAbsoluteTol,
        max_iters: HYPRE_Int => HYPRE_GMRESGetMaxIter,
        min_iters: HYPRE_Int => HYPRE_GMRESGetMinIter,
        k_dim: HYPRE_Int => HYPRE_GMRESGetKDim,
        logging: HYPRE_Int => HYPRE_GMRESGetLogging,
        print_level: HYPRE_Int => HYPRE_GMRESGetPrintLevel,
    }
}

krylov_solver! {
    /// Flexible GMRES solver, for general matrices: allows a preconditioner
    /// that changes from one iteration to the next.
    solver FlexGMRESSolver;
    /// [FlexGMRESSolver] configuration.
    config FlexGMRESSolverConfig, FlexGMRESSolverConfigBuilder;
    create HYPRE_ParCSRFlexGMRESCreate, destroy HYPRE_ParCSRFlexGMRESDestroy;
    setup HYPRE_ParCSRFlexGMRESSetup, solve HYPRE_ParCSRFlexGMRESSolve;
    status HYPRE_ParCSRFlexGMRESGetNumIterations,
        HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm,
        converged HYPRE_FlexGMRESGetConverged;
    positive [tol, abs_tol];
    params {
        /// Relative convergence tolerance
        tol: f64 => HYPRE_ParCSRFlexGMRESSetTol,
        /// Absolute convergence tolerance
        abs_tol: f64 => HYPRE_ParCSRFlexGMRESSetAbsoluteTol,
        /// Maximum number of iterations
        max_iters: usize => HYPRE_ParCSRFlexGMRESSetMaxIter,
        /// Minimum number of iterations
        min_iters: usize => HYPRE_ParCSRFlexGMRESSetMinIter,
        /// Dimension of the Krylov space: the solver restarts after `k_dim`
        /// iterations
        k_dim: usize => HYPRE_ParCSRFlexGMRESSetKDim,
        /// Logging level (1 stores the residual norms)
        logging: u32 => HYPRE_ParCSRFlexGMRESSetLogging,
        /// Print level (0: nothing, 2: residual norm of every iteration)
        print_level: u32 => HYPRE_ParCSRFlexGMRESSetPrintLevel,
    }
    getters {
        tol: HYPRE_Real => HYPRE_FlexGMRESGetTol,
        max_iters: HYPRE_Int => HYPRE_FlexGMRESGetMaxIter,
        min_iters: HYPRE_Int => HYPRE_FlexGMRESGetMinIter,
        k_dim: HYPRE_Int => HYPRE_FlexGMRESGetKDim,
        logging: HYPRE_Int => HYPRE_FlexGMRESGetLogging,
        print_level: HYPRE_Int => HYPRE_FlexGMRESGetPrintLevel,
    }
}

krylov_solver! {
    /// LGMRES (Loose GMRES) solver, for general matrices: GMRES whose
    /// restarts keep approximations of the error from previous cycles, which
    /// speeds up convergence.
    solver LGMRESSolver;
    /// [LGMRESSolver] configuration.
    config LGMRESSolverConfig, LGMRESSolverConfigBuilder;
    create HYPRE_ParCSRLGMRESCreate, destroy HYPRE_ParCSRLGMRESDestroy;
    setup HYPRE_ParCSRLGMRESSetup, solve HYPRE_ParCSRLGMRESSolve;
    status HYPRE_ParCSRLGMRESGetNumIterations, HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm,
        converged HYPRE_LGMRESGetConverged;
    positive [tol, abs_tol];
    params {
        /// Relative convergence tolerance
        tol: f64 => HYPRE_ParCSRLGMRESSetTol,
        /// Absolute convergence tolerance
        abs_tol: f64 => HYPRE_ParCSRLGMRESSetAbsoluteTol,
        /// Maximum number of iterations
        max_iters: usize => HYPRE_ParCSRLGMRESSetMaxIter,
        /// Minimum number of iterations
        min_iters: usize => HYPRE_ParCSRLGMRESSetMinIter,
        /// Dimension of the Krylov space: the solver restarts after `k_dim`
        /// iterations
        k_dim: usize => HYPRE_ParCSRLGMRESSetKDim,
        /// Number of error approximations kept from previous cycles
        aug_dim: usize => HYPRE_ParCSRLGMRESSetAugDim,
        /// Logging level (1 stores the residual norms)
        logging: u32 => HYPRE_ParCSRLGMRESSetLogging,
        /// Print level (0: nothing, 2: residual norm of every iteration)
        print_level: u32 => HYPRE_ParCSRLGMRESSetPrintLevel,
    }
    getters {
        tol: HYPRE_Real => HYPRE_LGMRESGetTol,
        max_iters: HYPRE_Int => HYPRE_LGMRESGetMaxIter,
        min_iters: HYPRE_Int => HYPRE_LGMRESGetMinIter,
        k_dim: HYPRE_Int => HYPRE_LGMRESGetKDim,
        aug_dim: HYPRE_Int => HYPRE_LGMRESGetAugDim,
        logging: HYPRE_Int => HYPRE_LGMRESGetLogging,
        print_level: HYPRE_Int => HYPRE_LGMRESGetPrintLevel,
    }
}

krylov_solver! {
    /// COGMRES (Communication-Optimal GMRES) solver, for general matrices:
    /// GMRES with an orthogonalization that needs fewer global reductions.
    solver COGMRESSolver;
    /// [COGMRESSolver] configuration.
    config COGMRESSolverConfig, COGMRESSolverConfigBuilder;
    create HYPRE_ParCSRCOGMRESCreate, destroy HYPRE_ParCSRCOGMRESDestroy;
    setup HYPRE_ParCSRCOGMRESSetup, solve HYPRE_ParCSRCOGMRESSolve;
    status HYPRE_ParCSRCOGMRESGetNumIterations,
        HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm,
        converged HYPRE_COGMRESGetConverged;
    positive [tol, abs_tol];
    params {
        /// Relative convergence tolerance
        tol: f64 => HYPRE_ParCSRCOGMRESSetTol,
        /// Absolute convergence tolerance
        abs_tol: f64 => HYPRE_ParCSRCOGMRESSetAbsoluteTol,
        /// Maximum number of iterations
        max_iters: usize => HYPRE_ParCSRCOGMRESSetMaxIter,
        /// Minimum number of iterations
        min_iters: usize => HYPRE_ParCSRCOGMRESSetMinIter,
        /// Dimension of the Krylov space: the solver restarts after `k_dim`
        /// iterations
        k_dim: usize => HYPRE_ParCSRCOGMRESSetKDim,
        /// Loop unrolling factor of the orthogonalization
        unroll: u32 => HYPRE_ParCSRCOGMRESSetUnroll,
        /// Gram-Schmidt variant: 1 for classical Gram-Schmidt, 2 for
        /// classical Gram-Schmidt with reorthogonalization
        cgs: u32 => HYPRE_ParCSRCOGMRESSetCGS,
        /// Logging level (1 stores the residual norms)
        logging: u32 => HYPRE_ParCSRCOGMRESSetLogging,
        /// Print level (0: nothing, 2: residual norm of every iteration)
        print_level: u32 => HYPRE_ParCSRCOGMRESSetPrintLevel,
    }
    getters {
        tol: HYPRE_Real => HYPRE_COGMRESGetTol,
        max_iters: HYPRE_Int => HYPRE_COGMRESGetMaxIter,
        min_iters: HYPRE_Int => HYPRE_COGMRESGetMinIter,
        k_dim: HYPRE_Int => HYPRE_COGMRESGetKDim,
        unroll: HYPRE_Int => HYPRE_COGMRESGetUnroll,
        cgs: HYPRE_Int => HYPRE_COGMRESGetCGS,
        logging: HYPRE_Int => HYPRE_COGMRESGetLogging,
        print_level: HYPRE_Int => HYPRE_COGMRESGetPrintLevel,
    }
}

krylov_solver! {
    /// BiCGSTAB (Biconjugate Gradient Stabilized) solver, for general
    /// matrices: low memory use, no restart, but a less smooth convergence
    /// than GMRES.
    ///
    /// hypre cannot read its parameters back, so it has no
    /// `current_config`.
    solver BiCGSTABSolver;
    /// [BiCGSTABSolver] configuration.
    config BiCGSTABSolverConfig, BiCGSTABSolverConfigBuilder;
    create HYPRE_ParCSRBiCGSTABCreate, destroy HYPRE_ParCSRBiCGSTABDestroy;
    setup HYPRE_ParCSRBiCGSTABSetup, solve HYPRE_ParCSRBiCGSTABSolve;
    status HYPRE_ParCSRBiCGSTABGetNumIterations,
        HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm;
    positive [tol, abs_tol];
    params {
        /// Relative convergence tolerance
        tol: f64 => HYPRE_ParCSRBiCGSTABSetTol,
        /// Absolute convergence tolerance
        abs_tol: f64 => HYPRE_ParCSRBiCGSTABSetAbsoluteTol,
        /// Maximum number of iterations
        max_iters: usize => HYPRE_ParCSRBiCGSTABSetMaxIter,
        /// Minimum number of iterations
        min_iters: usize => HYPRE_ParCSRBiCGSTABSetMinIter,
        /// Logging level (1 stores the residual norms)
        logging: u32 => HYPRE_ParCSRBiCGSTABSetLogging,
        /// Print level (0: nothing, 2: residual norm of every iteration)
        print_level: u32 => HYPRE_ParCSRBiCGSTABSetPrintLevel,
    }
}
