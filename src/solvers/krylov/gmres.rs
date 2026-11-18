//! GMRES and its variants (FlexGMRES, LGMRES, COGMRES), for general (not
//! necessarily symmetric) matrices.

use derive_builder::Builder;
use std::marker::PhantomData;
use std::ptr::null_mut;

use crate::error::HypreError;
use crate::matrix::Matrix;
use crate::solvers::{IterativeSolverStatus, LinearSolver};
use crate::{HypreContext, HypreResult, Vector};
use hypre_sys::*;

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
    solves LinearSolver for Matrix;
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
    solves LinearSolver for Matrix;
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
    solves LinearSolver for Matrix;
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
    solves LinearSolver for Matrix;
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
