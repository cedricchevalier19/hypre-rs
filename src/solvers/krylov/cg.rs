//! The Conjugate Gradient solver, for symmetric positive definite matrices.

use derive_builder::Builder;
use std::marker::PhantomData;
use std::ptr::null_mut;

use crate::error::HypreError;
use crate::matrix::SymmetricMatrix;
use crate::preconditioners::SymmetricPreconditioner;
use crate::solvers::{IterativeSolverStatus, SymmetricLinearSolver};
use crate::{HypreContext, HypreResult, Vector};
use hypre_sys::*;

hypre_solver! {
    /// Preconditioned Conjugate Gradient solver.
    ///
    /// CG requires a symmetric positive definite matrix, so it solves systems
    /// with a [SymmetricMatrix] only (see [SymmetricLinearSolver]).
    ///
    /// # Example
    ///
    /// ```
    /// use hypre_rs::solvers::PCGSolver;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let mpi_comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// // Create PCGSolver with hypre's default configuration.
    /// let solver = PCGSolver::new(&ctx, &mpi_comm, Default::default())?;
    /// # drop(solver);
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    solver PCGSolver;
    /// Conjugate Gradient Solver Configuration
    ///
    /// ## Convergence criteria
    ///
    /// According to hypre source code, default is
    /// $$<C*r, r>  < \text{max}({\text{tol}_a^2, tol^2 * <C*b, b>)$$
    ///
    /// If res_tol is set, $$||r_\text{new} - r_\text{old}||_C < \text{tol}_r * ||b||_C$$ is used.
    ///
    /// If two_norm is set, $$||.||_2$$ norm is used.
    ///
    /// hypre has no getter for `abs_tol`, `recompute_residual` and
    /// `recompute_residual_period`, which are therefore `None` in
    /// [PCGSolver::current_config].
    config PCGSolverConfig, PCGSolverConfigBuilder;
    create HYPRE_ParCSRPCGCreate, destroy HYPRE_ParCSRPCGDestroy;
    setup HYPRE_ParCSRPCGSetup, solve HYPRE_ParCSRPCGSolve;
    solves SymmetricLinearSolver for SymmetricMatrix;
    status HYPRE_ParCSRPCGGetNumIterations, HYPRE_ParCSRPCGGetFinalRelativeResidualNorm,
        converged HYPRE_PCGGetConverged;
    precond HYPRE_ParCSRPCGSetPrecond, SymmetricPreconditioner;
    positive [tol, abs_tol, res_tol, conv_tol_fact];
    params {
        /// Relative convergence tolerance
        tol: f64 => HYPRE_PCGSetTol,
        /// Absolute convergence tolerance
        abs_tol: f64 => HYPRE_PCGSetAbsoluteTol,
        /// Residual convergence tolerance
        res_tol: f64 => HYPRE_PCGSetResidualTol,
        /// Convergence factor tolerance, if >0 used for special test for slow convergence
        conv_tol_fact: f64 => HYPRE_PCGSetConvergenceFactorTol,
        /// Maximum number of iterations
        max_iters: usize => HYPRE_PCGSetMaxIter,
        /// Force using $||.||_2$ instead of $||.||_C$
        two_norm: bool => HYPRE_PCGSetTwoNorm,
        /// Convergence only if $x$ didn't change too much in the last iteration
        rel_change: bool => HYPRE_PCGSetRelChange,
        /// Force computing an explicit residual if implicit residual convergence
        recompute_residual: bool => HYPRE_PCGSetRecomputeResidual,
        /// Periodically recompute an explicit residual
        recompute_residual_period: usize => HYPRE_PCGSetRecomputeResidualP,
        /// Logging level (1 stores the residual norms)
        logging: u32 => HYPRE_PCGSetLogging,
        /// Print level (0: nothing, 2: residual norm of every iteration)
        print_level: u32 => HYPRE_PCGSetPrintLevel,
    }
    getters {
        tol: HYPRE_Real => HYPRE_PCGGetTol,
        res_tol: HYPRE_Real => HYPRE_PCGGetResidualTol,
        conv_tol_fact: HYPRE_Real => HYPRE_PCGGetConvergenceFactorTol,
        max_iters: HYPRE_Int => HYPRE_PCGGetMaxIter,
        two_norm: HYPRE_Int => HYPRE_PCGGetTwoNorm,
        rel_change: HYPRE_Int => HYPRE_PCGGetRelChange,
        logging: HYPRE_Int => HYPRE_PCGGetLogging,
        print_level: HYPRE_Int => HYPRE_PCGGetPrintLevel,
    }
}
