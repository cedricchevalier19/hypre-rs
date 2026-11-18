//! The BiCGSTAB solver, for general (not necessarily symmetric) matrices.

use derive_builder::Builder;
use std::marker::PhantomData;
use std::ptr::null_mut;

use crate::error::HypreError;
use crate::matrix::Matrix;
use crate::solvers::{IterativeSolverStatus, LinearSolver};
use crate::{HypreContext, HypreResult, Vector};
use hypre_sys::*;

hypre_solver! {
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
    solves LinearSolver for Matrix;
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
