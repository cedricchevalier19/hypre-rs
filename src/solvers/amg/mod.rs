//! BoomerAMG, hypre's algebraic multigrid solver.

use derive_builder::Builder;
use mpi::ffi::MPI_Comm;
use std::marker::PhantomData;
use std::ptr::null_mut;

use crate::error::HypreError;
use crate::matrix::Matrix;
use crate::solvers::{IterativeSolverStatus, LinearSolver};
use crate::{HypreContext, HypreResult, Vector};
use hypre_sys::*;

#[cfg(test)]
mod tests;

/// `HYPRE_BoomerAMGCreate` takes no communicator: BoomerAMG uses the
/// matrix's when it is set up.
unsafe fn boomer_amg_create(_comm: MPI_Comm, solver: *mut HYPRE_Solver) -> HYPRE_Int {
    unsafe { HYPRE_BoomerAMGCreate(solver) }
}

hypre_solver! {
    /// Algebraic multigrid (hypre's BoomerAMG), used as a solver.
    ///
    /// AMG is best suited to matrices arising from elliptic problems, such
    /// as discretized diffusion equations; it is not guaranteed to converge
    /// for other matrices. Like every solver for general matrices, it also
    /// solves systems with a [SymmetricMatrix](crate::SymmetricMatrix).
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use mpi::topology::Communicator;
    /// use hypre_rs::solvers::{AMGSolverConfigBuilder, LinearSolver, Solver};
    /// use hypre_rs::Vector;
    ///
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
    /// // The 1D Laplacian tridiag(-1, 2, -1), from its lower triangle.
    /// let n = 100;
    /// let rows = (rank * n / size)..((rank + 1) * n / size);
    /// let mut a = ctx.symmetric_matrix(&comm, rows.clone())?;
    /// a.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
    /// a.add_elements(rows.clone().filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
    /// let mut b = ctx.ij_vector(&comm, rows.clone())?;
    /// b.add_elements(rows.clone().map(|i| (i, 1.0)))?;
    /// let mut x = ctx.ij_vector(&comm, rows.clone())?;
    /// x.add_elements(rows.clone().map(|i| (i, 0.0)))?;
    /// let (b, mut x) = (Vector::IJ(b), Vector::IJ(x));
    ///
    /// let config = AMGSolverConfigBuilder::default()
    ///     .tol(1e-10)
    ///     .max_iters(100usize)
    ///     .build()?;
    /// let solver = Solver::AMG(ctx.amg_solver(&comm, config)?);
    /// let status = solver.solve(&a, &b, &mut x)?;
    /// assert!(status.converged);
    /// # drop((solver, a, b, x));
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    solver AMGSolver;
    /// Algebraic multigrid (BoomerAMG) configuration
    ///
    /// The integer options (`coarsen_type`, `interp_type`, `relax_type`,
    /// `cycle_type`, `converge_type`) take the values documented in hypre's
    /// reference manual for the matching `HYPRE_BoomerAMGSet*` function.
    ///
    /// hypre has no getter for `min_iters`, `interp_type`, `relax_type`,
    /// `relax_order`, `relax_wt`, `num_sweeps` and `agg_num_levels`, which
    /// are therefore `None` in [AMGSolver::current_config].
    config AMGSolverConfig, AMGSolverConfigBuilder;
    create boomer_amg_create, destroy HYPRE_BoomerAMGDestroy;
    setup HYPRE_BoomerAMGSetup, solve HYPRE_BoomerAMGSolve;
    solves LinearSolver for Matrix;
    status HYPRE_BoomerAMGGetNumIterations, HYPRE_BoomerAMGGetFinalRelativeResidualNorm;
    positive [tol, strong_threshold, max_row_sum, trunc_factor];
    params {
        /// Relative convergence tolerance (hypre's default is 1e-7)
        tol: f64 => HYPRE_BoomerAMGSetTol,
        /// Maximum number of multigrid cycles (hypre's default is 20)
        max_iters: usize => HYPRE_BoomerAMGSetMaxIter,
        /// Minimum number of multigrid cycles
        min_iters: usize => HYPRE_BoomerAMGSetMinIter,
        /// Maximum number of multigrid levels
        max_levels: usize => HYPRE_BoomerAMGSetMaxLevels,
        /// Maximum size of the coarsest grid
        max_coarse_size: usize => HYPRE_BoomerAMGSetMaxCoarseSize,
        /// Minimum size of the coarsest grid
        min_coarse_size: usize => HYPRE_BoomerAMGSetMinCoarseSize,
        /// Strength threshold of the coarsening (0.25 by default; 0.5 is
        /// often better for 3D problems)
        strong_threshold: f64 => HYPRE_BoomerAMGSetStrongThreshold,
        /// Rows whose off-diagonal sum exceeds this fraction of the diagonal
        /// are treated as weakly dependent (1 disables it)
        max_row_sum: f64 => HYPRE_BoomerAMGSetMaxRowSum,
        /// Truncation factor of the interpolation
        trunc_factor: f64 => HYPRE_BoomerAMGSetTruncFactor,
        /// Maximum number of elements per row of the interpolation
        p_max_elmts: usize => HYPRE_BoomerAMGSetPMaxElmts,
        /// Coarsening algorithm (6: Falgout, 8: PMIS, 10: HMIS, ...)
        coarsen_type: u32 => HYPRE_BoomerAMGSetCoarsenType,
        /// Interpolation operator (0: classical, 6: extended+i, ...)
        interp_type: u32 => HYPRE_BoomerAMGSetInterpType,
        /// Smoother on every level (3: hybrid Gauss-Seidel, 6: symmetric
        /// hybrid Gauss-Seidel, 8: ℓ1 Gauss-Seidel, ...)
        relax_type: u32 => HYPRE_BoomerAMGSetRelaxType,
        /// Relax C points before F points (`false`: lexicographic order)
        relax_order: bool => HYPRE_BoomerAMGSetRelaxOrder,
        /// Relaxation weight of the weighted Jacobi and SOR smoothers
        relax_wt: f64 => HYPRE_BoomerAMGSetRelaxWt,
        /// Number of smoother sweeps on every level
        num_sweeps: usize => HYPRE_BoomerAMGSetNumSweeps,
        /// Multigrid cycle (1: V-cycle, 2: W-cycle)
        cycle_type: u32 => HYPRE_BoomerAMGSetCycleType,
        /// Number of levels with aggressive coarsening
        agg_num_levels: usize => HYPRE_BoomerAMGSetAggNumLevels,
        /// Convergence criterion (0: relative residual, 1: residual relative
        /// to the initial residual)
        converge_type: u32 => HYPRE_BoomerAMGSetConvergeType,
        /// Logging level (1 computes the residual norms)
        logging: u32 => HYPRE_BoomerAMGSetLogging,
        /// Print level (0: nothing, 1: setup information, 2: residual norm
        /// of every cycle, 3: both)
        print_level: u32 => HYPRE_BoomerAMGSetPrintLevel,
    }
    getters {
        tol: HYPRE_Real => HYPRE_BoomerAMGGetTol,
        max_iters: HYPRE_Int => HYPRE_BoomerAMGGetMaxIter,
        max_levels: HYPRE_Int => HYPRE_BoomerAMGGetMaxLevels,
        max_coarse_size: HYPRE_Int => HYPRE_BoomerAMGGetMaxCoarseSize,
        min_coarse_size: HYPRE_Int => HYPRE_BoomerAMGGetMinCoarseSize,
        strong_threshold: HYPRE_Real => HYPRE_BoomerAMGGetStrongThreshold,
        max_row_sum: HYPRE_Real => HYPRE_BoomerAMGGetMaxRowSum,
        trunc_factor: HYPRE_Real => HYPRE_BoomerAMGGetTruncFactor,
        p_max_elmts: HYPRE_Int => HYPRE_BoomerAMGGetPMaxElmts,
        coarsen_type: HYPRE_Int => HYPRE_BoomerAMGGetCoarsenType,
        cycle_type: HYPRE_Int => HYPRE_BoomerAMGGetCycleType,
        converge_type: HYPRE_Int => HYPRE_BoomerAMGGetConvergeType,
        logging: HYPRE_Int => HYPRE_BoomerAMGGetLogging,
        print_level: HYPRE_Int => HYPRE_BoomerAMGGetPrintLevel,
    }
}
