//! BoomerAMG, hypre's algebraic multigrid, as a preconditioner.
//!
//! Applying the preconditioner runs `max_iters` multigrid cycles (one by
//! default) from a zero initial guess, with a tolerance of 0: it is then a
//! fixed linear operator, symmetric when the cycle is.

use derive_builder::Builder;
use mpi::ffi::MPI_Comm;
use std::marker::PhantomData;
use std::ptr::null_mut;

use super::RawPreconditioner;
use crate::error::HypreError;
use crate::solvers::{CoarsenType, CycleType, InterpType, Smoother};
use crate::{HypreContext, HypreResult};
use hypre_sys::*;

/// Creates a BoomerAMG object set up to be a preconditioner: one cycle,
/// without convergence test. `HYPRE_BoomerAMGCreate` takes no communicator.
unsafe fn create_amg_preconditioner(_comm: MPI_Comm, solver: *mut HYPRE_Solver) -> HYPRE_Int {
    unsafe {
        let err = HYPRE_BoomerAMGCreate(solver);
        if err != 0 {
            return err;
        }
        let err = HYPRE_BoomerAMGSetTol(*solver, 0.0);
        if err != 0 {
            return err;
        }
        HYPRE_BoomerAMGSetMaxIter(*solver, 1)
    }
}

/// Smoothers that keep the multigrid cycle symmetric.
///
/// The same smoother is used before and after the coarse-grid correction,
/// on every level, so the cycle is symmetric when the smoother is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetricSmoother {
    /// Weighted Jacobi (hypre's relaxation type 0); see `relax_wt`.
    Jacobi,
    /// Hybrid symmetric Gauss-Seidel, or SSOR (type 6).
    HybridSymmetricGaussSeidel,
    /// ℓ1-scaled hybrid symmetric Gauss-Seidel (type 8).
    L1SymmetricGaussSeidel,
    /// Chebyshev polynomial (type 16).
    Chebyshev,
    /// ℓ1-scaled Jacobi (type 18).
    L1Jacobi,
}

impl From<SymmetricSmoother> for HYPRE_Int {
    fn from(smoother: SymmetricSmoother) -> Self {
        match smoother {
            SymmetricSmoother::Jacobi => 0,
            SymmetricSmoother::HybridSymmetricGaussSeidel => 6,
            SymmetricSmoother::L1SymmetricGaussSeidel => 8,
            SymmetricSmoother::Chebyshev => 16,
            SymmetricSmoother::L1Jacobi => 18,
        }
    }
}

hypre_solver! {
    /// Algebraic multigrid preconditioner (hypre's BoomerAMG) with a
    /// symmetric cycle, for the conjugate gradient solver.
    ///
    /// Its configuration leaves out what would make the cycle
    /// non-symmetric: the smoother is a [SymmetricSmoother]. The C/F
    /// relaxation order (`relax_order`) keeps the cycle symmetric: hypre
    /// relaxes C points before F points on the way down, and F points
    /// before C points on the way up. hypre's default setup is
    /// symmetric (ℓ1 Gauss-Seidel, forward before and backward after the
    /// coarse-grid correction). The matrix must be symmetric positive
    /// definite.
    ///
    /// Use [AMGPreconditioner] for the full set of options, with the
    /// solvers of general matrices.
    solver SymmetricAMGPreconditioner;
    /// Configuration of the symmetric algebraic multigrid preconditioner
    ///
    /// See [AMGSolverConfig](crate::solvers::AMGSolverConfig) for the
    /// meaning of the parameters. hypre has no getter for `interp_type`,
    /// `relax_type`, `relax_order`, `relax_wt`, `num_sweeps` and
    /// `agg_num_levels`, which are therefore `None` in
    /// [SymmetricAMGPreconditioner::current_config].
    config SymmetricAMGPreconditionerConfig, SymmetricAMGPreconditionerConfigBuilder;
    create create_amg_preconditioner, destroy HYPRE_BoomerAMGDestroy;
    positive [strong_threshold, max_row_sum, trunc_factor];
    params {
        /// Number of multigrid cycles per application (1 by default)
        max_iters: usize => HYPRE_BoomerAMGSetMaxIter,
        /// Maximum number of multigrid levels
        max_levels: usize => HYPRE_BoomerAMGSetMaxLevels,
        /// Maximum size of the coarsest grid
        max_coarse_size: usize => HYPRE_BoomerAMGSetMaxCoarseSize,
        /// Minimum size of the coarsest grid
        min_coarse_size: usize => HYPRE_BoomerAMGSetMinCoarseSize,
        /// Strength threshold of the coarsening
        strong_threshold: f64 => HYPRE_BoomerAMGSetStrongThreshold,
        /// Threshold on the off-diagonal row sum for weak dependencies
        max_row_sum: f64 => HYPRE_BoomerAMGSetMaxRowSum,
        /// Truncation factor of the interpolation
        trunc_factor: f64 => HYPRE_BoomerAMGSetTruncFactor,
        /// Maximum number of elements per row of the interpolation
        p_max_elmts: usize => HYPRE_BoomerAMGSetPMaxElmts,
        /// Coarsening algorithm
        coarsen_type: CoarsenType => HYPRE_BoomerAMGSetCoarsenType,
        /// Interpolation operator
        interp_type: InterpType => HYPRE_BoomerAMGSetInterpType,
        /// Smoother on every level but the coarsest
        relax_type: SymmetricSmoother => HYPRE_BoomerAMGSetRelaxType,
        /// Relax C points before F points on the way down, and F points
        /// before C points on the way up (`false`: lexicographic order)
        relax_order: bool => HYPRE_BoomerAMGSetRelaxOrder,
        /// Relaxation weight of the weighted Jacobi smoother
        relax_wt: f64 => HYPRE_BoomerAMGSetRelaxWt,
        /// Number of smoother sweeps on every level
        num_sweeps: usize => HYPRE_BoomerAMGSetNumSweeps,
        /// Multigrid cycle
        cycle_type: CycleType => HYPRE_BoomerAMGSetCycleType,
        /// Number of levels with aggressive coarsening
        agg_num_levels: usize => HYPRE_BoomerAMGSetAggNumLevels,
        /// Logging level
        logging: u32 => HYPRE_BoomerAMGSetLogging,
        /// Print level (0: nothing, 1: setup information)
        print_level: u32 => HYPRE_BoomerAMGSetPrintLevel,
    }
    getters {
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
        logging: HYPRE_Int => HYPRE_BoomerAMGGetLogging,
        print_level: HYPRE_Int => HYPRE_BoomerAMGGetPrintLevel,
    }
}

hypre_solver! {
    /// Algebraic multigrid preconditioner (hypre's BoomerAMG), with every
    /// smoother, for the solvers of general matrices.
    ///
    /// A non-symmetric smoother or relaxation order gives a non-symmetric
    /// preconditioner; use [SymmetricAMGPreconditioner] with the conjugate
    /// gradient solver.
    solver AMGPreconditioner;
    /// Configuration of the algebraic multigrid preconditioner
    ///
    /// See [AMGSolverConfig](crate::solvers::AMGSolverConfig) for the
    /// meaning of the parameters. hypre has no getter for `interp_type`,
    /// `relax_type`, `relax_order`, `relax_wt`, `num_sweeps` and
    /// `agg_num_levels`, which are therefore `None` in
    /// [AMGPreconditioner::current_config].
    config AMGPreconditionerConfig, AMGPreconditionerConfigBuilder;
    create create_amg_preconditioner, destroy HYPRE_BoomerAMGDestroy;
    positive [strong_threshold, max_row_sum, trunc_factor];
    params {
        /// Number of multigrid cycles per application (1 by default)
        max_iters: usize => HYPRE_BoomerAMGSetMaxIter,
        /// Maximum number of multigrid levels
        max_levels: usize => HYPRE_BoomerAMGSetMaxLevels,
        /// Maximum size of the coarsest grid
        max_coarse_size: usize => HYPRE_BoomerAMGSetMaxCoarseSize,
        /// Minimum size of the coarsest grid
        min_coarse_size: usize => HYPRE_BoomerAMGSetMinCoarseSize,
        /// Strength threshold of the coarsening
        strong_threshold: f64 => HYPRE_BoomerAMGSetStrongThreshold,
        /// Threshold on the off-diagonal row sum for weak dependencies
        max_row_sum: f64 => HYPRE_BoomerAMGSetMaxRowSum,
        /// Truncation factor of the interpolation
        trunc_factor: f64 => HYPRE_BoomerAMGSetTruncFactor,
        /// Maximum number of elements per row of the interpolation
        p_max_elmts: usize => HYPRE_BoomerAMGSetPMaxElmts,
        /// Coarsening algorithm
        coarsen_type: CoarsenType => HYPRE_BoomerAMGSetCoarsenType,
        /// Interpolation operator
        interp_type: InterpType => HYPRE_BoomerAMGSetInterpType,
        /// Smoother on every level but the coarsest
        relax_type: Smoother => HYPRE_BoomerAMGSetRelaxType,
        /// Relax C points before F points (`false`: lexicographic order)
        relax_order: bool => HYPRE_BoomerAMGSetRelaxOrder,
        /// Relaxation weight of the weighted Jacobi and SOR smoothers
        relax_wt: f64 => HYPRE_BoomerAMGSetRelaxWt,
        /// Number of smoother sweeps on every level
        num_sweeps: usize => HYPRE_BoomerAMGSetNumSweeps,
        /// Multigrid cycle
        cycle_type: CycleType => HYPRE_BoomerAMGSetCycleType,
        /// Number of levels with aggressive coarsening
        agg_num_levels: usize => HYPRE_BoomerAMGSetAggNumLevels,
        /// Logging level
        logging: u32 => HYPRE_BoomerAMGSetLogging,
        /// Print level (0: nothing, 1: setup information)
        print_level: u32 => HYPRE_BoomerAMGSetPrintLevel,
    }
    getters {
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
        logging: HYPRE_Int => HYPRE_BoomerAMGGetLogging,
        print_level: HYPRE_Int => HYPRE_BoomerAMGGetPrintLevel,
    }
}

impl SymmetricAMGPreconditioner<'_> {
    pub(crate) fn raw(&self) -> RawPreconditioner {
        RawPreconditioner {
            solve: Some(HYPRE_BoomerAMGSolve),
            setup: Some(HYPRE_BoomerAMGSetup),
            solver: self.internal_solver,
        }
    }
}

impl AMGPreconditioner<'_> {
    pub(crate) fn raw(&self) -> RawPreconditioner {
        RawPreconditioner {
            solve: Some(HYPRE_BoomerAMGSolve),
            setup: Some(HYPRE_BoomerAMGSetup),
            solver: self.internal_solver,
        }
    }
}
