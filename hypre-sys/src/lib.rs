#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use mpi_sys::*;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Fallback for hypre < 2.29, which only provides `HYPRE_Init`.
///
/// # Safety
/// Same contract as the C function: call once, before any other hypre call.
#[cfg(not(hypre_has_initialize))]
pub unsafe fn HYPRE_Initialize() -> HYPRE_Int {
    HYPRE_Init()
}

// Declared in hypre's internal `_hypre_parcsr_ls.h` only, before 2.29.
#[cfg(not(hypre_has_amg_getters))]
extern "C" {
    pub fn HYPRE_BoomerAMGGetTol(solver: HYPRE_Solver, tol: *mut HYPRE_Real) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetMaxIter(solver: HYPRE_Solver, max_iter: *mut HYPRE_Int) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetMaxLevels(
        solver: HYPRE_Solver,
        max_levels: *mut HYPRE_Int,
    ) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetMaxCoarseSize(
        solver: HYPRE_Solver,
        max_coarse_size: *mut HYPRE_Int,
    ) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetMinCoarseSize(
        solver: HYPRE_Solver,
        min_coarse_size: *mut HYPRE_Int,
    ) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetStrongThreshold(
        solver: HYPRE_Solver,
        strong_threshold: *mut HYPRE_Real,
    ) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetMaxRowSum(
        solver: HYPRE_Solver,
        max_row_sum: *mut HYPRE_Real,
    ) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetTruncFactor(
        solver: HYPRE_Solver,
        trunc_factor: *mut HYPRE_Real,
    ) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetPMaxElmts(
        solver: HYPRE_Solver,
        p_max_elmts: *mut HYPRE_Int,
    ) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetCoarsenType(
        solver: HYPRE_Solver,
        coarsen_type: *mut HYPRE_Int,
    ) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetCycleType(
        solver: HYPRE_Solver,
        cycle_type: *mut HYPRE_Int,
    ) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetConvergeType(solver: HYPRE_Solver, type_: *mut HYPRE_Int)
        -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetLogging(solver: HYPRE_Solver, logging: *mut HYPRE_Int) -> HYPRE_Int;
    pub fn HYPRE_BoomerAMGGetPrintLevel(
        solver: HYPRE_Solver,
        print_level: *mut HYPRE_Int,
    ) -> HYPRE_Int;
}
