//! The algebraic multigrid solver (BoomerAMG), on any number of MPI ranks.

use hypre_rs::solvers::{
    AMGSolver, AMGSolverConfigBuilder, CoarsenType, ConvergeType, CycleType, InterpType, Smoother,
};
use hypre_rs::vector::ParVector;
use hypre_rs::{HypreContext, HypreError, Matrix};
use mpi::topology::SimpleCommunicator;
use std::ops::Range;

/// Global number of rows; not a multiple of 2 or 3, so ranks get uneven rows.
const N: u32 = 100;

/// `[1, 0, ..., 0, 1]`, which is `A * [1, ..., 1]` for the 1D Laplacian
/// `tridiag(-1, 2, -1)`.
fn rhs<'a>(
    ctx: &'a HypreContext<'_>,
    comm: &'a SimpleCommunicator,
    rows: Range<u32>,
) -> Result<ParVector<'a>, HypreError> {
    let b = ParVector::from_fn(ctx, comm, rows.clone(), |i| {
        if i == 0 || i == N - 1 {
            1.0
        } else {
            0.0
        }
    })?;
    Ok(b)
}

/// Solves `a * x = b` from `x = 0` and checks that `x = [1, ..., 1]`.
fn check_solution(
    ctx: &HypreContext<'_>,
    comm: &SimpleCommunicator,
    rows: Range<u32>,
    solver: &AMGSolver<'_>,
    a: &(impl Matrix + ?Sized),
    b: &ParVector<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut x = ParVector::zeros(ctx, comm, rows.clone())?;
    let status = solver.solve(a, b, &mut x)?;
    assert!(status.converged, "{status}");
    assert!(status.res_num <= 1e-10, "{status}");
    for (i, v) in rows.zip(x.local_values()?) {
        assert!((v - 1.0).abs() < 1e-6, "x[{i}] = {v}, expected 1");
    }
    Ok(())
}

#[test]
fn amg_solver() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let rows = hypre_rs::distribute_rows(&comm, N);

        // The 1D Laplacian, as a SymmetricMatrix: every solver accepts it.
        let mut a = ctx.symmetric_matrix_builder(&comm, rows.clone())?;
        a.add(rows.clone().map(|i| (i, i, 2.0)))?;
        a.add(rows.clone().filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
        let a = a.assemble()?;
        let b = rhs(&ctx, &comm, rows.clone())?;

        // Default AMG setup, and a few common variations.
        let configs = [
            AMGSolverConfigBuilder::default(),
            *AMGSolverConfigBuilder::default()
                .coarsen_type(CoarsenType::Falgout)
                .interp_type(InterpType::ClassicalModified)
                .relax_type(Smoother::HybridSymmetricGaussSeidel)
                .num_sweeps(2),
            *AMGSolverConfigBuilder::default()
                .coarsen_type(CoarsenType::Pmis)
                .agg_num_levels(1)
                .cycle_type(CycleType::W),
        ];
        for mut config in configs {
            let config = config.tol(1e-10).max_iters(100).build()?;
            let solver = ctx.amg_solver(&comm, config)?;
            check_solution(&ctx, &comm, rows.clone(), &solver, &a, &b)?;
        }

        // Too few cycles: hypre reports a convergence error.
        let config = AMGSolverConfigBuilder::default()
            .tol(1e-10)
            .max_iters(1)
            .build()?;
        let solver = ctx.amg_solver(&comm, config)?;
        let mut x = ParVector::zeros(&ctx, &comm, rows.clone())?;
        // The error keeps the status: one cycle, residual above the tolerance.
        let result = solver.solve(&a, &b, &mut x);
        assert!(
            matches!(
                result,
                Err(HypreError::NotConverged(status))
                    if !status.converged && status.num_iters == 1 && status.res_num > 1e-10
            ),
            "{result:?}"
        );
    }
    {
        // Every field that hypre can read back round-trips.
        let config = AMGSolverConfigBuilder::default()
            .tol(1e-9)
            .max_iters(42)
            .max_levels(7)
            .max_coarse_size(20)
            .min_coarse_size(2)
            .strong_threshold(0.5)
            .max_row_sum(0.8)
            .trunc_factor(0.1)
            .p_max_elmts(3)
            .coarsen_type(CoarsenType::Pmis)
            .cycle_type(CycleType::W)
            .converge_type(ConvergeType::RelativeToInitialResidual)
            .logging(1)
            .print_level(0)
            .build()?;
        let solver = AMGSolver::new(&ctx, &comm, config)?;
        let current = solver.current_config()?;
        assert_eq!(current.tol, Some(1e-9));
        assert_eq!(current.max_iters, Some(42));
        assert_eq!(current.max_levels, Some(7));
        assert_eq!(current.max_coarse_size, Some(20));
        assert_eq!(current.min_coarse_size, Some(2));
        assert_eq!(current.strong_threshold, Some(0.5));
        assert_eq!(current.max_row_sum, Some(0.8));
        assert_eq!(current.trunc_factor, Some(0.1));
        assert_eq!(current.p_max_elmts, Some(3));
        assert_eq!(current.coarsen_type, Some(CoarsenType::Pmis));
        assert_eq!(current.cycle_type, Some(CycleType::W));
        assert_eq!(
            current.converge_type,
            Some(ConvergeType::RelativeToInitialResidual)
        );
        assert_eq!(current.logging, Some(1));
        assert_eq!(current.print_level, Some(0));
        // hypre has no getter for these.
        assert_eq!(current.min_iters, None);
        assert_eq!(current.relax_type, None);
        assert_eq!(current.agg_num_levels, None);

        // `config` changes an existing solver; unset fields keep their value.
        let update = AMGSolverConfigBuilder::default()
            .tol(1e-3)
            .cycle_type(CycleType::V)
            .build()?;
        let solver = solver.config(update)?;
        let current = solver.current_config()?;
        assert_eq!(current.tol, Some(1e-3));
        assert_eq!(current.cycle_type, Some(CycleType::V));
        assert_eq!(current.max_iters, Some(42));
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
