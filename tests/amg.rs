//! The algebraic multigrid solver (BoomerAMG), on any number of MPI ranks.

use hypre_rs::solvers::{AMGSolver, AMGSolverConfigBuilder, LinearSolver, Solver};
use hypre_rs::{HypreContext, HypreError, Matrix, Vector};
use mpi::topology::{Communicator, SimpleCommunicator};
use std::ops::Range;

/// Global number of rows; not a multiple of 2 or 3, so ranks get uneven rows.
const N: u32 = 100;

/// `[1, 0, ..., 0, 1]`, which is `A * [1, ..., 1]` for the 1D Laplacian
/// `tridiag(-1, 2, -1)`.
fn rhs<'a>(
    ctx: &'a HypreContext<'_>,
    comm: &'a SimpleCommunicator,
    rows: Range<u32>,
) -> Result<Vector<'a>, HypreError> {
    let mut b = ctx.ij_vector(comm, rows.clone())?;
    b.add_elements(rows.map(|i| (i, if i == 0 || i == N - 1 { 1.0 } else { 0.0 })))?;
    Ok(Vector::IJ(b))
}

/// Solves `a * x = b` from `x = 0` and checks that `x = [1, ..., 1]`.
fn check_solution(
    ctx: &HypreContext<'_>,
    comm: &SimpleCommunicator,
    rows: Range<u32>,
    solver: &Solver<'_>,
    a: &Matrix<'_>,
    b: &Vector<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut x = ctx.ij_vector(comm, rows.clone())?;
    x.add_elements(rows.clone().map(|i| (i, 0.0)))?;
    let mut x = Vector::IJ(x);
    let status = solver.solve(a, b, &mut x)?;
    assert!(status.converged, "{status}");
    assert!(status.res_num <= 1e-10, "{status}");
    let Vector::IJ(x) = &x;
    for (i, v) in rows.clone().zip(x.get_values(rows)?) {
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
        let (rank, size) = (comm.rank() as u32, comm.size() as u32);
        let rows = (rank * N / size)..((rank + 1) * N / size);

        // The 1D Laplacian, as a SymmetricMatrix: AMG accepts it through Deref.
        let mut a = ctx.symmetric_matrix(&comm, rows.clone())?;
        a.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
        a.add_elements(rows.clone().filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
        let b = rhs(&ctx, &comm, rows.clone())?;

        // Default AMG setup, and a few common variations.
        let configs = [
            AMGSolverConfigBuilder::default(),
            *AMGSolverConfigBuilder::default()
                .coarsen_type(6u32) // Falgout
                .interp_type(0u32) // classical
                .relax_type(6u32) // symmetric hybrid Gauss-Seidel
                .num_sweeps(2usize),
            *AMGSolverConfigBuilder::default()
                .coarsen_type(8u32) // PMIS
                .agg_num_levels(1usize)
                .cycle_type(2u32), // W-cycle
        ];
        for mut config in configs {
            let config = config.tol(1e-10).max_iters(100usize).build()?;
            let solver = Solver::AMG(ctx.amg_solver(&comm, config)?);
            check_solution(&ctx, &comm, rows.clone(), &solver, &a, &b)?;
        }

        // Too few cycles: hypre reports a convergence error.
        let config = AMGSolverConfigBuilder::default()
            .tol(1e-10)
            .max_iters(1usize)
            .build()?;
        let solver = Solver::AMG(ctx.amg_solver(&comm, config)?);
        let mut x = Vector::IJ(ctx.ij_vector(&comm, rows.clone())?);
        assert!(matches!(
            solver.solve(&a, &b, &mut x),
            Err(HypreError::HypreConvError)
        ));
    }
    {
        // Every field that hypre can read back round-trips.
        let config = AMGSolverConfigBuilder::default()
            .tol(1e-9)
            .max_iters(42usize)
            .max_levels(7usize)
            .max_coarse_size(20usize)
            .min_coarse_size(2usize)
            .strong_threshold(0.5)
            .max_row_sum(0.8)
            .trunc_factor(0.1)
            .p_max_elmts(3usize)
            .coarsen_type(8u32)
            .cycle_type(2u32)
            .converge_type(1u32)
            .logging(1u32)
            .print_level(0u32)
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
        assert_eq!(current.coarsen_type, Some(8));
        assert_eq!(current.cycle_type, Some(2));
        assert_eq!(current.converge_type, Some(1));
        assert_eq!(current.logging, Some(1));
        assert_eq!(current.print_level, Some(0));
        // hypre has no getter for these.
        assert_eq!(current.min_iters, None);
        assert_eq!(current.relax_type, None);
        assert_eq!(current.agg_num_levels, None);

        // `config` changes an existing solver; unset fields keep their value.
        let update = AMGSolverConfigBuilder::default()
            .tol(1e-3)
            .cycle_type(1u32)
            .build()?;
        let solver = solver.config(update)?;
        let current = solver.current_config()?;
        assert_eq!(current.tol, Some(1e-3));
        assert_eq!(current.cycle_type, Some(1));
        assert_eq!(current.max_iters, Some(42));
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
