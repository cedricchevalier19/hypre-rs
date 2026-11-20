//! Krylov solvers for general matrices, on a non-symmetric system whose exact
//! solution is known, on any number of MPI ranks.

use hypre_rs::solvers::{
    BiCGSTABSolverConfigBuilder, COGMRESSolverConfigBuilder, FlexGMRESSolverConfigBuilder,
    GMRESSolverConfigBuilder, LGMRESSolverConfigBuilder, Solver,
};
use hypre_rs::vector::ParVector;
use hypre_rs::{HypreContext, HypreError, Matrix};
use mpi::topology::{Communicator, SimpleCommunicator};
use std::ops::Range;

/// Global number of rows; not a multiple of 2 or 3, so ranks get uneven rows.
const N: u32 = 50;

/// Below and above the diagonal of the convection-diffusion matrix
/// `tridiag(LOWER, 2, UPPER)`, which is not symmetric.
const LOWER: f64 = -1.5;
const UPPER: f64 = -0.5;

/// `A * [1, ..., 1]` for the convection-diffusion matrix.
fn row_sum(i: u32) -> f64 {
    2.0 + if i > 0 { LOWER } else { 0.0 } + if i + 1 < N { UPPER } else { 0.0 }
}

/// Solvers that converge to a tolerance of `tol` within `max_iters`.
fn solvers<'a>(
    ctx: &'a HypreContext<'_>,
    comm: &'a SimpleCommunicator,
    tol: f64,
    max_iters: usize,
) -> Result<Vec<Solver<'a>>, HypreError> {
    Ok(vec![
        ctx.gmres_solver(
            comm,
            GMRESSolverConfigBuilder::default()
                .tol(tol)
                .max_iters(max_iters)
                .k_dim(10usize)
                .build()?,
        )?
        .into(),
        ctx.flex_gmres_solver(
            comm,
            FlexGMRESSolverConfigBuilder::default()
                .tol(tol)
                .max_iters(max_iters)
                .build()?,
        )?
        .into(),
        ctx.lgmres_solver(
            comm,
            LGMRESSolverConfigBuilder::default()
                .tol(tol)
                .max_iters(max_iters)
                .k_dim(10usize)
                .aug_dim(2usize)
                .build()?,
        )?
        .into(),
        ctx.cogmres_solver(
            comm,
            COGMRESSolverConfigBuilder::default()
                .tol(tol)
                .max_iters(max_iters)
                .build()?,
        )?
        .into(),
        ctx.bicgstab_solver(
            comm,
            BiCGSTABSolverConfigBuilder::default()
                .tol(tol)
                .max_iters(max_iters)
                .build()?,
        )?
        .into(),
    ])
}

/// Solves `a * x = b` from `x = 0` and checks that `x = [1, ..., 1]`.
fn check_solution(
    ctx: &HypreContext<'_>,
    comm: &SimpleCommunicator,
    rows: Range<u32>,
    solver: &Solver<'_>,
    a: &(impl Matrix + ?Sized),
    b: &ParVector<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut x = ctx.ij_vector(comm, rows.clone())?;
    x.add_elements(rows.clone().map(|i| (i, 0.0)))?;
    let mut x = x.assemble()?;
    let status = solver.solve(a, b, &mut x)?;
    assert!(status.converged, "{solver:?}: {status}");
    for (i, v) in rows.clone().zip(x.get_values(rows)?) {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "{solver:?}: x[{i}] = {v}, expected 1"
        );
    }
    Ok(())
}

#[test]
fn krylov_solvers() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let (rank, size) = (comm.rank() as u32, comm.size() as u32);
        let rows = (rank * N / size)..((rank + 1) * N / size);

        let mut ij = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
        for i in rows.clone() {
            ij.add_elements([(i, i, 2.0)].into_iter())?;
            if i > 0 {
                ij.add_elements([(i, i - 1, LOWER)].into_iter())?;
            }
            if i + 1 < N {
                ij.add_elements([(i, i + 1, UPPER)].into_iter())?;
            }
        }
        let a = ij.assemble()?;
        assert!(!a.is_symmetric(0.0)?);

        let mut b = ctx.ij_vector(&comm, rows.clone())?;
        b.add_elements(rows.clone().map(|i| (i, row_sum(i))))?;
        let b = b.assemble()?;

        // Every solver finds the exact solution.
        for solver in solvers(&ctx, &comm, 1e-12, 10 * N as usize)? {
            check_solution(&ctx, &comm, rows.clone(), &solver, &a, &b)?;
        }

        // Too few iterations: hypre reports a convergence error.
        for solver in solvers(&ctx, &comm, 1e-12, 1)? {
            let mut x = ctx.ij_vector(&comm, rows.clone())?.assemble()?;
            assert!(
                matches!(
                    solver.solve(&a, &b, &mut x),
                    Err(HypreError::HypreConvError)
                ),
                "{solver:?}"
            );
        }
        drop(a);

        // A symmetric matrix is a Matrix: general solvers accept it.
        // tridiag(-1, 2, -1) * [1, ..., 1] = [1, 0, ..., 0, 1].
        let mut s = ctx.symmetric_matrix(&comm, rows.clone())?;
        s.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
        s.add_elements(rows.clone().filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
        let s = s.assemble()?;
        let mut b = ctx.ij_vector(&comm, rows.clone())?;
        b.add_elements(
            rows.clone()
                .map(|i| (i, if i == 0 || i == N - 1 { 1.0 } else { 0.0 })),
        )?;
        let b = b.assemble()?;
        let gmres: Solver<'_> = ctx
            .gmres_solver(
                &comm,
                GMRESSolverConfigBuilder::default()
                    // Without restarts, GMRES converges in at most N iterations.
                    .k_dim(N as usize)
                    .tol(1e-12)
                    .build()?,
            )?
            .into();
        check_solution(&ctx, &comm, rows.clone(), &gmres, &s, &b)?;
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
