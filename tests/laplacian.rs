//! Solves a 1D Laplacian whose exact solution is known, and checks the
//! values of the solution, not just that the solver converged.

use hypre_rs::solvers::{LinearSolver, PCGSolverConfigBuilder, Solver};
use hypre_rs::{HypreError, Matrix, Vector};
use mpi::topology::Communicator;

/// Global number of rows; not a multiple of 2 or 3, so ranks get uneven rows.
const N: u32 = 100;

#[test]
fn laplacian_solution_values() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let (rank, size) = (comm.rank() as u32, comm.size() as u32);
        let rows = (rank * N / size)..((rank + 1) * N / size);

        // tridiag(-1, 2, -1): A * [1, ..., 1] = [1, 0, ..., 0, 1].
        let mut a = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
        let mut entries = Vec::new();
        for i in rows.clone() {
            if i > 0 {
                entries.push((i, i - 1, -1.0));
            }
            entries.push((i, i, 2.0));
            if i + 1 < N {
                entries.push((i, i + 1, -1.0));
            }
        }
        a.add_elements(entries.into_iter())?;
        let mut a = Matrix::IJ(a);

        let mut b = ctx.ij_vector(&comm, rows.clone())?;
        b.add_elements(
            rows.clone()
                .map(|i| (i, if i == 0 || i == N - 1 { 1.0 } else { 0.0 })),
        )?;
        let b = Vector::IJ(b);

        let mut x = ctx.ij_vector(&comm, rows.clone())?;
        x.add_elements(rows.clone().map(|i| (i, 0.0)))?;
        let mut x = Vector::IJ(x);

        // Too few iterations: hypre reports a convergence error.
        let config = PCGSolverConfigBuilder::default()
            .tol(1e-12)
            .max_iters(1usize)
            .build()?;
        let solver = Solver::CG(ctx.pcg_solver(&comm, config)?);
        assert!(matches!(
            solver.solve(&mut a, &b, &mut x),
            Err(HypreError::HypreConvError)
        ));

        // CG converges in at most N iterations on this system.
        let config = PCGSolverConfigBuilder::default()
            .tol(1e-12)
            .max_iters(2 * N as usize)
            .two_norm(true)
            .build()?;
        let solver = Solver::CG(ctx.pcg_solver(&comm, config)?);
        let status = solver.solve(&mut a, &b, &mut x)?;
        assert!(status.converged, "{status}");
        assert!(status.num_iters <= N as usize, "{status}");

        let Vector::IJ(x) = &x;
        let values = x.get_values(rows.clone())?;
        assert_eq!(values.len(), rows.len());
        for (i, v) in rows.zip(values) {
            assert!((v - 1.0).abs() < 1e-8, "x[{i}] = {v}, expected 1");
        }
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
