//! Solves a 1D Laplacian whose exact solution is known, and checks the
//! values of the solution, not just that the solver converged.

use hypre_rs::solvers::PCGSolverConfigBuilder;
use hypre_rs::{HypreError, ParVector};

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
        let rows = hypre_rs::distribute_rows(&comm, N);

        // tridiag(-1, 2, -1): A * [1, ..., 1] = [1, 0, ..., 0, 1].
        // Only the lower triangle is given; the super-diagonal comes from
        // symmetry, and for the first local row it belongs to the previous
        // rank. A wrong mirroring would give a wrong solution below.
        let mut a = ctx.symmetric_matrix(&comm, rows.clone())?;
        let mut entries = Vec::new();
        for i in rows.clone() {
            if i > 0 {
                entries.push((i, i - 1, -1.0));
            }
            entries.push((i, i, 2.0));
        }
        a.add_elements(entries)?;
        let a = a.assemble()?;

        let mut b = ctx.ij_vector(&comm, rows.clone())?;
        b.add_elements(
            rows.clone()
                .map(|i| (i, if i == 0 || i == N - 1 { 1.0 } else { 0.0 })),
        )?;
        let b = b.assemble()?;

        let mut x = ParVector::zeros(&ctx, &comm, rows.clone())?;

        // Too few iterations: hypre reports a convergence error.
        let config = PCGSolverConfigBuilder::default()
            .tol(1e-12)
            .max_iters(1)
            .build()?;
        let solver = ctx.pcg_solver(&comm, config)?;
        // The error keeps the status: one iteration, residual above the
        // tolerance.
        let result = solver.solve(&a, &b, &mut x);
        assert!(
            matches!(
                result,
                Err(HypreError::NotConverged(status))
                    if !status.converged && status.num_iters == 1 && status.res_num > 1e-12
            ),
            "{result:?}"
        );

        // In exact arithmetic CG converges in at most N iterations; a solve
        // that does not converge within max_iters returns NotConverged.
        let config = PCGSolverConfigBuilder::default()
            .tol(1e-12)
            .max_iters(N as usize)
            .two_norm(true)
            .build()?;
        let solver = ctx.pcg_solver(&comm, config)?;
        let status = solver.solve(&a, &b, &mut x)?;
        assert!(status.converged, "{status}");

        let values = x.local_values()?;
        assert_eq!(values.len(), rows.len());
        for (i, v) in rows.zip(values) {
            assert!((v - 1.0).abs() < 1e-8, "x[{i}] = {v}, expected 1");
        }
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
