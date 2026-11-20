//! Converting an IJMatrix into a CSRMatrix hands the ParCSR object over: it must
//! stay valid once the IJMatrix is consumed, and be destroyed exactly once.

use hypre_rs::matrix::{CSRMatrix, IJMatrix, SymmetricMatrix};
use hypre_rs::solvers::PCGSolver;
use hypre_rs::vector::IJVector;
use hypre_rs::ParVector;

#[test]
fn ij_to_csr_conversion_outlives_ij() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let rows = hypre_rs::distribute_rows(&comm, 10);
        let (b, e) = (rows.start, rows.end);
        let mut ij = IJMatrix::new(&ctx, &comm, b..e, b..e)?;
        ij.add_elements((b..e).map(|i| (i, i, 2.0f64)))?;
        // `try_from` consumes `ij`: the IJMatrix is destroyed here, while its
        // ParCSR object is handed over to (and later destroyed by) `csr`.
        let csr = CSRMatrix::try_from(ij)?;
        drop(csr);

        // Same through a symmetric matrix, which CG accepts.
        let mut matrix = SymmetricMatrix::new(&ctx, &comm, b..e)?;
        matrix.add_elements((b..e).map(|i| (i, i, 2.0f64)))?;
        // Solvers need it assembled; it then has no `add_elements` (checked
        // by the compiler; see the `SymmetricMatrix::assemble` doctest).
        let matrix = matrix.assemble()?;

        let rhs = ParVector::from_fn(&ctx, &comm, b..e, |_| 2.0f64)?;
        let mut x = IJVector::new(&ctx, &comm, b..e)?;
        x.add_elements((b..e).map(|i| (i, 0.0f64)))?;
        let mut x = x.assemble()?;

        let solver = PCGSolver::new(&ctx, &comm, Default::default())?;
        let status = solver.solve(&matrix, &rhs, &mut x)?;
        assert!(status.converged);
        // A = 2 * I and b = [2, ..., 2], so the exact solution is x = [1, ..., 1].
        for v in x.local_values()? {
            assert!((v - 1.0).abs() < 1e-12, "x = {v}, expected 1");
        }
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
