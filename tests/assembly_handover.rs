//! Assembling a builder hands hypre's ParCSR matrix or ParVector over: it
//! must stay valid once the builder's IJ object is destroyed, and be
//! destroyed exactly once.

use hypre_rs::matrix::{DistributedMatrixBuilder, SymmetricMatrixBuilder};
use hypre_rs::solvers::PCGSolver;
use hypre_rs::vector::DistributedVectorBuilder;

#[test]
fn assembly_outlives_the_builder() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let rows = hypre_rs::distribute_rows(&comm, 10);
        // `assemble` consumes the builder: its IJ matrix is destroyed there,
        // while the ParCSR object is handed over to (and later destroyed
        // by) `csr`.
        let mut ij = DistributedMatrixBuilder::new(&ctx, &comm, rows.clone(), rows.clone())?;
        ij.add(rows.clone().map(|i| (i, i, 2.0)))?;
        let csr = ij.assemble()?;
        drop(csr);

        // Same through a symmetric matrix, which CG accepts, and vectors.
        let mut matrix = SymmetricMatrixBuilder::new(&ctx, &comm, rows.clone())?;
        matrix.add(rows.clone().map(|i| (i, i, 2.0)))?;
        let matrix = matrix.assemble()?;
        let mut rhs = DistributedVectorBuilder::new(&ctx, &comm, rows.clone())?;
        rhs.add(rows.clone().map(|i| (i, 2.0)))?;
        let rhs = rhs.assemble()?;
        let mut x = DistributedVectorBuilder::new(&ctx, &comm, rows.clone())?.assemble()?;

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
