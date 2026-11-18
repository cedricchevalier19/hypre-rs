//! Converting an IJMatrix into a CSRMatrix hands the ParCSR object over: it must
//! stay valid once the IJMatrix is consumed, and be destroyed exactly once.

use hypre_rs::matrix::{CSRMatrix, IJMatrix};
use hypre_rs::solvers::{LinearSolver, PCGSolver};
use hypre_rs::vector::IJVector;
use hypre_rs::{Matrix, Vector};
use mpi::topology::Communicator;

#[test]
fn ij_to_csr_conversion_outlives_ij() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let (rank, size) = (comm.rank() as u32, comm.size() as u32);
        let (b, e) = (rank * 10 / size, (rank + 1) * 10 / size);
        let mut ij = IJMatrix::new(&ctx, &comm, b..e, b..e)?;
        ij.add_elements((b..e).map(|i| (i, i, 2.0f64)))?;
        // `try_from` consumes `ij`: the IJMatrix is destroyed here, while its
        // ParCSR object is handed over to (and later destroyed by) `matrix`.
        let mut matrix = Matrix::ParCSR(CSRMatrix::try_from(ij)?);

        let mut rhs = ctx.ij_vector(&comm, b..e)?;
        rhs.add_elements((b..e).map(|i| (i, 2.0f64)))?;
        let mut x = IJVector::new(&ctx, &comm, b..e)?;
        x.add_elements((b..e).map(|i| (i, 0.0f64)))?;
        let mut x = Vector::IJ(x);

        let solver = PCGSolver::new(&ctx, &comm, Default::default())?;
        let status = solver.solve(&mut matrix, &Vector::IJ(rhs), &mut x)?;
        assert!(status.converged);
        // 2 * x = 2
        let Vector::IJ(x) = &x;
        for v in x.get_values(b..e)? {
            assert!((v - 1.0).abs() < 1e-12, "x = {v}, expected 1");
        }
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
