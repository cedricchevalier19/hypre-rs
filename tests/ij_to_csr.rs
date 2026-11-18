//! Converting an IJMatrix into a CSRMatrix hands the ParCSR object over: it must
//! stay valid once the IJMatrix is consumed, and be destroyed exactly once.

use hypre_rs::matrix::{CSRMatrix, IJMatrix};
use hypre_rs::solvers::{LinearSolver, PCGSolver};
use hypre_rs::vector::IJVector;
use hypre_rs::{Matrix, Vector};
use mpi::topology::Communicator;

#[test]
fn ij_to_csr_conversion_outlives_ij() {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let hypre = hypre_rs::initialize(&universe).unwrap();
    {
        let (rank, size) = (comm.rank() as u32, comm.size() as u32);
        let (b, e) = (rank * 10 / size, (rank + 1) * 10 / size);
        let mut ij = IJMatrix::new(&hypre, &comm, b..e, b..e).unwrap();
        ij.add_elements((b..e).map(|i| (i, i, 2.0f64))).unwrap();
        // `try_from` consumes `ij`: the IJMatrix is destroyed here, while its
        // ParCSR object is handed over to (and later destroyed by) `matrix`.
        let mut matrix = Matrix::ParCSR(CSRMatrix::try_from(ij).unwrap());

        let mut rhs = hypre.ij_vector(&comm, b..e).unwrap();
        rhs.add_elements((b..e).map(|i| (i, 2.0f64))).unwrap();
        let mut x = IJVector::new(&hypre, &comm, b..e).unwrap();
        x.add_elements((b..e).map(|i| (i, 0.0f64))).unwrap();

        let solver = PCGSolver::new(&hypre, &comm, Default::default()).unwrap();
        let status = solver
            .solve(&mut matrix, &Vector::IJ(rhs), &mut Vector::IJ(x))
            .unwrap();
        assert!(status.converged);
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    hypre.finalize().unwrap();
}
