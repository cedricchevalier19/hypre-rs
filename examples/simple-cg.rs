extern crate hypre_rs;
use hypre_rs::matrix::{IJMatrix, NNZ};
use hypre_rs::solvers::{LinearSolver, PCGSolver, PCGSolverConfigBuilder, Solver};
use hypre_rs::vector::IJVector;
use hypre_rs::{Matrix, Vector};
use mpi::initialize;
use mpi::topology::Communicator;

fn main() {
    let universe = mpi::initialize().unwrap();
    let mpi_comm = universe.world();

    // Define matrix sizes
    let global_size: usize = 100;
    // Cannot panic as global_size is properly represented on usize
    let step: usize = (global_size as i64 / mpi_comm.size() as i64)
        .try_into()
        .unwrap();
    let local_begin: usize = mpi_comm.rank() as usize * step;
    let local_end = (local_begin + step).clamp(0usize, global_size);

    // Create a new matrix
    let mut ij_matrix = IJMatrix::new(
        &mpi_comm,
        (local_begin, local_end),
        (local_begin, local_end),
    )
    .unwrap();

    // Fill the matrix on the local diagonal
    ij_matrix
        .add_elements::<u32, f64>((local_begin..local_end).map(|id| NNZ::<u32, f64> {
            row_id: id as u32,
            col_id: id as u32,
            value: 1.0,
        }))
        .unwrap();

    let rhs = Vector::IJ(IJVector::new(&mpi_comm, (local_begin, local_end)).unwrap());
    let mut x = Vector::IJ(IJVector::new(&mpi_comm, (local_begin, local_end)).unwrap());

    // CG solver parameters
    let my_parameters = PCGSolverConfigBuilder::default()
        .tol(1e-9)
        .max_iters(500usize)
        .two_norm(true)
        .recompute_residual_period(8usize)
        .build()
        .unwrap();

    // Create new CG solver with previous parameters
    let solver = Solver::CG(PCGSolver::new(&mpi_comm, my_parameters).unwrap());

    let mut c_matrix = Matrix::IJ(ij_matrix);
    match solver.solve(&mut c_matrix, &rhs, &mut x) {
        Ok(info) => println!("Solver has converged: {}", info),
        Err(e) => println!("error {:?}", e),
    }
}
