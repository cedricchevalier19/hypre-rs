extern crate hypre_rs;
use hypre_rs::matrix::IJMatrix;
use hypre_rs::solvers::{BoomerAMG, PCGSolver, PCGSolverConfigBuilder, SymmetricLinearSolver};
use hypre_rs::vector::IJVector;
use hypre_rs::{Matrix, Vector};
use mpi::topology::Communicator;

fn main() {
    let universe = mpi::initialize().unwrap();
    let mpi_comm = universe.world();

    // Define matrix sizes
    let mesh_size: u32 = 10;
    let global_size = mesh_size * mesh_size;
    let step: u32 = (global_size as i64 / mpi_comm.size() as i64)
        .try_into()
        .unwrap();
    let local_begin: u32 = (mpi_comm.rank() * step as i32).try_into().unwrap();
    let local_end = (local_begin + step).clamp(0u32, global_size);

    // Create a new matrix
    let mut ij_matrix = IJMatrix::new(
        &mpi_comm,
        (local_begin, local_end),
        (local_begin, local_end),
    )
    .unwrap();

    // Fill the matrix
    let mut nnz_buffer = Vec::<(u32, u32, f64)>::with_capacity(5);
    for row in local_begin..local_end {
        if row >= mesh_size {
            nnz_buffer.push((row, row - mesh_size, -1.0));
        }
        if (row % mesh_size) != 0 {
            nnz_buffer.push((row, row - 1, -1.0));
        }
        nnz_buffer.push((row, row, 4.0));
        if ((row + 1) % mesh_size) != 0 {
            nnz_buffer.push((row, row + 1, -1.0));
        }
        if row + mesh_size < global_size {
            nnz_buffer.push((row, row + mesh_size, -1.0));
        }
        ij_matrix
            .add_elements(&mut nnz_buffer.iter().map(|x| *x))
            .unwrap();
        // Allow nnz_buffer to not disappear
        nnz_buffer.clear();
    }

    let mut rhs = IJVector::new(&mpi_comm, (local_begin, local_end)).unwrap();
    rhs.add_elements((local_begin..local_end).map(|i| (i, i as f64)))
        .unwrap();

    let mut x = IJVector::new(&mpi_comm, (local_begin, local_end)).unwrap();
    x.add_elements((local_begin..local_end).map(|i| (i, 0f64)))
        .unwrap();

    // CG solver parameters
    let my_parameters = PCGSolverConfigBuilder::default()
        .tol(1e-9)
        .max_iters(500usize)
        .two_norm(true)
        .recompute_residual_period(8usize)
        .build()
        .unwrap();

    // Create new CG solver with previous parameters
    let mut solver = PCGSolver::new(&mpi_comm, my_parameters).unwrap();

    let precond = BoomerAMG::new();
    solver.set_precond(precond.unwrap());

    let mut c_matrix = Matrix::IJ(ij_matrix);
    let c_rhs = Vector::IJ(rhs);
    let mut c_x = Vector::IJ(x);
    match solver.solve(&mut c_matrix, &c_rhs, &mut c_x) {
        Ok(info) => println!("Solver has converged: {}", info),
        Err(e) => println!("error {:?}", e),
    }
}
