extern crate hypre_rs;
use hypre_rs::solvers::{LinearSolver, PCGSolverConfigBuilder, Solver};
use hypre_rs::{Matrix, Vector};
use mpi::topology::Communicator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let mpi_comm = universe.world();

    let ctx = hypre_rs::initialize(&universe)?;

    // Define matrix sizes
    let mesh_size: u32 = 10;
    let global_size = mesh_size * mesh_size;
    // Split rows evenly; the first ranks get one extra row when the size
    // does not divide evenly.
    let (rank, size) = (mpi_comm.rank() as u64, mpi_comm.size() as u64);
    let local_begin: u32 = (rank * global_size as u64 / size).try_into()?;
    let local_end: u32 = ((rank + 1) * global_size as u64 / size).try_into()?;
    let local_rows = local_begin..local_end;

    // Create a new matrix
    let mut ij_matrix = ctx.ij_matrix(&mpi_comm, local_rows.clone(), local_rows.clone())?;

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
        ij_matrix.add_elements(&mut nnz_buffer.iter().map(|x| *x))?;
        // Allow nnz_buffer to not disappear
        nnz_buffer.clear();
    }

    let mut rhs = ctx.ij_vector(&mpi_comm, local_rows.clone())?;
    rhs.add_elements((local_begin..local_end).map(|i| (i, i as f64)))?;

    let mut x = ctx.ij_vector(&mpi_comm, local_rows.clone())?;
    x.add_elements((local_begin..local_end).map(|i| (i, 0f64)))?;

    // CG solver parameters
    let my_parameters = PCGSolverConfigBuilder::default()
        .tol(1e-9)
        .max_iters(500usize)
        .two_norm(true)
        .recompute_residual_period(8usize)
        .build()?;

    // Create new CG solver with previous parameters
    let solver = Solver::CG(ctx.pcg_solver(&mpi_comm, my_parameters)?);

    let mut c_matrix = Matrix::IJ(ij_matrix);
    let c_rhs = Vector::IJ(rhs);
    let mut c_x = Vector::IJ(x);
    let info = solver.solve(&mut c_matrix, &c_rhs, &mut c_x)?;
    println!("Solver has converged: {}", info);

    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    drop((solver, c_matrix, c_rhs, c_x));
    ctx.finalize()?;
    Ok(())
}
