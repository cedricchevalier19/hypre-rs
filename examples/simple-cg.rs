extern crate hypre_rs;
use hypre_rs::solvers::PCGSolverConfigBuilder;
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

    // Create a new symmetric matrix
    let mut matrix = ctx.symmetric_matrix(&mpi_comm, local_rows.clone())?;

    // Fill the lower triangle of the 2D Laplacian; the upper triangle is
    // added by symmetry.
    let mut nnz_buffer = Vec::<(u32, u32, f64)>::with_capacity(3);
    for row in local_begin..local_end {
        if row >= mesh_size {
            nnz_buffer.push((row, row - mesh_size, -1.0));
        }
        if (row % mesh_size) != 0 {
            nnz_buffer.push((row, row - 1, -1.0));
        }
        nnz_buffer.push((row, row, 4.0));
        matrix.add_elements(nnz_buffer.iter().copied())?;
        // Allow nnz_buffer to not disappear
        nnz_buffer.clear();
    }

    // Solvers need an assembled matrix; it cannot be modified afterwards.
    let matrix = matrix.assemble()?;

    let mut rhs = ctx.ij_vector(&mpi_comm, local_rows.clone())?;
    rhs.add_elements((local_begin..local_end).map(|i| (i, i as f64)))?;
    let rhs = rhs.assemble()?;

    let mut x = ctx.ij_vector(&mpi_comm, local_rows.clone())?;
    x.add_elements((local_begin..local_end).map(|i| (i, 0f64)))?;
    let mut x = x.assemble()?;

    // CG solver parameters
    let my_parameters = PCGSolverConfigBuilder::default()
        .tol(1e-9)
        .max_iters(500usize)
        .two_norm(true)
        .recompute_residual_period(8usize)
        .build()?;

    // Create new CG solver with previous parameters
    let solver = ctx.pcg_solver(&mpi_comm, my_parameters)?;

    let info = solver.solve(&matrix, &rhs, &mut x)?;
    println!("Solver has converged: {}", info);

    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    drop((solver, matrix, rhs, x));
    ctx.finalize()?;
    Ok(())
}
