extern crate hypre_rs;
use hypre_rs::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let mpi_comm = universe.world();

    let ctx = hypre_rs::initialize(&universe)?;

    // Define matrix sizes
    let mesh_size: u32 = 10;
    let global_size = mesh_size * mesh_size;
    // Rows owned by this MPI rank, in balanced contiguous blocks.
    let local_rows = distribute_rows(&mpi_comm, global_size);

    // Create a new symmetric matrix
    let mut matrix = ctx.symmetric_matrix(&mpi_comm, local_rows.clone())?;

    // Fill the lower triangle of the 2D Laplacian; the upper triangle is
    // added by symmetry.
    let mut nnz_buffer = Vec::<(u32, u32, f64)>::with_capacity(3);
    for row in local_rows.clone() {
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

    let rhs = ParVector::from_fn(&ctx, &mpi_comm, local_rows.clone(), |i| i as f64)?;

    let mut x = ParVector::zeros(&ctx, &mpi_comm, local_rows.clone())?;

    // CG solver parameters
    let my_parameters = PCGSolverConfigBuilder::default()
        .tol(1e-9)
        .max_iters(500)
        .two_norm(true)
        .recompute_residual_period(8)
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
