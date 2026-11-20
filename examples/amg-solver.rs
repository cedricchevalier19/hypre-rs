//! Algebraic multigrid (BoomerAMG) as a standalone solver.
//!
//! Configures AMG with named options, reads the configuration back from
//! hypre, handles a solve that does not converge within its iteration
//! limit, and continues from the iterate it reached.
//!
//! ```bash
//! cargo run --example amg-solver
//! mpirun -n 4 target/debug/examples/amg-solver
//! ```

use hypre_rs::prelude::*;
use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::topology::{Communicator, SimpleCommunicator};

/// Grid size: the matrix has `M * M` rows.
const M: u32 = 64;

/// Largest `|x_i - 1|` over all MPI ranks.
fn max_error(comm: &SimpleCommunicator, x: &ParVector<'_>) -> Result<f64, HypreError> {
    let local = x
        .local_values()?
        .iter()
        .fold(0.0_f64, |m, v| m.max((v - 1.0).abs()));
    let mut global = 0.0;
    comm.all_reduce_into(&local, &mut global, SystemOperation::max());
    Ok(global)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let root = comm.rank() == 0;
    let ctx = hypre_rs::initialize(&universe)?;

    // The 2D Poisson problem (5-point Laplacian), from its lower triangle.
    let rows = distribute_rows(&comm, M * M);
    let mut a = ctx.symmetric_matrix_builder(&comm, rows.clone())?;
    for row in rows.clone() {
        a.add([(row, row, 4.0)])?;
        if row % M > 0 {
            a.add([(row, row - 1, -1.0)])?;
        }
        if row >= M {
            a.add([(row, row - M, -1.0)])?;
        }
    }
    let a = a.assemble()?;
    // b = A * [1, ..., 1]: the number of missing neighbors of each row.
    let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |row| {
        let (i, j) = (row / M, row % M);
        [i == 0, i == M - 1, j == 0, j == M - 1]
            .into_iter()
            .filter(|&on_boundary| on_boundary)
            .count() as f64
    })?;

    // A deliberately low cycle limit, to show what happens when it is hit.
    let config = AMGSolverConfigBuilder::default()
        .tol(1e-10)
        .max_iters(3)
        .coarsen_type(CoarsenType::Pmis)
        .interp_type(InterpType::ExtendedPlusI)
        .relax_type(Smoother::L1SymmetricGaussSeidel)
        .num_sweeps(1)
        .cycle_type(CycleType::V)
        .converge_type(ConvergeType::RelativeToRhs)
        .build()?;
    let solver = ctx.amg_solver(&comm, config)?;
    let current = solver.current_config()?;
    if root {
        println!(
            "AMG: {:?} coarsening, {:?}-cycle, tolerance {:?}",
            current.coarsen_type, current.cycle_type, current.tol
        );
    }

    let mut x = ParVector::zeros(&ctx, &comm, rows)?;
    match solver.solve(&a, &b, &mut x) {
        Ok(status) => return Err(format!("expected no convergence, got {status}").into()),
        // `x` holds the last iterate: the next solve starts from it.
        Err(HypreError::NotConverged(status)) => {
            if root {
                println!("first attempt: {status}");
            }
        }
        Err(error) => return Err(error.into()),
    }

    // Raise the limit on the same solver; unset fields keep their values.
    let solver = solver.config(AMGSolverConfigBuilder::default().max_iters(50).build()?)?;
    let status = solver.solve(&a, &b, &mut x)?;
    let error = max_error(&comm, &x)?;
    if root {
        println!("second attempt: {status}");
        println!("max error {error:.1e}");
    }
    if error > 1e-6 {
        return Err(format!("max error {error} is too large").into());
    }

    // hypre objects must be dropped before hypre is finalized (checked by
    // the compiler).
    drop((solver, a, b, x));
    ctx.finalize()?;
    Ok(())
}
