//! Preconditioned conjugate gradient on a 2D diffusion problem.
//!
//! Discretizes `-div(k grad u)` on an `M x M` grid (5 points, Dirichlet
//! boundary), with a coefficient `k` that jumps from 1 to 100 halfway
//! through the domain, and solves it with CG three times: without
//! preconditioner, with Jacobi, and with algebraic multigrid (BoomerAMG).
//! The iteration counts show what each preconditioner buys.
//!
//! ```bash
//! cargo run --example pcg-preconditioners
//! mpirun -n 4 target/debug/examples/pcg-preconditioners
//! ```

use hypre_rs::prelude::*;
use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::topology::{Communicator, SimpleCommunicator};
use std::ops::Range;

/// Grid size: the matrix has `M * M` rows.
const M: u32 = 64;

/// Diffusion coefficient of the cell of `row`: 1 in the bottom half of the
/// grid, 100 in the top half.
fn k(row: u32) -> f64 {
    if row / M < M / 2 {
        1.0
    } else {
        100.0
    }
}

/// The neighbors of `row` (west, east, south, north), `None` on the
/// boundary.
fn neighbors(row: u32) -> [Option<u32>; 4] {
    let (i, j) = (row / M, row % M);
    [
        (j > 0).then(|| row - 1),
        (j < M - 1).then(|| row + 1),
        (i > 0).then(|| row - M),
        (i < M - 1).then(|| row + M),
    ]
}

/// Lower triangle of the matrix; the upper triangle is added by symmetry.
/// The coupling between two cells is the mean of their coefficients, and a
/// boundary face adds the cell's own coefficient to the diagonal.
fn diffusion_lower(rows: Range<u32>) -> Vec<(u32, u32, f64)> {
    let mut entries = Vec::new();
    for row in rows {
        let mut diagonal = 0.0;
        for neighbor in neighbors(row) {
            match neighbor {
                Some(n) => {
                    let coupling = (k(row) + k(n)) / 2.0;
                    diagonal += coupling;
                    if n < row {
                        entries.push((row, n, -coupling));
                    }
                }
                None => diagonal += k(row),
            }
        }
        entries.push((row, row, diagonal));
    }
    entries
}

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

    let rows = distribute_rows(&comm, M * M);
    let mut a = ctx.symmetric_matrix_builder(&comm, rows.clone())?;
    a.add(diffusion_lower(rows.clone()))?;
    let a = a.assemble()?;

    // b = A * [1, ..., 1], so the exact solution is x = [1, ..., 1]. The
    // row sums only keep the boundary faces' contributions.
    let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |row| {
        neighbors(row).iter().filter(|n| n.is_none()).count() as f64 * k(row)
    })?;

    let config = PCGSolverConfigBuilder::default()
        .tol(1e-10)
        .max_iters(5000)
        .two_norm(true)
        .build()?;
    let amg = SymmetricAMGPreconditionerConfigBuilder::default()
        .coarsen_type(CoarsenType::Hmis)
        .relax_type(SymmetricSmoother::L1SymmetricGaussSeidel)
        .build()?;
    let preconditioners: [(&str, Option<SymmetricPreconditioner<'_>>); 3] = [
        ("none", None),
        ("Jacobi", Some(JacobiPreconditioner.into())),
        (
            "AMG",
            Some(ctx.symmetric_amg_preconditioner(&comm, amg)?.into()),
        ),
    ];

    let mut iterations = Vec::new();
    for (name, preconditioner) in preconditioners {
        let mut solver = ctx.pcg_solver(&comm, config)?;
        if let Some(preconditioner) = preconditioner {
            solver = solver.with_preconditioner(preconditioner)?;
        }
        let mut x = ParVector::zeros(&ctx, &comm, rows.clone())?;
        let status = solver.solve(&a, &b, &mut x)?;
        let error = max_error(&comm, &x)?;
        if root {
            println!(
                "{name:>6}: {:4} iterations, max error {error:.1e}",
                status.num_iters
            );
        }
        if error > 1e-6 {
            return Err(format!("{name}: max error {error} is too large").into());
        }
        iterations.push(status.num_iters);
    }
    if iterations[2] >= iterations[0] {
        return Err("AMG should need fewer iterations than plain CG".into());
    }

    // hypre objects must be dropped before hypre is finalized (checked by
    // the compiler).
    drop((a, b));
    ctx.finalize()?;
    Ok(())
}
