//! Non-symmetric systems: convection-diffusion with GMRES and BiCGSTAB.
//!
//! Discretizes `-Δu + c · ∇u` on an `M x M` grid, with upwind differences
//! for the convection term, which gives a non-symmetric matrix. CG does not
//! apply; GMRES and BiCGSTAB do, preconditioned by algebraic multigrid with
//! a (non-symmetric) forward Gauss-Seidel smoother, or by Jacobi.
//!
//! ```bash
//! cargo run --example gmres-convection-diffusion
//! mpirun -n 4 target/debug/examples/gmres-convection-diffusion
//! ```

use hypre_rs::prelude::*;
use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::topology::{Communicator, SimpleCommunicator};
use std::ops::Range;

/// Grid size: the matrix has `M * M` rows.
const M: u32 = 64;

/// Convection strength, in grid units, towards the north-east.
const C: f64 = 2.0;

/// The `(column, value)` entries of `row`: the 5-point Laplacian, plus the
/// upwind convection term, which only couples to the west and south
/// neighbors.
fn row_entries(row: u32) -> Vec<(u32, f64)> {
    let (i, j) = (row / M, row % M);
    let mut entries = vec![(row, 4.0 + 2.0 * C)];
    if j > 0 {
        entries.push((row - 1, -1.0 - C));
    }
    if j < M - 1 {
        entries.push((row + 1, -1.0));
    }
    if i > 0 {
        entries.push((row - M, -1.0 - C));
    }
    if i < M - 1 {
        entries.push((row + M, -1.0));
    }
    entries
}

/// The matrix rows of `rows`, assembled.
fn convection_diffusion<'a>(
    ctx: &'a HypreContext<'_>,
    comm: &'a SimpleCommunicator,
    rows: Range<u32>,
) -> Result<CSRMatrix<'a>, HypreError> {
    let mut a = ctx.local_matrix_builder(comm, rows.clone(), rows.clone())?;
    for row in rows {
        a.add(
            row_entries(row)
                .into_iter()
                .map(|(col, value)| (row, col, value)),
        )?;
    }
    a.assemble()
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
    let a = convection_diffusion(&ctx, &comm, rows.clone())?;
    if a.is_symmetric(0.0)? {
        return Err("the convection-diffusion matrix should not be symmetric".into());
    }
    // b = A * [1, ..., 1], so the exact solution is x = [1, ..., 1].
    let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |row| {
        row_entries(row).iter().map(|(_, value)| value).sum()
    })?;

    // AMG with a forward Gauss-Seidel smoother and C/F ordering: a good
    // preconditioner for this matrix, but not a symmetric one.
    let amg = AMGPreconditionerConfigBuilder::default()
        .coarsen_type(CoarsenType::Hmis)
        .interp_type(InterpType::ExtendedPlusI)
        .relax_type(Smoother::HybridGaussSeidelForward)
        .relax_order(true)
        .build()?;
    let gmres = GMRESSolverConfigBuilder::default()
        .tol(1e-10)
        .k_dim(30)
        .max_iters(1000)
        .build()?;
    let bicgstab = BiCGSTABSolverConfigBuilder::default()
        .tol(1e-10)
        .max_iters(1000)
        .build()?;

    let solvers: [(&str, Solver<'_>); 4] = [
        ("GMRES", ctx.gmres_solver(&comm, gmres)?.into()),
        (
            "GMRES + AMG",
            ctx.gmres_solver(&comm, gmres)?
                .with_preconditioner(ctx.amg_preconditioner(&comm, amg)?)?
                .into(),
        ),
        (
            "BiCGSTAB + Jacobi",
            ctx.bicgstab_solver(&comm, bicgstab)?
                .with_preconditioner(JacobiPreconditioner)?
                .into(),
        ),
        (
            "BiCGSTAB + AMG",
            ctx.bicgstab_solver(&comm, bicgstab)?
                .with_preconditioner(ctx.amg_preconditioner(&comm, amg)?)?
                .into(),
        ),
    ];
    for (name, solver) in solvers {
        let mut x = ParVector::zeros(&ctx, &comm, rows.clone())?;
        let status = solver.solve(&a, &b, &mut x)?;
        let error = max_error(&comm, &x)?;
        if root {
            println!(
                "{name:>17}: {:4} iterations, max error {error:.1e}",
                status.num_iters
            );
        }
        if error > 1e-6 {
            return Err(format!("{name}: max error {error} is too large").into());
        }
    }

    // hypre objects must be dropped before hypre is finalized (checked by
    // the compiler).
    drop((a, b));
    ctx.finalize()?;
    Ok(())
}
