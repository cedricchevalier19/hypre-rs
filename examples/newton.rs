//! Newton's method on a nonlinear problem, refilling the Jacobian and the
//! residual at every iteration.
//!
//! Solves `-u'' + u³ = f` on `(0, 1)` with `N` interior grid points
//! (finite differences, `A = tridiag(-1, 2, -1) / h²` with `h = 1 / (N + 1)`),
//! with `f = A·1 + 1` so that the solution is `u = [1, ..., 1]`. Each iteration solves `J(u) δ = F(u)` and sets
//! `u ← u - δ`, where
//!
//! - `F(u) = A u + u³ - f`, the residual,
//! - `J(u) = A + diag(3 u²)`, its Jacobian, whose sparsity pattern never
//!   changes.
//!
//! The Jacobian, the residual and the correction are assembled once, then
//! refilled with new values at every iteration, reusing hypre's objects.
//!
//! ```bash
//! cargo run --example newton
//! mpirun -n 4 target/debug/examples/newton
//! ```

use hypre_rs::prelude::*;
use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::topology::Communicator;

/// Number of grid points.
const N: u32 = 200;

/// The columns of row `i` of `A`, with their values.
fn a_row(i: u32) -> Vec<(u32, f64)> {
    let h = 1.0 / f64::from(N + 1);
    let scale = 1.0 / (h * h);
    let mut row = vec![(i, 2.0 * scale)];
    if i > 0 {
        row.push((i - 1, -scale));
    }
    if i + 1 < N {
        row.push((i + 1, -scale));
    }
    row
}

/// `f = A·1 + 1`: the row sums of `A` (`1 / h²` next to the boundary, 0
/// inside) plus 1.
fn f(i: u32) -> f64 {
    a_row(i).iter().map(|(_, v)| v).sum::<f64>() + 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let root = comm.rank() == 0;
    let ctx = hypre_rs::initialize(&universe)?;

    let rows = distribute_rows(&comm, N);
    // This rank's values of u, starting from 0.
    let mut u = vec![0.0; rows.len()];

    // Assemble the Jacobian once, to fix its pattern; the values do not
    // matter yet. Each rank gives its own rows.
    let mut jacobian = ctx.local_matrix_builder(&comm, rows.clone(), rows.clone())?;
    for i in rows.clone() {
        jacobian.add(a_row(i).into_iter().map(|(j, v)| (i, j, v)))?;
    }
    let mut jacobian = jacobian.assemble()?;
    let mut residual = ParVector::zeros(&ctx, &comm, rows.clone())?;
    let mut delta = ParVector::zeros(&ctx, &comm, rows.clone())?;

    // The absolute tolerance stops GMRES once the residual is near round-off,
    // instead of asking for 1e-12 of an already tiny residual.
    let config = GMRESSolverConfigBuilder::default()
        .tol(1e-12)
        .abs_tol(1e-9)
        .build()?;
    let preconditioner = ctx.amg_preconditioner(&comm, Default::default())?;
    let solver = ctx
        .gmres_solver(&comm, config)?
        .with_preconditioner(preconditioner)?;

    for iteration in 1..=20 {
        // J(u) = A + diag(3 u²): same pattern, new diagonal.
        let mut j = jacobian.refill()?;
        for (k, i) in rows.clone().enumerate() {
            j.add(a_row(i).into_iter().map(|(col, v)| (i, col, v)))?;
            j.add([(i, i, 3.0 * u[k] * u[k])])?;
        }
        jacobian = j.assemble()?;

        // F(u) = A u + u³ - f. The owner of u_k adds its contribution to
        // every row of column k of A (A is symmetric, so those are the
        // entries of row k), including rows owned by the neighbor ranks:
        // assembling sends those to their owners.
        let mut r = residual.refill()?;
        for (k, i) in rows.clone().enumerate() {
            r.add(a_row(i).into_iter().map(|(row, v)| (row, v * u[k])))?;
            r.add([(i, u[k] * u[k] * u[k] - f(i))])?;
        }
        residual = r.assemble()?;

        // Solve J δ = F from δ = 0, then u ← u - δ.
        delta = delta.refill()?.assemble()?;
        let status = solver.solve(&jacobian, &residual, &mut delta)?;
        let mut local_max = 0.0_f64;
        for (u_k, d) in u.iter_mut().zip(delta.local_values()?) {
            *u_k -= d;
            local_max = local_max.max(d.abs());
        }
        let mut correction = 0.0;
        comm.all_reduce_into(&local_max, &mut correction, SystemOperation::max());
        if root {
            println!(
                "iteration {iteration}: max |δ| = {correction:.2e} ({} GMRES iterations)",
                status.num_iters
            );
        }
        if correction < 1e-10 {
            break;
        }
    }

    let error = u.iter().fold(0.0_f64, |m, v| m.max((v - 1.0).abs()));
    let mut max_error = 0.0;
    comm.all_reduce_into(&error, &mut max_error, SystemOperation::max());
    if root {
        println!("max error {max_error:.1e}");
    }
    if max_error > 1e-8 {
        return Err(format!("Newton did not converge: max error {max_error}").into());
    }

    // hypre objects must be dropped before hypre is finalized (checked by
    // the compiler).
    drop((solver, jacobian, residual, delta));
    ctx.finalize()?;
    Ok(())
}
