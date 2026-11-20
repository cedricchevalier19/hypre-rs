//! Choosing the solver at runtime, from the command line.
//!
//! The `Solver` enum holds any solver of general matrices, so the choice can
//! come from a configuration file or, here, from the first argument: one of
//! `gmres` (the default), `flexgmres`, `lgmres`, `cogmres`, `bicgstab` or
//! `amg`.
//!
//! ```bash
//! cargo run --example choose-solver -- bicgstab
//! mpirun -n 4 target/debug/examples/choose-solver amg
//! ```

use hypre_rs::prelude::*;
use mpi::topology::Communicator;

/// Global number of rows.
const N: u32 = 1000;

/// The solver named `name`, preconditioned by algebraic multigrid when it
/// is a Krylov solver.
fn make_solver<'a>(
    ctx: &'a HypreContext<'_>,
    comm: &'a mpi::topology::SimpleCommunicator,
    name: &str,
) -> Result<Solver<'a>, Box<dyn std::error::Error>> {
    let tol = 1e-10;
    let amg = || ctx.symmetric_amg_preconditioner(comm, Default::default());
    Ok(match name {
        "gmres" => {
            let config = GMRESSolverConfigBuilder::default().tol(tol).build()?;
            ctx.gmres_solver(comm, config)?
                .with_preconditioner(amg()?)?
                .into()
        }
        "flexgmres" => {
            let config = FlexGMRESSolverConfigBuilder::default().tol(tol).build()?;
            ctx.flex_gmres_solver(comm, config)?
                .with_preconditioner(amg()?)?
                .into()
        }
        "lgmres" => {
            let config = LGMRESSolverConfigBuilder::default().tol(tol).build()?;
            ctx.lgmres_solver(comm, config)?
                .with_preconditioner(amg()?)?
                .into()
        }
        "cogmres" => {
            let config = COGMRESSolverConfigBuilder::default()
                .tol(tol)
                .cgs(GramSchmidtPasses::Two)
                .build()?;
            ctx.cogmres_solver(comm, config)?
                .with_preconditioner(amg()?)?
                .into()
        }
        "bicgstab" => {
            let config = BiCGSTABSolverConfigBuilder::default().tol(tol).build()?;
            ctx.bicgstab_solver(comm, config)?
                .with_preconditioner(amg()?)?
                .into()
        }
        "amg" => {
            let config = AMGSolverConfigBuilder::default().tol(tol).build()?;
            ctx.amg_solver(comm, config)?.into()
        }
        other => return Err(format!(
            "unknown solver {other:?}: expected gmres, flexgmres, lgmres, cogmres, bicgstab or amg"
        )
        .into()),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let name = std::env::args().nth(1).unwrap_or_else(|| "gmres".into());

    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;

    // The 1D Laplacian tridiag(-1, 2, -1), and b = A * [1, ..., 1].
    let rows = distribute_rows(&comm, N);
    let mut a = ctx.symmetric_matrix(&comm, rows.clone())?;
    a.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
    a.add_elements(rows.clone().filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
    let a = a.assemble()?;
    let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |i| {
        if i == 0 || i == N - 1 {
            1.0
        } else {
            0.0
        }
    })?;
    let mut x = ParVector::zeros(&ctx, &comm, rows.clone())?;

    let solver = make_solver(&ctx, &comm, &name)?;
    let status = solver.solve(&a, &b, &mut x)?;
    if comm.rank() == 0 {
        println!("{name}: {status}");
    }
    let wrong = rows
        .zip(x.local_values()?)
        .find(|(_, v)| (v - 1.0).abs() > 1e-6);
    if let Some((i, v)) = wrong {
        return Err(format!("{name}: x[{i}] = {v}, expected 1").into());
    }

    // hypre objects must be dropped before hypre is finalized (checked by
    // the compiler).
    drop((solver, a, b, x));
    ctx.finalize()?;
    Ok(())
}
