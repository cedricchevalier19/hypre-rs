//! Turning a matrix built with both triangles into a `SymmetricMatrix`.
//!
//! CG only accepts a `SymmetricMatrix`. When the matrix comes from code that
//! gives every entry (here, a finite-element-like assembly where each
//! element adds its 2x2 contribution), it can be checked at runtime and
//! converted. The check is collective and costs about as much as a
//! transposition. A matrix that is not symmetric is rejected.
//!
//! ```bash
//! cargo run --example check-symmetry
//! mpirun -n 4 target/debug/examples/check-symmetry
//! ```

use hypre_rs::prelude::*;
use mpi::topology::Communicator;

/// Number of nodes of the 1D mesh.
const N: u32 = 500;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let root = comm.rank() == 0;
    let ctx = hypre_rs::initialize(&universe)?;
    let rows = distribute_rows(&comm, N);

    // Each element (i - 1, i) adds the stiffness matrix [[1, -1], [-1, 1]],
    // with both triangles, to the rows it touches; some of them belong to
    // the previous MPI rank. A unit diagonal term makes the matrix definite.
    let mut ij = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
    for i in rows.clone() {
        ij.add_elements([(i, i, 1.0)])?;
        if i > 0 {
            ij.add_elements([
                (i - 1, i - 1, 1.0),
                (i - 1, i, -1.0),
                (i, i - 1, -1.0),
                (i, i, 1.0),
            ])?;
        }
    }
    let a = ij.assemble()?;

    // Exact check; `try_from_with_tolerance` accepts small differences.
    let a = SymmetricMatrix::try_from(a)?;
    if root {
        println!("the stiffness matrix is symmetric");
    }

    // An advection term breaks the symmetry: the conversion fails.
    let mut ij = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
    for i in rows.clone() {
        ij.add_elements([(i, i, 2.0)])?;
        if i > 0 {
            ij.add_elements([(i, i - 1, -1.5)])?;
        }
        if i + 1 < N {
            ij.add_elements([(i, i + 1, -0.5)])?;
        }
    }
    match SymmetricMatrix::try_from(ij.assemble()?) {
        Err(HypreError::NotSymmetric) if root => println!("the advection matrix is not"),
        Err(HypreError::NotSymmetric) => {}
        Ok(_) => return Err("the advection matrix should not be symmetric".into()),
        Err(error) => return Err(error.into()),
    }

    // CG on the symmetric one. A * [1, ..., 1] is the unit diagonal term.
    let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |_| 1.0)?;
    let mut x = ParVector::zeros(&ctx, &comm, rows.clone())?;
    let config = PCGSolverConfigBuilder::default().tol(1e-12).build()?;
    let solver = ctx
        .pcg_solver(&comm, config)?
        .with_preconditioner(JacobiPreconditioner)?;
    let status = solver.solve(&a, &b, &mut x)?;
    if root {
        println!("CG: {status}");
    }
    let wrong = rows
        .zip(x.local_values()?)
        .find(|(_, v)| (v - 1.0).abs() > 1e-8);
    if let Some((i, v)) = wrong {
        return Err(format!("x[{i}] = {v}, expected 1").into());
    }

    // hypre objects must be dropped before hypre is finalized (checked by
    // the compiler).
    drop((solver, a, b, x));
    ctx.finalize()?;
    Ok(())
}
