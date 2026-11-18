//! hypre can be initialized again once the previous context is gone, whether
//! it was finalized explicitly or dropped.

use hypre_rs::HypreError;

#[test]
fn reinitialize() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();

    let ctx = hypre_rs::initialize(&universe)?;
    // Only one hypre context may be alive.
    assert!(matches!(
        hypre_rs::initialize(&universe),
        Err(HypreError::AlreadyInitialized)
    ));
    drop(ctx.ij_vector(&comm, 0..0)?);
    ctx.finalize()?;

    // Dropping the context finalizes hypre too.
    let ctx = hypre_rs::initialize(&universe)?;
    drop(ctx.ij_vector(&comm, 0..0)?);
    drop(ctx);

    let ctx = hypre_rs::initialize(&universe)?;
    drop(ctx.ij_vector(&comm, 0..0)?);
    ctx.finalize()?;
    Ok(())
}
