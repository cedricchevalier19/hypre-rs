//! Unit tests of the Krylov solvers.

use super::*;
use crate::error::HypreError;
use hypre_sys::HYPRE_Int;

#[test]
fn config_test() -> Result<(), Box<dyn std::error::Error>> {
    // hypre calls return its global error flag, which other unit tests set.
    let _lock = crate::utils::hypre_error_lock();
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let mpi_comm = universe.world();
    let ctx = crate::initialize(&universe)?;
    // Only one hypre context may be alive.
    assert!(matches!(
        crate::initialize(&universe),
        Err(HypreError::AlreadyInitialized)
    ));

    // Solvers must be destroyed before hypre is finalized.
    {
        let solver = PCGSolver::new(&ctx, &mpi_comm, Default::default())?;

        let parameters = solver.current_config();
        println!("{:?}", parameters);

        let my_parameters = PCGSolverConfigBuilder::default()
            .tol(1e-9)
            .max_iters(500usize)
            .two_norm(true)
            .recompute_residual_period(8usize)
            .build()?;
        let solver = PCGSolver::new(&ctx, &mpi_comm, my_parameters)?;
        let parameters = solver.current_config()?;
        println!("{:?}", parameters);
        assert_eq!(my_parameters.tol, parameters.tol);
        assert_eq!(my_parameters.max_iters, parameters.max_iters);
        assert_eq!(my_parameters.two_norm, parameters.two_norm);

        // `config` changes an existing solver; unset fields keep their value.
        let other_parameters = PCGSolverConfigBuilder::default()
            .tol(1e-3)
            .max_iters(7usize)
            .build()?;
        let solver = solver.config(other_parameters)?;
        let parameters = solver.current_config()?;
        assert_eq!(parameters.tol, Some(1e-3));
        assert_eq!(parameters.max_iters, Some(7));
        assert_eq!(parameters.two_norm, my_parameters.two_norm);

        // Every field that hypre can read back round-trips.
        let all_parameters = PCGSolverConfigBuilder::default()
            .tol(1e-4)
            .res_tol(1e-5)
            .conv_tol_fact(0.5)
            .max_iters(42usize)
            .two_norm(false)
            .rel_change(true)
            .logging(1u32)
            .print_level(2u32)
            .build()?;
        let solver = solver.config(all_parameters)?;
        let parameters = solver.current_config()?;
        assert_eq!(parameters.tol, Some(1e-4));
        assert_eq!(parameters.res_tol, Some(1e-5));
        assert_eq!(parameters.conv_tol_fact, Some(0.5));
        assert_eq!(parameters.max_iters, Some(42));
        assert_eq!(parameters.two_norm, Some(false));
        assert_eq!(parameters.rel_change, Some(true));
        assert_eq!(parameters.logging, Some(1));
        assert_eq!(parameters.print_level, Some(2));
        // hypre has no getter for these.
        assert_eq!(parameters.abs_tol, None);
        assert_eq!(parameters.recompute_residual, None);
        assert_eq!(parameters.recompute_residual_period, None);
        drop(solver);

        // A value that does not fit hypre's int is an error, not a panic.
        if usize::try_from(HYPRE_Int::MAX).is_ok_and(|max| max < usize::MAX) {
            let too_many = PCGSolverConfigBuilder::default()
                .max_iters(usize::MAX)
                .build()?;
            assert!(matches!(
                PCGSolver::new(&ctx, &mpi_comm, too_many),
                Err(HypreError::ConversionError(_))
            ));
        }
    }
    ctx.finalize()?;
    Ok(())
}

#[test]
fn config_rejects_negative_tolerances() {
    for builder in [
        *PCGSolverConfigBuilder::default().tol(-1.0),
        *PCGSolverConfigBuilder::default().abs_tol(-1.0),
        *PCGSolverConfigBuilder::default().res_tol(-1.0),
        *PCGSolverConfigBuilder::default().conv_tol_fact(-1.0),
    ] {
        assert!(matches!(
            builder.build(),
            Err(HypreError::InvalidParameterPositive)
        ));
    }
    assert!(PCGSolverConfigBuilder::default().tol(0.0).build().is_ok());
}
