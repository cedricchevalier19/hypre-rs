//! The Conjugate Gradient solver, for symmetric positive definite matrices.

use derive_builder::Builder;
use std::marker::PhantomData;
use std::ptr::null_mut;

use crate::error::HypreError;
use crate::matrix::SymmetricMatrix;
use crate::solvers::{IterativeSolverStatus, SymmetricLinearSolver};
use crate::{HypreContext, HypreResult, Vector};
use hypre_sys::*;

krylov_solver! {
    /// Preconditioned Conjugate Gradient solver.
    ///
    /// CG requires a symmetric positive definite matrix, so it solves systems
    /// with a [SymmetricMatrix] only (see [SymmetricLinearSolver]).
    ///
    /// # Example
    ///
    /// ```
    /// use hypre_rs::solvers::PCGSolver;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let mpi_comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// // Create PCGSolver with hypre's default configuration.
    /// let solver = PCGSolver::new(&ctx, &mpi_comm, Default::default())?;
    /// # drop(solver);
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    solver PCGSolver;
    /// Conjugate Gradient Solver Configuration
    ///
    /// ## Convergence criteria
    ///
    /// According to hypre source code, default is
    /// $$<C*r, r>  < \text{max}({\text{tol}_a^2, tol^2 * <C*b, b>)$$
    ///
    /// If res_tol is set, $$||r_\text{new} - r_\text{old}||_C < \text{tol}_r * ||b||_C$$ is used.
    ///
    /// If two_norm is set, $$||.||_2$$ norm is used.
    ///
    /// hypre has no getter for `abs_tol`, `recompute_residual` and
    /// `recompute_residual_period`, which are therefore `None` in
    /// [PCGSolver::current_config].
    config PCGSolverConfig, PCGSolverConfigBuilder;
    create HYPRE_ParCSRPCGCreate, destroy HYPRE_ParCSRPCGDestroy;
    setup HYPRE_ParCSRPCGSetup, solve HYPRE_ParCSRPCGSolve;
    solves SymmetricLinearSolver for SymmetricMatrix;
    status HYPRE_ParCSRPCGGetNumIterations, HYPRE_ParCSRPCGGetFinalRelativeResidualNorm,
        converged HYPRE_PCGGetConverged;
    positive [tol, abs_tol, res_tol, conv_tol_fact];
    params {
        /// Relative convergence tolerance
        tol: f64 => HYPRE_PCGSetTol,
        /// Absolute convergence tolerance
        abs_tol: f64 => HYPRE_PCGSetAbsoluteTol,
        /// Residual convergence tolerance
        res_tol: f64 => HYPRE_PCGSetResidualTol,
        /// Convergence factor tolerance, if >0 used for special test for slow convergence
        conv_tol_fact: f64 => HYPRE_PCGSetConvergenceFactorTol,
        /// Maximum number of iterations
        max_iters: usize => HYPRE_PCGSetMaxIter,
        /// Force using $||.||_2$ instead of $||.||_C$
        two_norm: bool => HYPRE_PCGSetTwoNorm,
        /// Convergence only if $x$ didn't change too much in the last iteration
        rel_change: bool => HYPRE_PCGSetRelChange,
        /// Force computing an explicit residual if implicit residual convergence
        recompute_residual: bool => HYPRE_PCGSetRecomputeResidual,
        /// Periodically recompute an explicit residual
        recompute_residual_period: usize => HYPRE_PCGSetRecomputeResidualP,
        /// Logging level (1 stores the residual norms)
        logging: u32 => HYPRE_PCGSetLogging,
        /// Print level (0: nothing, 2: residual norm of every iteration)
        print_level: u32 => HYPRE_PCGSetPrintLevel,
    }
    getters {
        tol: HYPRE_Real => HYPRE_PCGGetTol,
        res_tol: HYPRE_Real => HYPRE_PCGGetResidualTol,
        conv_tol_fact: HYPRE_Real => HYPRE_PCGGetConvergenceFactorTol,
        max_iters: HYPRE_Int => HYPRE_PCGGetMaxIter,
        two_norm: HYPRE_Int => HYPRE_PCGGetTwoNorm,
        rel_change: HYPRE_Int => HYPRE_PCGGetRelChange,
        logging: HYPRE_Int => HYPRE_PCGGetLogging,
        print_level: HYPRE_Int => HYPRE_PCGGetPrintLevel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_test() -> Result<(), Box<dyn std::error::Error>> {
        // hypre calls return its global error flag, which other unit tests set.
        let _lock = crate::utils::hypre_error_lock();
        // Keep the universe alive: dropping it calls MPI_Finalize.
        let universe = mpi::initialize()
            .ok_or("MPI was already initialized (or finalized) in this process")?;
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
}
