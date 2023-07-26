use derive_builder::Builder;

use std::ptr::null_mut;

use crate::error::HypreError;
use crate::solvers::{
    boomer_amg::BoomerAMG, IterativeSolverStatus, SymmetricLinearPreconditioner,
    SymmetricLinearSolver,
};

use crate::matrix::Matrix;
use crate::{HypreResult, Vector};
use hypre_sys::*;
use mpi;

/// Access to hypre Conjugate Gradient solver
#[derive(Debug)]
pub struct PCGSolver {
    internal_solver: HYPRE_Solver,
}

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
#[derive(Default, Debug, Clone, Builder)]
#[builder(setter(into, strip_option), default)]
#[builder(build_fn(validate = "Self::validate", error = "HypreError"))]
pub struct PCGSolverConfig {
    /// Relative convergence tolerance
    pub tol: Option<f64>,
    /// Absolute convergence tolerance
    pub abs_tol: Option<f64>,
    /// Residual convergence tolerance
    pub res_tol: Option<f64>,
    /// Convergence factor tolerance, if >0 used for special test for slow convergence
    pub conv_tol_fact: Option<f64>,
    /// Maximum number of iterations
    pub max_iters: Option<usize>,
    /// Force using $||.||_2$ instead of $||.||_C$
    pub two_norm: Option<bool>,
    /// Convergence only if $x$ didn't change too much in the last iteration
    pub rel_change: Option<bool>,
    pub logging: Option<u32>,
    pub print_level: Option<u32>,
    /// Force computing an explicit residual if implicit residual convergence
    pub recompute_residual: Option<bool>,
    /// Periodically recompute an explicit residual
    pub recompute_residual_period: Option<usize>,
    /// Preconditioner
    pub precond: Option<BoomerAMG>,
}

impl PCGSolverConfigBuilder {
    /// Validates valid parameters for [PCGSolverConfig]
    fn validate(&self) -> HypreResult<()> {
        check_positive_parameter![self, tol];
        check_positive_parameter![self, abs_tol];
        check_positive_parameter![self, res_tol];
        check_positive_parameter![self, conv_tol_fact];
        Ok(())
    }
}

impl PCGSolverConfig {
    fn apply(&self, pcg: &mut PCGSolver) -> HypreResult<()> {
        set_parameter![HYPRE_PCGSetTol, pcg.internal_solver, self.tol];
        set_parameter![HYPRE_PCGSetAbsoluteTol, pcg.internal_solver, self.abs_tol];
        set_parameter![HYPRE_PCGSetResidualTol, pcg.internal_solver, self.res_tol];
        set_parameter![HYPRE_PCGSetMaxIter, pcg.internal_solver, self.max_iters];
        set_parameter![HYPRE_PCGSetTwoNorm, pcg.internal_solver, self.two_norm];
        set_parameter![HYPRE_PCGSetRelChange, pcg.internal_solver, self.rel_change];
        set_parameter![
            HYPRE_PCGSetRecomputeResidual,
            pcg.internal_solver,
            self.recompute_residual
        ];
        set_parameter![
            HYPRE_PCGSetRecomputeResidualP,
            pcg.internal_solver,
            self.recompute_residual_period
        ];
        Ok(())
    }

    fn observe(&mut self, pcg: &PCGSolver) -> HypreResult<()> {
        self.tol = get_parameter![HYPRE_PCGGetTol, pcg.internal_solver, HYPRE_Real]?;
        self.res_tol = get_parameter![HYPRE_PCGGetResidualTol, pcg.internal_solver, HYPRE_Real]?;
        let max_iters: HYPRE_Int =
            get_parameter![HYPRE_PCGGetMaxIter, pcg.internal_solver, HYPRE_Int]?;
        if max_iters.is_negative() {
            return Err(HypreError::HypreGenericError);
        }
        self.max_iters = Some(max_iters as usize);
        let boolean: HYPRE_Int =
            get_parameter![HYPRE_PCGGetTwoNorm, pcg.internal_solver, HYPRE_Int]?;
        self.two_norm = Some(boolean != 0);

        let boolean: HYPRE_Int =
            get_parameter![HYPRE_PCGGetRelChange, pcg.internal_solver, HYPRE_Int]?;
        self.rel_change = Some(boolean != 0);
        Ok(())
    }
}

/// Preconditioned Conjugate Gradient solver
///
/// # Example
/// ```
/// use hypre_rs::solvers::PCGSolver;
///
/// # let mpi_comm = mpi::initialize().unwrap().world();
/// // Create PCGSolver with hypre's default configuration.
/// let solver = PCGSolver::new(&mpi_comm, Default::default()).unwrap();
///
/// ```
impl PCGSolver {
    /// Creates a new PCGSolver according to the given configuration
    pub fn new(
        comm: &impl mpi::topology::Communicator,
        config: PCGSolverConfig,
    ) -> HypreResult<Self> {
        let mut solver = PCGSolver {
            internal_solver: null_mut(),
        };
        unsafe {
            let h_solver = &mut solver.internal_solver as *mut _ as *mut HYPRE_Solver;
            match HYPRE_ParCSRPCGCreate(comm.as_raw(), h_solver) {
                0 => {}
                x => return Err(HypreError::new(x)),
            }
        }
        solver.config(config)
    }

    /// Changes the configuration of the solver
    pub fn config(mut self, config: PCGSolverConfig) -> HypreResult<Self> {
        config.apply(&mut self)?;
        Ok(self)
    }

    /// Returns the configuration of the solver
    pub fn current_config(&self) -> HypreResult<PCGSolverConfig> {
        let mut config: PCGSolverConfig = Default::default();
        config.observe(&self)?;
        Ok(config)
    }
}

impl SymmetricLinearSolver for PCGSolver {
    fn set_precond(&mut self, precond: impl SymmetricLinearPreconditioner) {
        let internal_precond = precond.precond_descriptor();
        unsafe {
            HYPRE_PCGSetPrecond(
                self.internal_solver,
                internal_precond.get_precond(),
                internal_precond.get_precond_setup(),
                internal_precond.get_internal(),
            );
        }
    }

    /// Solves a linear system using Preconditioned Conjugate Gradient algorithm.
    fn solve(
        &self,
        mat: &mut Matrix,
        rhs: &Vector,
        x: &mut Vector,
    ) -> HypreResult<IterativeSolverStatus> {
        unsafe {
            check_hypre!(HYPRE_PCGSetup(
                self.internal_solver,
                mat.get_internal()?,
                rhs.get_internal()?,
                x.get_internal()?,
            ));

            match HYPRE_PCGSolve(
                self.internal_solver,
                mat.get_internal()?,
                rhs.get_internal()?,
                x.get_internal()?,
            ) {
                0 => {
                    let mut res_norm: HYPRE_Real = 0.0.into();
                    // We do not care about return value of HYPRE_PCGGet functions has they cannot generate new errors
                    HYPRE_PCGGetFinalRelativeResidualNorm(
                        self.internal_solver,
                        &mut res_norm as *mut _,
                    );
                    let mut converged: HYPRE_Int = 0.into();
                    HYPRE_PCGGetConverged(self.internal_solver, &mut converged as *mut _);
                    let mut num_iters: HYPRE_Int = 0.into();
                    HYPRE_PCGGetNumIterations(self.internal_solver, &mut num_iters as *mut _);
                    Ok(IterativeSolverStatus {
                        num_iters: num_iters.try_into()?,
                        res_num: res_norm.into(),
                        converged: (converged != 0),
                    })
                }
                x => Err(HypreError::new(x)),
            }
        }
    }
}

impl Drop for PCGSolver {
    /// Calls hypre's destroy function.
    fn drop(&mut self) {
        unsafe {
            HYPRE_ParCSRPCGDestroy(self.internal_solver);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_test() {
        let mpi_comm = mpi::initialize().unwrap().world();
        let solver = PCGSolver::new(&mpi_comm, Default::default()).unwrap();

        let parameters = solver.current_config();
        println!("{:?}", parameters);

        let my_parameters = PCGSolverConfigBuilder::default()
            .tol(1e-9)
            .max_iters(500usize)
            .two_norm(true)
            .recompute_residual_period(8usize)
            .build()
            .unwrap();
        let solver = PCGSolver::new(&mpi_comm, my_parameters.clone()).unwrap();
        let parameters = solver.current_config().unwrap();
        println!("{:?}", parameters);
        assert_eq!(my_parameters.tol, parameters.tol);
        assert_eq!(my_parameters.max_iters, parameters.max_iters);
        assert_eq!(my_parameters.two_norm, parameters.two_norm);
    }
}
