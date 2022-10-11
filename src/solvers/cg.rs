use derive_builder::Builder;

use std::ptr::null_mut;

use crate::error::HypreError;
use crate::solvers::{IterativeSolverStatus, LinearSolver};

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
#[derive(Default, Debug, Clone, Builder, Copy)]
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
}

macro_rules! check_positive_parameter {
    ( $obj:expr, $param:ident) => {{
        if let Some(Some($param)) = $obj.$param {
            if $param < 0.into() {
                return Err(HypreError::InvalidParameterPositive);
            }
        }
    }};
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
    pub fn config(self, config: PCGSolverConfig) -> HypreResult<Self> {
        set_parameter![HYPRE_PCGSetTol, self.internal_solver, config.tol];
        set_parameter![
            HYPRE_PCGSetAbsoluteTol,
            self.internal_solver,
            config.abs_tol
        ];
        set_parameter![
            HYPRE_PCGSetResidualTol,
            self.internal_solver,
            config.res_tol
        ];
        set_parameter![HYPRE_PCGSetMaxIter, self.internal_solver, config.max_iters];
        set_parameter![HYPRE_PCGSetTwoNorm, self.internal_solver, config.two_norm];
        set_parameter![
            HYPRE_PCGSetRelChange,
            self.internal_solver,
            config.rel_change
        ];
        set_parameter![
            HYPRE_PCGSetRecomputeResidual,
            self.internal_solver,
            config.recompute_residual
        ];
        set_parameter![
            HYPRE_PCGSetRecomputeResidualP,
            self.internal_solver,
            config.recompute_residual_period
        ];

        Ok(self)
    }

    /// Returns the configuration of the solver
    pub fn current_config(&self) -> HypreResult<PCGSolverConfig> {
        let mut config: PCGSolverConfig = Default::default();

        config.tol = get_parameter![HYPRE_PCGGetTol, self.internal_solver, HYPRE_Real]?;
        config.res_tol = get_parameter![HYPRE_PCGGetResidualTol, self.internal_solver, HYPRE_Real]?;
        let max_iters: HYPRE_Int =
            get_parameter![HYPRE_PCGGetMaxIter, self.internal_solver, HYPRE_Int]?;
        if max_iters.is_negative() {
            return Err(HypreError::HypreGenericError);
        }
        config.max_iters = Some(max_iters as usize);
        let boolean: HYPRE_Int =
            get_parameter![HYPRE_PCGGetTwoNorm, self.internal_solver, HYPRE_Int]?;
        config.two_norm = Some(boolean != 0);

        let boolean: HYPRE_Int =
            get_parameter![HYPRE_PCGGetRelChange, self.internal_solver, HYPRE_Int]?;
        config.rel_change = Some(boolean != 0);
        Ok(config)
    }
}

impl LinearSolver for PCGSolver {
    /// Solves a linear system using Preconditioned Conjugate Gradient algorithm.
    fn solve(&self, mat: Matrix, rhs: Vector, x: Vector) -> HypreResult<IterativeSolverStatus> {
        unsafe {
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
