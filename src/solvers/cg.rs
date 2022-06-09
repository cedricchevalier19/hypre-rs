use std::ptr::{null, null_mut};

use super::GenericIterativeSolverConfig;
use hypre_sys::*;
use mpi::{
    topology::{AsCommunicator, SystemCommunicator},
    traits::AsRaw,
    *,
};

pub struct PCGSolver {
    internal_solver: HYPRE_Solver,
}

#[derive(Default, Debug)]
pub struct PCGSolverConfig {
    pub generic: GenericIterativeSolverConfig,
    pub recompute_residual: Option<bool>,
    pub recompute_residual_period: Option<usize>,
}

impl From<GenericIterativeSolverConfig> for PCGSolverConfig {
    fn from(generic: GenericIterativeSolverConfig) -> Self {
        PCGSolverConfig {
            generic,
            recompute_residual: None,
            recompute_residual_period: None,
        }
    }
}

impl PCGSolverConfig {
    pub fn validate(&self) -> bool {
        if let Some(p) = self.recompute_residual_period {
            p > 0 && self.generic.validate()
        } else {
            self.generic.validate()
        }
    }
}

macro_rules! set_parameter {
    ( $func:expr, $obj:expr, $param:expr ) => {{
        if let Some(p_value) = $param {
            let err = unsafe { $func($obj, p_value.try_into().unwrap()) };
            if err != 0 {
                return None;
            }
        }
    }};
}

macro_rules! get_parameter {
    ( $func:expr, $obj:expr, $t:ty ) => {{
        let mut p_t: $t = Default::default();
        let err = unsafe { $func($obj, &mut p_t) };
        if err != 0 {
            None
        } else {
            Some(p_t.try_into().unwrap())
        }
    }};
}

impl PCGSolver {
    fn check_hypre_error(self, err: i32) -> Option<PCGSolver> {
        if err != 0 {
            None
        } else {
            Some(self)
        }
    }

    pub fn new(comm: impl topology::Communicator, config: PCGSolverConfig) -> Option<Self> {
        let mut solver = PCGSolver {
            internal_solver: null_mut(),
        };
        unsafe {
            let h_solver = &mut solver.internal_solver as *mut _ as *mut HYPRE_Solver;
            if HYPRE_ParCSRPCGCreate(comm.as_raw(), h_solver) != 0 {
                return None;
            }
        }
        solver.config(config)
    }

    pub fn config(self, config: PCGSolverConfig) -> Option<Self> {
        if !config.validate() {
            return None;
        }
        set_parameter![HYPRE_PCGSetTol, self.internal_solver, config.generic.tol];
        set_parameter![
            HYPRE_PCGSetAbsoluteTol,
            self.internal_solver,
            config.generic.abs_tol
        ];
        set_parameter![
            HYPRE_PCGSetResidualTol,
            self.internal_solver,
            config.generic.res_tol
        ];
        set_parameter![
            HYPRE_PCGSetAbsoluteTolFactor,
            self.internal_solver,
            config.generic.abs_tol_fact
        ];
        set_parameter![
            HYPRE_PCGSetMaxIter,
            self.internal_solver,
            config.generic.max_iters
        ];
        set_parameter![
            HYPRE_PCGSetTwoNorm,
            self.internal_solver,
            config.generic.two_norm
        ];
        set_parameter![
            HYPRE_PCGSetRelChange,
            self.internal_solver,
            config.generic.rel_change
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

        Some(self)
    }

    pub fn current_config(&self) -> PCGSolverConfig {
        let mut config: PCGSolverConfig = Default::default();

        config.generic.tol = get_parameter![HYPRE_PCGGetTol, self.internal_solver, HYPRE_Real];
        config.generic.res_tol =
            get_parameter![HYPRE_PCGGetResidualTol, self.internal_solver, HYPRE_Real];
        config.generic.abs_tol_fact = get_parameter![
            HYPRE_PCGGetAbsoluteTolFactor,
            self.internal_solver,
            HYPRE_Real
        ];
        config.generic.max_iters =
            get_parameter![HYPRE_PCGGetMaxIter, self.internal_solver, HYPRE_Int];
        config.generic.two_norm =
            get_parameter![HYPRE_PCGGetTwoNorm, self.internal_solver, HYPRE_Int]
                .map(|v: HYPRE_Int| v != 0);
        config.generic.rel_change =
            get_parameter![HYPRE_PCGGetRelChange, self.internal_solver, HYPRE_Int]
                .map(|v: HYPRE_Int| v != 0);
        config
    }

    pub fn solve(&self, mat: HYPRE_Matrix, rhs: HYPRE_Vector, x: HYPRE_Vector) {
        unsafe {
            HYPRE_PCGSolve(self.internal_solver, mat, rhs, x);
        }
    }
}

impl Drop for PCGSolver {
    fn drop(&mut self) {
        unsafe {
            HYPRE_ParCSRPCGDestroy(self.internal_solver);
        }
    }
}

#[cfg(test)]
mod tests {
    use mpi::traits::AsRaw;

    use super::*;
    #[test]
    fn config_test() {
        let universe = mpi::initialize().unwrap();
        let solver = PCGSolver::new(universe.world(), Default::default()).unwrap();

        let parameters = solver.current_config();
        println!("{:?}", parameters);
    }
}
