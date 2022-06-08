use std::ptr::{null, null_mut};

use super::GenericIterativeSolverConfig;
use hypre_sys::*;

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
            false
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

impl PCGSolver {
    fn check_hypre_error(self, err: i32) -> Option<PCGSolver> {
        if err != 0 {
            None
        } else {
            Some(self)
        }
    }

    pub fn new(config: PCGSolverConfig) -> Option<Self> {
        let mut solver = PCGSolver {
            internal_solver: null_mut(),
        };
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
}
