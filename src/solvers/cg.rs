extern crate derive_builder;

use derive_builder::Builder;

use std::ptr::null_mut;

use hypre_sys::*;
use mpi;

pub struct PCGSolver {
    internal_solver: HYPRE_Solver,
}

#[derive(Default, Debug, Clone, Builder)]
#[builder(setter(into, strip_option), default)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct PCGSolverConfig {
    pub tol: Option<f64>,
    pub abs_tol: Option<f64>,
    pub res_tol: Option<f64>,
    pub abs_tol_fact: Option<f64>,
    pub conv_tol_fact: Option<f64>,
    pub stop_crit: Option<usize>,
    pub max_iters: Option<usize>,
    pub two_norm: Option<bool>,
    pub rel_change: Option<bool>,
    pub logging: Option<u32>,
    pub print_level: Option<u32>,
    pub recompute_residual: Option<bool>,
    pub recompute_residual_period: Option<usize>,
}

macro_rules! check_positive_parameter {
    ( $obj:expr, $param:ident) => {{
        if let Some(Some($param)) = $obj.$param {
            if $param < 0.into() {
                return Err("parameter must be positive".to_string());
            }
        }
    }};
}

impl PCGSolverConfigBuilder {
    fn validate(&self) -> Result<(), String> {
        check_positive_parameter![self, tol];
        check_positive_parameter![self, abs_tol];
        check_positive_parameter![self, res_tol];
        check_positive_parameter![self, conv_tol_fact];
        Ok(())
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

    pub fn new(comm: impl mpi::topology::Communicator, config: PCGSolverConfig) -> Option<Self> {
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
        set_parameter![
            HYPRE_PCGSetAbsoluteTolFactor,
            self.internal_solver,
            config.abs_tol_fact
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

        Some(self)
    }

    pub fn current_config(&self) -> PCGSolverConfig {
        let mut config: PCGSolverConfig = Default::default();

        config.tol = get_parameter![HYPRE_PCGGetTol, self.internal_solver, HYPRE_Real];
        config.res_tol = get_parameter![HYPRE_PCGGetResidualTol, self.internal_solver, HYPRE_Real];
        config.abs_tol_fact = get_parameter![
            HYPRE_PCGGetAbsoluteTolFactor,
            self.internal_solver,
            HYPRE_Real
        ];
        config.max_iters = get_parameter![HYPRE_PCGGetMaxIter, self.internal_solver, HYPRE_Int];
        config.two_norm = get_parameter![HYPRE_PCGGetTwoNorm, self.internal_solver, HYPRE_Int]
            .map(|v: HYPRE_Int| v != 0);
        config.rel_change = get_parameter![HYPRE_PCGGetRelChange, self.internal_solver, HYPRE_Int]
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
    use super::*;
    #[test]
    fn config_test() {
        let universe = mpi::initialize().unwrap();
        let solver = PCGSolver::new(universe.world(), Default::default()).unwrap();

        let parameters = solver.current_config();
        println!("{:?}", parameters);

        let my_parameters = PCGSolverConfigBuilder::default()
            .tol(1e-9)
            .max_iters(500usize)
            .two_norm(true)
            .recompute_residual_period(8usize)
            .build()
            .unwrap();
        let solver = PCGSolver::new(universe.world(), my_parameters.clone()).unwrap();
        let parameters = solver.current_config();
        println!("{:?}", parameters);
        assert_eq!(my_parameters.tol, parameters.tol);
        assert_eq!(my_parameters.max_iters, parameters.max_iters);
        assert_eq!(my_parameters.two_norm, parameters.two_norm);
    }
}
