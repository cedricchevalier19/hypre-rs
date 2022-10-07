use enum_dispatch::enum_dispatch;
use hypre_sys::HYPRE_Vector;

macro_rules! set_parameter {
    ( $func:expr, $obj:expr, $param:expr ) => {{
        if let Some(p_value) = $param {
            check_hypre!(unsafe { $func($obj, p_value.try_into().unwrap()) });
        }
    }};
}

macro_rules! get_parameter {
    ( $func:expr, $obj:expr, $t:ty ) => {{
        let mut p_t: $t = Default::default();
        let err = unsafe { $func($obj, &mut p_t) };
        if err != 0 {
            Err(HypreError::new(err))
        } else {
            Ok(p_t.try_into().unwrap())
        }
    }};
}

mod cg;

use std::fmt;
use std::fmt::Formatter;

use crate::error::HypreError;
use crate::matrix::Matrix;
pub use cg::PCGSolver;
pub use cg::PCGSolverConfig;

/// Solver status information
#[derive(Debug, Clone)]
pub struct IterativeSolverStatus {
    /// Number of iterations done
    pub num_iters: usize,
    /// Relative residual norm
    pub res_num: f64,
    /// Whether the solver has converged
    pub converged: bool,
}

impl fmt::Display for IterativeSolverStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.converged {
            write!(
                f,
                "Solver has converged in {} iterations, reaching a relative residual of {}",
                self.num_iters, self.res_num
            )
        } else {
            write!(
                f,
                "Solver has NOT converged in {} iterations, reaching a relative residual of {}",
                self.num_iters, self.res_num
            )
        }
    }
}

#[enum_dispatch]
pub trait LinearSolver {
    fn solve(
        &self,
        mat: Matrix,
        rhs: HYPRE_Vector,
        x: HYPRE_Vector,
    ) -> Result<IterativeSolverStatus, HypreError>;
}

#[enum_dispatch(LinearSolver)]
enum Solver {
    CG(PCGSolver),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let universe = mpi::initialize().unwrap();
        let solver = Solver::CG(PCGSolver::new(universe.world(), Default::default()).unwrap());
        //solver.solve();
    }
}
