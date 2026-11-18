use enum_dispatch::enum_dispatch;

/// Calls the hypre setter `$func` on `$obj` if `$param` is `Some`, converting
/// the value to the setter's type; returns early on a conversion or hypre
/// error.
macro_rules! set_parameter {
    ( $func:expr, $obj:expr, $param:expr ) => {{
        if let Some(p_value) = $param {
            check_hypre!(unsafe { $func($obj, p_value.try_into()?) });
        }
    }};
}

/// Calls the hypre getter `$func` on `$obj`, reading a `$t`, and converts it
/// to the expected type. Evaluates to a `HypreResult`.
macro_rules! get_parameter {
    ( $func:expr, $obj:expr, $t:ty ) => {{
        let mut p_t: $t = Default::default();
        let err = unsafe { $func($obj, &mut p_t) };
        if err != 0 {
            Err(HypreError::new(err))
        } else {
            p_t.try_into().map_err(HypreError::from)
        }
    }};
}

mod cg;

use std::fmt;
use std::fmt::Formatter;

use crate::matrix::SymmetricMatrix;
use crate::{HypreResult, Vector};
pub use cg::PCGSolver;
pub use cg::PCGSolverConfig;
pub use cg::PCGSolverConfigBuilder;

/// Solver status information
#[derive(Debug, Clone, Copy)]
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

/// Solver for linear systems whose matrix is symmetric.
#[enum_dispatch]
pub trait SymmetricLinearSolver {
    /// Solves `mat * x = rhs`, starting from the current value of `x`.
    ///
    /// Assembles `mat`, `rhs` and `x`, so it is collective.
    fn solve(
        &self,
        mat: &mut SymmetricMatrix<'_>,
        rhs: &Vector<'_>,
        x: &mut Vector<'_>,
    ) -> HypreResult<IterativeSolverStatus>;
}

/// Solvers for symmetric matrices, chosen at runtime.
#[enum_dispatch(SymmetricLinearSolver)]
#[derive(Debug)]
pub enum SymmetricSolver<'h> {
    /// Conjugate Gradient, for symmetric positive definite matrices.
    CG(PCGSolver<'h>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_display() {
        let mut status = IterativeSolverStatus {
            num_iters: 12,
            res_num: 0.5,
            converged: true,
        };
        assert_eq!(
            status.to_string(),
            "Solver has converged in 12 iterations, reaching a relative residual of 0.5"
        );
        status.converged = false;
        assert_eq!(
            status.to_string(),
            "Solver has NOT converged in 12 iterations, reaching a relative residual of 0.5"
        );
    }
}
