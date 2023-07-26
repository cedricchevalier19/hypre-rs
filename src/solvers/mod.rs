mod boomer_amg;
mod cg;

use std::fmt;
use std::fmt::Formatter;

use crate::matrix::Matrix;
use crate::{HypreResult, Vector};
pub use boomer_amg::BoomerAMG;
pub use cg::PCGSolver;
pub use cg::PCGSolverConfig;
pub use cg::PCGSolverConfigBuilder;
use hypre_sys::{HYPRE_PtrToSolverFcn, HYPRE_Solver};

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

pub trait InternalLinearPreconditioner {
    fn get_precond(&self) -> HYPRE_PtrToSolverFcn;
    fn get_precond_setup(&self) -> HYPRE_PtrToSolverFcn;
    fn get_internal(&self) -> HYPRE_Solver;
}

pub trait LinearPreconditioner {
    fn precond_descriptor(&self) -> &dyn InternalLinearPreconditioner;
}

pub trait SymmetricLinearPreconditioner {
    fn precond_descriptor(&self) -> &dyn InternalLinearPreconditioner;
}

// impl<T> LinearPreconditioner for T
// where
//     T: SymmetricLinearPreconditioner,
// {
//     fn precond_descriptor(&self) -> &dyn InternalLinearPreconditioner {
//         self.precond_descriptor()
//     }
// }

pub trait LinearSolver {
    fn solve(
        &self,
        mat: &mut Matrix,
        rhs: &Vector,
        x: &mut Vector,
    ) -> HypreResult<IterativeSolverStatus>;
}

pub trait SymmetricLinearSolver {
    fn set_precond(&mut self, precond: impl SymmetricLinearPreconditioner);

    fn solve(
        &self,
        mat: &mut Matrix,
        rhs: &Vector,
        x: &mut Vector,
    ) -> HypreResult<IterativeSolverStatus>;
}

// impl<T> LinearSolver for T
// where
//     T: SymmetricLinearSolver,
// {
//     fn solve(
//         &self,
//         mat: &mut Matrix,
//         rhs: &Vector,
//         x: &mut Vector,
//     ) -> HypreResult<IterativeSolverStatus> {
//         self.solve(mat, rhs, x)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let _mpi_comm = mpi::initialize().unwrap().world();
        // let _solver = Solver::CG(PCGSolver::new(&mpi_comm, Default::default()).unwrap());
        //solver.solve();
    }
}
