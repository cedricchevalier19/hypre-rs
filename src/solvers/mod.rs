use enum_dispatch::enum_dispatch;

#[macro_use]
mod macros;
#[macro_use]
pub(crate) mod solver_macro;
mod amg;
mod krylov;
#[cfg(test)]
mod tests;

use std::fmt;
use std::fmt::Formatter;

use crate::matrix::{Matrix, SymmetricMatrix};
use crate::{HypreResult, Vector};
pub use amg::{AMGSolver, AMGSolverConfig, AMGSolverConfigBuilder};
pub use krylov::{
    BiCGSTABSolver, BiCGSTABSolverConfig, BiCGSTABSolverConfigBuilder, COGMRESSolver,
    COGMRESSolverConfig, COGMRESSolverConfigBuilder, FlexGMRESSolver, FlexGMRESSolverConfig,
    FlexGMRESSolverConfigBuilder, GMRESSolver, GMRESSolverConfig, GMRESSolverConfigBuilder,
    LGMRESSolver, LGMRESSolverConfig, LGMRESSolverConfigBuilder, PCGSolver, PCGSolverConfig,
    PCGSolverConfigBuilder,
};

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

/// Solver for general linear systems.
///
/// A [SymmetricMatrix] is a [Matrix] (it dereferences to one), so these
/// solvers also accept `&symmetric_matrix`.
#[enum_dispatch]
pub trait LinearSolver {
    /// Solves `mat * x = rhs`, starting from the current value of `x`.
    ///
    /// Assembles `mat`, `rhs` and `x`, so it is collective.
    ///
    /// # Errors
    ///
    /// [HypreError::HypreConvError](crate::HypreError::HypreConvError) if
    /// the solver did not converge within its maximum number of iterations,
    /// or another error returned by hypre.
    fn solve(
        &self,
        mat: &Matrix<'_>,
        rhs: &Vector<'_>,
        x: &mut Vector<'_>,
    ) -> HypreResult<IterativeSolverStatus>;
}

/// Solvers for general matrices, chosen at runtime.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use mpi::topology::Communicator;
/// use hypre_rs::solvers::{GMRESSolverConfigBuilder, LinearSolver, Solver};
/// use hypre_rs::{Matrix, Vector};
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
/// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
/// // Upper bidiagonal, not symmetric: 2 on the diagonal, -1 above.
/// let mut a = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
/// a.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
/// a.add_elements(rows.clone().filter(|&i| i < 9).map(|i| (i, i + 1, -1.0)))?;
/// let a = Matrix::IJ(a);
///
/// // b = A * [1, ..., 1]
/// let mut b = ctx.ij_vector(&comm, rows.clone())?;
/// b.add_elements(rows.clone().map(|i| (i, if i < 9 { 1.0 } else { 2.0 })))?;
/// let b = Vector::IJ(b);
/// let mut x = Vector::IJ(ctx.ij_vector(&comm, rows)?);
///
/// let config = GMRESSolverConfigBuilder::default().tol(1e-10).build()?;
/// let solver = Solver::GMRES(ctx.gmres_solver(&comm, config)?);
/// let status = solver.solve(&a, &b, &mut x)?;
/// assert!(status.converged);
/// # drop((solver, a, b, x));
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
#[enum_dispatch(LinearSolver)]
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum Solver<'h> {
    /// Restarted GMRES.
    GMRES(GMRESSolver<'h>),
    /// Flexible GMRES.
    FlexGMRES(FlexGMRESSolver<'h>),
    /// Loose GMRES.
    LGMRES(LGMRESSolver<'h>),
    /// Communication-optimal GMRES.
    COGMRES(COGMRESSolver<'h>),
    /// Biconjugate Gradient Stabilized.
    BiCGSTAB(BiCGSTABSolver<'h>),
    /// Algebraic multigrid (BoomerAMG).
    AMG(AMGSolver<'h>),
}

/// Solver for linear systems whose matrix is symmetric.
#[enum_dispatch]
pub trait SymmetricLinearSolver {
    /// Solves `mat * x = rhs`, starting from the current value of `x`.
    ///
    /// Assembles `mat`, `rhs` and `x`, so it is collective.
    fn solve(
        &self,
        mat: &SymmetricMatrix<'_>,
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
