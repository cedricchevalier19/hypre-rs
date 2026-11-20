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
use crate::vector::ParVector;
use crate::HypreResult;
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
/// They accept any [Matrix], including a [SymmetricMatrix] and, when the
/// storage is chosen at runtime, a `&dyn Matrix`.
#[enum_dispatch]
pub trait LinearSolver {
    /// Solves `mat * x = rhs`, starting from the current value of `x`.
    ///
    /// Collective: every MPI rank of the matrix must call it.
    ///
    /// # Errors
    ///
    /// [HypreError::HypreConvError](crate::HypreError::HypreConvError) if
    /// the solver did not converge within its maximum number of iterations,
    /// or another error returned by hypre.
    fn solve<M: Matrix + ?Sized>(
        &self,
        mat: &M,
        rhs: &ParVector<'_>,
        x: &mut ParVector<'_>,
    ) -> HypreResult<IterativeSolverStatus>;
}

/// Solvers for general matrices, chosen at runtime.
///
/// Only needed when the solver is chosen at runtime: every solver has its own
/// `solve` method. Each solver type converts into its variant with `into()`.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use mpi::topology::Communicator;
/// use hypre_rs::solvers::{GMRESSolverConfigBuilder, Solver};
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
/// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
/// // Upper bidiagonal, not symmetric: 2 on the diagonal, -1 above.
/// let mut a = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
/// a.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
/// a.add_elements(rows.clone().filter(|&i| i < 9).map(|i| (i, i + 1, -1.0)))?;
/// let a = a.assemble()?;
///
/// // b = A * [1, ..., 1]
/// let mut b = ctx.ij_vector(&comm, rows.clone())?;
/// b.add_elements(rows.clone().map(|i| (i, if i < 9 { 1.0 } else { 2.0 })))?;
/// let b = b.assemble()?;
/// let mut x = ctx.ij_vector(&comm, rows)?.assemble()?;
///
/// // For example from a configuration file.
/// let use_bicgstab = false;
/// let solver: Solver<'_> = if use_bicgstab {
///     ctx.bicgstab_solver(&comm, Default::default())?.into()
/// } else {
///     let config = GMRESSolverConfigBuilder::default().tol(1e-10).build()?;
///     ctx.gmres_solver(&comm, config)?.into()
/// };
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

impl Solver<'_> {
    /// Solves `mat * x = rhs` with the chosen solver; same as
    /// [LinearSolver::solve], without importing the trait.
    ///
    /// # Errors
    ///
    /// See [LinearSolver::solve].
    pub fn solve<M: Matrix + ?Sized>(
        &self,
        mat: &M,
        rhs: &ParVector<'_>,
        x: &mut ParVector<'_>,
    ) -> HypreResult<IterativeSolverStatus> {
        LinearSolver::solve(self, mat, rhs, x)
    }
}

/// Solver for linear systems whose matrix is symmetric.
#[enum_dispatch]
pub trait SymmetricLinearSolver {
    /// Solves `mat * x = rhs`, starting from the current value of `x`.
    ///
    /// Collective: every MPI rank of the matrix must call it.
    fn solve<M: Matrix + ?Sized>(
        &self,
        mat: &SymmetricMatrix<M>,
        rhs: &ParVector<'_>,
        x: &mut ParVector<'_>,
    ) -> HypreResult<IterativeSolverStatus>;
}

/// Solvers for symmetric matrices, chosen at runtime.
///
/// Only needed when the solver is chosen at runtime: every solver has its own
/// `solve` method. Each solver type converts into its variant with `into()`.
#[enum_dispatch(SymmetricLinearSolver)]
#[derive(Debug)]
pub enum SymmetricSolver<'h> {
    /// Conjugate Gradient, for symmetric positive definite matrices.
    CG(PCGSolver<'h>),
}

impl SymmetricSolver<'_> {
    /// Solves `mat * x = rhs` with the chosen solver; same as
    /// [SymmetricLinearSolver::solve], without importing the trait.
    ///
    /// # Errors
    ///
    /// See [SymmetricLinearSolver::solve].
    pub fn solve<M: Matrix + ?Sized>(
        &self,
        mat: &SymmetricMatrix<M>,
        rhs: &ParVector<'_>,
        x: &mut ParVector<'_>,
    ) -> HypreResult<IterativeSolverStatus> {
        SymmetricLinearSolver::solve(self, mat, rhs, x)
    }
}
