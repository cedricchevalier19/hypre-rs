//! Preconditioners for the Krylov solvers.
//!
//! They come in two families:
//!
//! - [SymmetricPreconditioner]: symmetric positive definite preconditioners,
//!   which keep the preconditioned conjugate gradient ([PCGSolver]) valid.
//!   They are the only ones it accepts.
//! - [Preconditioner]: every preconditioner, for the solvers of general
//!   matrices (GMRES and its variants, BiCGSTAB). A symmetric preconditioner
//!   converts into one.
//!
//! A solver owns its preconditioner: pass it to `with_preconditioner`.
//!
//! [PCGSolver]: crate::solvers::PCGSolver
//!
//! # Example
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use mpi::topology::Communicator;
//! use hypre_rs::preconditioners::SymmetricAMGPreconditionerConfigBuilder;
//! use hypre_rs::solvers::{PCGSolverConfigBuilder, SymmetricLinearSolver, SymmetricSolver};
//!
//! # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
//! # let comm = universe.world();
//! # let ctx = hypre_rs::initialize(&universe)?;
//! # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
//! // The 1D Laplacian tridiag(-1, 2, -1), from its lower triangle.
//! let n = 100;
//! let rows = (rank * n / size)..((rank + 1) * n / size);
//! let mut a = ctx.symmetric_matrix(&comm, rows.clone())?;
//! a.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
//! a.add_elements(rows.clone().filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
//! let a = a.assemble()?;
//! let mut b = ctx.ij_vector(&comm, rows.clone())?;
//! b.add_elements(rows.clone().map(|i| (i, 1.0)))?;
//! let mut x = ctx.ij_vector(&comm, rows.clone())?;
//! x.add_elements(rows.clone().map(|i| (i, 0.0)))?;
//!
//! // CG preconditioned by one AMG V-cycle.
//! let amg = ctx.symmetric_amg_preconditioner(&comm, Default::default())?;
//! let config = PCGSolverConfigBuilder::default().tol(1e-10).build()?;
//! let solver = ctx.pcg_solver(&comm, config)?.with_preconditioner(amg)?;
//! let status = SymmetricSolver::CG(solver).solve(&a, &b, &mut x)?;
//! assert!(status.converged);
//! assert!(status.num_iters < 10);
//! # drop((a, b, x));
//! # ctx.finalize()?;
//! # Ok(())
//! # }
//! ```

use hypre_sys::{
    HYPRE_ParCSRDiagScale, HYPRE_ParCSRDiagScaleSetup, HYPRE_PtrToParSolverFcn, HYPRE_Solver,
};
use std::ptr::null_mut;

mod amg;
#[cfg(test)]
mod tests;

pub use amg::{
    AMGPreconditioner, AMGPreconditionerConfig, AMGPreconditionerConfigBuilder,
    SymmetricAMGPreconditioner, SymmetricAMGPreconditionerConfig,
    SymmetricAMGPreconditionerConfigBuilder, SymmetricSmoother,
};

/// What hypre's `SetPrecond` functions take.
#[derive(Debug)]
pub(crate) struct RawPreconditioner {
    pub(crate) solve: HYPRE_PtrToParSolverFcn,
    pub(crate) setup: HYPRE_PtrToParSolverFcn,
    pub(crate) solver: HYPRE_Solver,
}

/// Jacobi preconditioner: scaling by the inverse of the diagonal.
///
/// It is symmetric, and needs a matrix without zeros on its diagonal.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct JacobiPreconditioner;

impl JacobiPreconditioner {
    fn raw(&self) -> RawPreconditioner {
        RawPreconditioner {
            solve: Some(HYPRE_ParCSRDiagScale),
            setup: Some(HYPRE_ParCSRDiagScaleSetup),
            // Diagonal scaling has no hypre object.
            solver: null_mut(),
        }
    }
}

/// Symmetric positive definite preconditioner, for the conjugate gradient
/// solver; it also works with the solvers of general matrices (see
/// [Preconditioner]).
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum SymmetricPreconditioner<'h> {
    /// Diagonal scaling.
    Jacobi(JacobiPreconditioner),
    /// Algebraic multigrid with a symmetric cycle.
    AMG(SymmetricAMGPreconditioner<'h>),
}

impl SymmetricPreconditioner<'_> {
    pub(crate) fn raw(&self) -> RawPreconditioner {
        match self {
            Self::Jacobi(p) => p.raw(),
            Self::AMG(p) => p.raw(),
        }
    }
}

/// Preconditioner for the solvers of general matrices.
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum Preconditioner<'h> {
    /// Any symmetric preconditioner.
    Symmetric(SymmetricPreconditioner<'h>),
    /// Algebraic multigrid, with any smoother.
    AMG(AMGPreconditioner<'h>),
}

impl Preconditioner<'_> {
    pub(crate) fn raw(&self) -> RawPreconditioner {
        match self {
            Self::Symmetric(p) => p.raw(),
            Self::AMG(p) => p.raw(),
        }
    }
}

impl<'h> From<SymmetricPreconditioner<'h>> for Preconditioner<'h> {
    fn from(p: SymmetricPreconditioner<'h>) -> Self {
        Self::Symmetric(p)
    }
}

impl From<JacobiPreconditioner> for SymmetricPreconditioner<'_> {
    fn from(p: JacobiPreconditioner) -> Self {
        Self::Jacobi(p)
    }
}

impl From<JacobiPreconditioner> for Preconditioner<'_> {
    fn from(p: JacobiPreconditioner) -> Self {
        Self::Symmetric(p.into())
    }
}

impl<'h> From<SymmetricAMGPreconditioner<'h>> for SymmetricPreconditioner<'h> {
    fn from(p: SymmetricAMGPreconditioner<'h>) -> Self {
        Self::AMG(p)
    }
}

impl<'h> From<SymmetricAMGPreconditioner<'h>> for Preconditioner<'h> {
    fn from(p: SymmetricAMGPreconditioner<'h>) -> Self {
        Self::Symmetric(p.into())
    }
}

impl<'h> From<AMGPreconditioner<'h>> for Preconditioner<'h> {
    fn from(p: AMGPreconditioner<'h>) -> Self {
        Self::AMG(p)
    }
}
