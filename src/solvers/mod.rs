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

/// Returns [HypreError::InvalidParameterPositive] from a builder's
/// `validate` if the optional parameter `$param` is set and negative.
macro_rules! check_positive_parameter {
    ( $obj:expr, $param:ident) => {{
        if let Some(Some($param)) = $obj.$param {
            if $param < 0.into() {
                return Err(HypreError::InvalidParameterPositive);
            }
        }
    }};
}

mod cg;
mod krylov;

use std::fmt;
use std::fmt::Formatter;

use crate::matrix::{Matrix, SymmetricMatrix};
use crate::{HypreResult, Vector};
pub use cg::PCGSolver;
pub use cg::PCGSolverConfig;
pub use cg::PCGSolverConfigBuilder;
pub use krylov::{
    BiCGSTABSolver, BiCGSTABSolverConfig, BiCGSTABSolverConfigBuilder, COGMRESSolver,
    COGMRESSolverConfig, COGMRESSolverConfigBuilder, FlexGMRESSolver, FlexGMRESSolverConfig,
    FlexGMRESSolverConfigBuilder, GMRESSolver, GMRESSolverConfig, GMRESSolverConfigBuilder,
    LGMRESSolver, LGMRESSolverConfig, LGMRESSolverConfigBuilder,
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
