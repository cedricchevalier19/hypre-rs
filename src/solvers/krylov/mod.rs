//! Krylov solvers: CG for symmetric matrices, GMRES and its variants and
//! BiCGSTAB for general ones.
//!
//! They only differ by their hypre functions and parameters, so the
//! `hypre_solver!` macro generates each of them.

mod bicgstab;
mod cg;
mod gmres;
#[cfg(test)]
mod tests;

pub use bicgstab::{BiCGSTABSolver, BiCGSTABSolverConfig, BiCGSTABSolverConfigBuilder};
pub use cg::{PCGSolver, PCGSolverConfig, PCGSolverConfigBuilder};
pub use gmres::{
    COGMRESSolver, COGMRESSolverConfig, COGMRESSolverConfigBuilder, FlexGMRESSolver,
    FlexGMRESSolverConfig, FlexGMRESSolverConfigBuilder, GMRESSolver, GMRESSolverConfig,
    GMRESSolverConfigBuilder, LGMRESSolver, LGMRESSolverConfig, LGMRESSolverConfigBuilder,
};
