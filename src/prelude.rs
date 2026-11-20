//! The types and functions most programs need, in one import.
//!
//! ```
//! use hypre_rs::prelude::*;
//! ```
//!
//! It brings in the context, matrices and vectors, the solvers'
//! configuration builders, option enums and runtime-choice enums, the solver traits for
//! generic code, and the preconditioners. Solver and configuration types
//! themselves (`PCGSolver`, `GMRESSolverConfig`, ...) are rarely named in
//! user code, so they stay in [solvers](crate::solvers) and
//! [preconditioners](crate::preconditioners).
//!
//! # Example
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use hypre_rs::prelude::*;
//!
//! let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
//! let comm = universe.world();
//! let ctx = hypre_rs::initialize(&universe)?;
//!
//! let rows = distribute_rows(&comm, 10);
//! let mut a = ctx.symmetric_matrix_builder(&comm, rows.clone())?;
//! a.add(rows.clone().map(|i| (i, i, 2.0)))?;
//! let a = a.assemble()?;
//! let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |_| 2.0)?;
//! let mut x = ParVector::zeros(&ctx, &comm, rows)?;
//!
//! let config = PCGSolverConfigBuilder::default().tol(1e-12).build()?;
//! let solver = ctx
//!     .pcg_solver(&comm, config)?
//!     .with_preconditioner(JacobiPreconditioner)?;
//! let status = solver.solve(&a, &b, &mut x)?;
//! assert!(status.converged);
//! assert!(x.local_values()?.iter().all(|&v| (v - 1.0).abs() < 1e-12));
//! # drop((solver, a, b, x));
//! # ctx.finalize()?;
//! # Ok(())
//! # }
//! ```

pub use crate::matrix::{
    CSRMatrix, DistributedMatrixBuilder, LocalMatrixBuilder, Matrix, SymmetricMatrix,
    SymmetricMatrixBuilder,
};
pub use crate::preconditioners::{
    AMGPreconditionerConfigBuilder, JacobiPreconditioner, Preconditioner,
    SymmetricAMGPreconditionerConfigBuilder, SymmetricPreconditioner, SymmetricSmoother,
};
pub use crate::solvers::{
    AMGSolverConfigBuilder, BiCGSTABSolverConfigBuilder, COGMRESSolverConfigBuilder, CoarsenType,
    ConvergeType, CycleType, FlexGMRESSolverConfigBuilder, GMRESSolverConfigBuilder,
    GramSchmidtPasses, InterpType, IterativeSolverStatus, LGMRESSolverConfigBuilder, LinearSolver,
    PCGSolverConfigBuilder, Smoother, Solver, SymmetricLinearSolver, SymmetricSolver,
};
pub use crate::vector::{DistributedVectorBuilder, LocalVectorBuilder, ParVector};
pub use crate::{distribute_rows, HypreContext, HypreError};
