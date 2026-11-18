//! hypre linear solvers in Rust
//!
//! This crate provides a Rust designed interface on [hypre] library.
//!
//! # Motivating example
//!
//! Here is a small example of a call to hypre's CG solver.
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! extern crate hypre_rs;
//! # use mpi::initialize;
//! # use mpi::topology::Communicator;
//! use hypre_rs::{Matrix, Vector};
//! use hypre_rs::solvers::{PCGSolverConfigBuilder, Solver};
//! use hypre_rs::solvers::LinearSolver;
//!
//! let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
//! let mpi_comm = universe.world();
//!
//! let ctx = hypre_rs::initialize(&universe)?;
//!
//! // Split the 12 rows between the MPI ranks
//! let (rank, size) = (mpi_comm.rank() as u32, mpi_comm.size() as u32);
//! let rows = (rank * 12 / size)..((rank + 1) * 12 / size);
//!
//! let mut matrix = Matrix::IJ(ctx.ij_matrix(&mpi_comm, rows.clone(), rows.clone())?);
//! let rhs = Vector::IJ(ctx.ij_vector(&mpi_comm, rows.clone())?);
//! let mut x = Vector::IJ(ctx.ij_vector(&mpi_comm, rows)?);
//!
//! // CG solver parameters
//! let my_parameters = PCGSolverConfigBuilder::default()
//!             .tol(1e-9)
//!             .max_iters(500usize)
//!             .two_norm(true)
//!             .recompute_residual_period(8usize)
//!             .build()?;
//!
//! // Create new CG solver with previous parameters
//! let solver = Solver::CG(ctx.pcg_solver(&mpi_comm, my_parameters)?);
//!
//! let info = solver.solve(&mut matrix, &rhs, &mut x)?;
//! println!("Solver has converged: {}", info);
//!
//! // hypre objects must be dropped before hypre is finalized (checked by the compiler)
//! drop((solver, matrix, rhs, x));
//! ctx.finalize()?;
//! # Ok(())
//! # }
//! ```

#![warn(
    missing_copy_implementations,
    missing_debug_implementations,
    rust_2018_idioms
)]

pub mod error;
#[macro_use]
mod utils;
// Macros from utils are now available.

pub mod matrix;
pub mod solvers;
pub mod vector;

type HypreResult<T> = Result<T, error::HypreError>;
pub use error::HypreError;
pub use matrix::Matrix;
pub use vector::Vector;

mod context;
pub use context::{initialize, HypreContext};
