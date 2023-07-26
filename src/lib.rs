//! hypre linear solvers in Rust
//!
//! This crate provides a Rust designed interface on [hypre] library.
//!
//! # Motivating example
//!
//! Here is a small example of a call to hypre's CG solver.
//! ```
//! # fn main() -> Result<(), hypre_rs::HypreError> {
//! extern crate hypre_rs;
//! # use mpi::initialize;
//! # use mpi::topology::Communicator;
//! use hypre_rs::{Matrix, Vector};
//! use hypre_rs::matrix::IJMatrix;
//! use hypre_rs::vector::IJVector;
//! use hypre_rs::solvers::{PCGSolverConfigBuilder, PCGSolver, SymmetricLinearSolver};
//! use hypre_rs::Vector::IJ;
//! use crate::hypre_rs::solvers::LinearSolver;
//!
//! let mpi_comm = mpi::initialize().unwrap().world();
//!
//! let mut matrix = Matrix::IJ(IJMatrix::new(&mpi_comm, (0, 12), (0, 12))?);
//! let rhs = Vector::IJ(IJVector::new(&mpi_comm, (0, 12))?);
//! let mut x = Vector::IJ(IJVector::new(&mpi_comm, (0,12))?);
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
//! let solver = PCGSolver::new(&mpi_comm, my_parameters)?;
//!
//! match solver.solve(&mut matrix, &rhs, &mut x) {
//!     Ok(info) => println!("Solver has converged: {}", info),
//!     Err(e) => return Err(e),
//! }
//!
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
