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
//! use hypre_rs::Matrix;
//! use hypre_rs::matrix::IJMatrix;
//! use hypre_rs::solvers::{PCGSolverConfigBuilder, PCGSolver, Solver};
//!
//! let universe = mpi::initialize().unwrap();
//!
//! let matrix = Matrix::IJ(IJMatrix::new(universe, (0, 12), (0, 12))?);
//! let rhs = Vector(universe, (0, 12)?);
//! let b = Vector(universe, (0,12)?);
//!
//! // CG solver parameters
//! let my_parameters = PCGSolverConfigBuilder::default()
//! .tol(1e-9)
//!             .max_iters(500usize)
//!             .two_norm(true)
//!             .recompute_residual_period(8usize)
//!             .build()?;
//!
//! // Create new CG solver with previous parameters
//! let solver = Solver::CG(PCGSolver::new(universe.world(), my_parameters)?);
//!
//! match solver.solve(matrix, rhs, b) {
//!     Ok(info) => println!("Solver has converged: {}", info),
//!     Err(E) => return E,
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

type HypreResult<T> = Result<T, error::HypreError>;
pub use error::HypreError;
pub use matrix::Matrix;
