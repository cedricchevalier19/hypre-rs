pub mod error;
#[macro_use]
mod utils;
// Macros from utils are now available.

mod matrix;
pub mod solvers;

type HypreResult<T> = Result<T, error::HypreError>;
pub use error::HypreError;
pub use matrix::Matrix;
