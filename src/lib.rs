pub mod error;
#[macro_escape]
mod utils;
// Macros from utils are now available.

mod matrix;
pub mod solvers;

type HypreResult<T> = Result<T, error::HypreError>;
