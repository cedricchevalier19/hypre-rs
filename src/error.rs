use crate::error::HypreError::{
    HypreArgError, HypreConvError, HypreGenericError, HypreMemoryError, UnknowError,
};
use crate::HypreError::ConversionError;
use hypre_sys::{HYPRE_ClearError, HYPRE_GetErrorArg, HYPRE_Int};
use std::convert::Infallible;
use std::num::TryFromIntError;
use thiserror::Error;

#[derive(Debug, Error, Copy, Clone)]
pub enum HypreError {
    #[error("Hypre internal HYPRE_ERROR_GENERIC")]
    HypreGenericError,
    #[error("Hypre internal HYPRE_ERROR_MEMORY")]
    HypreMemoryError,
    #[error("Hypre internal HYPRE_ERROR_ARG on argument {0}")]
    HypreArgError(u32),
    #[error("Hypre convergence error")]
    HypreConvError,
    #[error("MPI Error")]
    MpiError,
    #[error("Unknown error")]
    UnknowError,
    #[error("Invalid parameter, should be positive")]
    InvalidParameterPositive,
    #[error(transparent)]
    ConversionError(#[from] TryFromIntError),
    #[error(transparent)]
    NoError(#[from] Infallible),
}

impl HypreError {
    pub(crate) fn new(error_num: HYPRE_Int) -> Self {
        let out = match error_num as u32 {
            hypre_sys::HYPRE_ERROR_MEMORY => HypreMemoryError,
            hypre_sys::HYPRE_ERROR_ARG => unsafe {
                match HYPRE_GetErrorArg().try_into() {
                    Ok(argnum) => HypreArgError(argnum),
                    Err(e) => ConversionError(e),
                }
            },
            hypre_sys::HYPRE_ERROR_CONV => HypreConvError,
            hypre_sys::HYPRE_ERROR_GENERIC => HypreGenericError,
            _ => UnknowError,
        };
        unsafe {
            HYPRE_ClearError(error_num);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_create() {
        match HypreError::new(hypre_sys::HYPRE_ERROR_MEMORY as i64) {
            HypreMemoryError => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(hypre_sys::HYPRE_ERROR_ARG as i64) {
            HypreArgError(_) => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(hypre_sys::HYPRE_ERROR_CONV as i64) {
            HypreConvError => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(hypre_sys::HYPRE_ERROR_GENERIC as i64) {
            HypreGenericError => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(42) {
            UnknowError => {}
            _ => panic!("Incorrect error"),
        }
    }
}
