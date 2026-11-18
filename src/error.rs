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
    #[error("Invalid range: start is greater than end")]
    InvalidRange,
    #[error("hypre is already initialized")]
    AlreadyInitialized,
    #[error(
        "Entry above the diagonal: symmetric matrices take lower-triangle entries (col <= row)"
    )]
    NotLowerTriangular,
    #[error("Matrix is not symmetric")]
    NotSymmetric,
    #[error("Operation requires a matrix in IJ format")]
    NotIJMatrix,
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
        let _lock = crate::utils::hypre_error_lock();
        match HypreError::new(hypre_sys::HYPRE_ERROR_MEMORY as HYPRE_Int) {
            HypreMemoryError => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(hypre_sys::HYPRE_ERROR_ARG as HYPRE_Int) {
            HypreArgError(_) => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(hypre_sys::HYPRE_ERROR_CONV as HYPRE_Int) {
            HypreConvError => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(hypre_sys::HYPRE_ERROR_GENERIC as HYPRE_Int) {
            HypreGenericError => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(42) {
            UnknowError => {}
            _ => panic!("Incorrect error"),
        }
        // A combination of error bits is not one of the known codes.
        let both = (hypre_sys::HYPRE_ERROR_MEMORY | hypre_sys::HYPRE_ERROR_CONV) as HYPRE_Int;
        assert!(matches!(HypreError::new(both), UnknowError));
    }

    /// Building the error from a code clears that code from hypre's global
    /// error flag, so it does not leak into the next hypre call.
    #[test]
    fn error_new_clears_hypre_error_flag() {
        let _lock = crate::utils::hypre_error_lock();
        // Destroying a null handle is a hypre error that sets the flag.
        let code = unsafe { hypre_sys::HYPRE_IJMatrixDestroy(std::ptr::null_mut()) };
        assert_ne!(code, 0);
        assert_eq!(unsafe { hypre_sys::HYPRE_GetError() }, code);
        let _ = HypreError::new(code);
        assert_eq!(unsafe { hypre_sys::HYPRE_GetError() }, 0);
    }

    #[test]
    fn error_messages() {
        assert_eq!(
            HypreArgError(3).to_string(),
            "Hypre internal HYPRE_ERROR_ARG on argument 3"
        );
        assert_eq!(
            HypreError::InvalidRange.to_string(),
            "Invalid range: start is greater than end"
        );
        // Conversion errors keep the message of the original error.
        let Err(conversion) = u8::try_from(256u32) else {
            panic!("256 fits in a u8");
        };
        assert_eq!(
            HypreError::from(conversion).to_string(),
            conversion.to_string()
        );
        assert!(matches!(
            HypreError::from(conversion),
            HypreError::ConversionError(_)
        ));
    }
}
