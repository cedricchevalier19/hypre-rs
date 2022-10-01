use crate::error::HypreError::{
    HypreArgError, HypreConvError, HypreGenericError, HypreMemoryError, UnknowError,
};
use hypre_sys::{HYPRE_ClearError, HYPRE_GetErrorArg, HYPRE_Int};
use thiserror::Error;

#[derive(Debug, Error)]
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
    #[error(transparent)]
    Other(#[from] anyhow::Error), // source and Display delegate to anyhow::Error
}

impl HypreError {
    pub(crate) fn new(error_num: HYPRE_Int) -> Self {
        let out = match error_num as u32 {
            hypre_sys::HYPRE_ERROR_MEMORY => HypreMemoryError,
            hypre_sys::HYPRE_ERROR_ARG => {
                let argnum = unsafe { HYPRE_GetErrorArg().try_into().unwrap() };
                HypreArgError(argnum)
            }
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
        match HypreError::new(hypre_sys::HYPRE_ERROR_MEMORY as i32) {
            HypreMemoryError => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(hypre_sys::HYPRE_ERROR_ARG as i32) {
            HypreArgError(_) => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(hypre_sys::HYPRE_ERROR_CONV as i32) {
            HypreConvError => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(hypre_sys::HYPRE_ERROR_GENERIC as i32) {
            HypreGenericError => {}
            _ => panic!("Incorrect error"),
        };
        match HypreError::new(42) {
            UnknowError => {}
            _ => panic!("Incorrect error"),
        }
    }
}
