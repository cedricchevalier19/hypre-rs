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
    /// A solver did not converge within its maximum number of iterations;
    /// the status holds the iterations done and the residual reached.
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use hypre_rs::prelude::*;
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// # let rows = distribute_rows(&comm, 100);
    /// # let mut a = ctx.symmetric_matrix_builder(&comm, rows.clone())?;
    /// # a.add(rows.clone().map(|i| (i, i, 2.0)))?;
    /// # a.add(rows.clone().filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
    /// # let a = a.assemble()?;
    /// # let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |_| 1.0)?;
    /// # let mut x = ParVector::zeros(&ctx, &comm, rows)?;
    /// let config = PCGSolverConfigBuilder::default().tol(1e-12).max_iters(2).build()?;
    /// let solver = ctx.pcg_solver(&comm, config)?;
    /// match solver.solve(&a, &b, &mut x) {
    ///     Ok(status) => println!("{status}"),
    ///     // `x` holds the last iterate.
    ///     Err(HypreError::NotConverged(status)) => assert_eq!(status.num_iters, 2),
    ///     Err(error) => return Err(error.into()),
    /// }
    /// # drop((solver, a, b, x));
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    #[error("{0}")]
    NotConverged(crate::solvers::IterativeSolverStatus),
    #[error("MPI Error")]
    MpiError,
    #[error("Unknown error")]
    UnknowError,
    #[error("Invalid parameter, should be positive")]
    InvalidParameterPositive,
    /// hypre returned an option value that the matching enum does not
    /// list (for example a newer hypre option).
    #[error("Unknown option value {0} returned by hypre")]
    UnknownOptionValue(hypre_sys::HYPRE_Int),
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
    /// A local builder was given an entry for a row that another MPI rank
    /// owns; use a distributed builder for those.
    #[error("Row {0} is not owned by this MPI rank: local builders only take local rows")]
    NotLocalRow(hypre_sys::HYPRE_BigInt),
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
