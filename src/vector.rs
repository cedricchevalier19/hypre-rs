use crate::error::HypreError;
use crate::vector::Vector::IJ;
use crate::HypreResult;
use hypre_sys::*;
use mpi::topology::Communicator;
use std::num::TryFromIntError;
use std::{ffi::c_void, ptr::null_mut};

#[derive(Debug)]
pub struct IJVector {
    internal: HYPRE_IJVector,
}

impl IJVector {
    /// Creates an IJVector from a communicator [comm] and sizes
    pub fn new(comm: impl mpi::topology::Communicator, rows: (usize, usize)) -> HypreResult<Self> {
        let mut out = Self {
            internal: null_mut(),
        };
        unsafe {
            let h_matrix = &mut out.internal as *mut _ as *mut HYPRE_IJVector;
            check_hypre!(HYPRE_IJVectorCreate(
                comm.as_raw(),
                rows.0.try_into()?,
                rows.1.try_into()?,
                h_matrix,
            ));

            check_hypre!(HYPRE_IJVectorSetObjectType(*h_matrix, 0));
            check_hypre!(HYPRE_IJVectorInitialize(*h_matrix));
        }
        Ok(out)
    }

    fn get_internal(self) -> HypreResult<HYPRE_Vector> {
        Ok(self.internal as HYPRE_Vector)
    }
}

impl Drop for IJVector {
    fn drop(&mut self) {
        unsafe {
            HYPRE_IJVectorDestroy(self.internal);
        }
    }
}

/// Vector datatype
///
/// This type is generic, to handle the different hypre implementations
#[derive(Debug)]
pub enum Vector {
    /// IJ matrix, defined algebraically
    IJ(IJVector),
}

impl Vector {
    pub(crate) fn get_internal(self) -> HypreResult<HYPRE_Vector> {
        match self {
            IJ(m) => m.get_internal(),
        }
    }
}