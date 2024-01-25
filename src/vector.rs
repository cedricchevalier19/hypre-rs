use crate::error::HypreError;
use crate::vector::Vector::IJ;
use crate::HypreResult;
use hypre_sys::*;
use itertools::Itertools;
use std::ffi::c_void;
use std::ptr::null_mut;

#[derive(Debug)]
pub struct IJVector {
    internal: HYPRE_IJVector,
}

impl IJVector {
    /// Creates an IJVector from a communicator [comm] and sizes
    pub fn new(comm: &impl mpi::topology::Communicator, rows: (u32, u32)) -> HypreResult<Self> {
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

            // unwrap because HYPRE_PARCSR is i32 in C
            check_hypre!(HYPRE_IJVectorSetObjectType(
                *h_matrix,
                HYPRE_PARCSR.try_into().unwrap()
            ));
            check_hypre!(HYPRE_IJVectorInitialize(*h_matrix));
        }
        Ok(out)
    }

    fn get_internal(&self) -> HypreResult<HYPRE_Vector> {
        let mut out: HYPRE_Vector = null_mut();
        unsafe {
            check_hypre!(HYPRE_IJVectorAssemble(self.internal));
            let csr_ptr = &mut out as *mut _ as *mut *mut c_void;
            check_hypre!(HYPRE_IJVectorGetObject(self.internal, csr_ptr));
        }
        Ok(out)
    }

    // Helper function for preparing non-zero elements
    fn get_elements<'a, Id, V>(
        &mut self,
        nnz: impl Iterator<Item = (Id, V)>,
    ) -> (Vec<HYPRE_BigInt>, Vec<HYPRE_Complex>)
    where
        Id: Copy + TryInto<HYPRE_BigInt> + 'a,
        V: Copy + TryInto<HYPRE_Complex> + 'a,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        nnz.into_iter()
            .map(|nnz| {
                (
                    nnz.0.try_into().unwrap_or_default(),
                    nnz.1.try_into().unwrap_or_default(),
                )
            })
            .multiunzip()
    }

    pub fn add_elements<'a, Id, V>(&mut self, nnz: impl Iterator<Item = (Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt> + 'a,
        V: Copy + TryInto<HYPRE_Complex> + 'a,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let (mut indices, mut values) = self.get_elements(nnz);

        unsafe {
            check_hypre!(HYPRE_IJVectorAddToValues(
                self.internal,
                indices.len().try_into()?,
                indices.as_mut_ptr(),
                values.as_mut_ptr(),
            ));
        }
        Ok(())
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
    pub(crate) fn get_internal(&self) -> HypreResult<HYPRE_Vector> {
        match self {
            IJ(m) => m.get_internal(),
        }
    }
}
