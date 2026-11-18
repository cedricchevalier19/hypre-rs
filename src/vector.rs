use crate::error::HypreError;
use crate::utils::inclusive_bounds;
use crate::vector::Vector::IJ;
use crate::{Hypre, HypreResult};
use hypre_sys::*;
use itertools::Itertools;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::null_mut;

#[derive(Debug)]
pub struct IJVector<'h> {
    internal: HYPRE_IJVector,
    _hypre: PhantomData<&'h ()>,
}

impl<'h> IJVector<'h> {
    /// Creates an IJVector from a communicator [comm] and sizes
    ///
    /// `rows` is the half-open range of global indices owned by the current MPI
    /// rank (`0..10` is 10 rows); it may be empty.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if `rows.start > rows.end`.
    pub fn new(
        _hypre: &'h Hypre<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
    ) -> HypreResult<Self> {
        let (lower, upper) = inclusive_bounds(&rows)?;
        let mut out = Self {
            internal: null_mut(),
            _hypre: PhantomData,
        };
        // If a call below fails, dropping `out` destroys whatever was created.
        unsafe {
            check_hypre!(HYPRE_IJVectorCreate(
                comm.as_raw(),
                lower,
                upper,
                &mut out.internal
            ));

            // unwrap because HYPRE_PARCSR is i32 in C
            check_hypre!(HYPRE_IJVectorSetObjectType(
                out.internal,
                HYPRE_PARCSR.try_into().unwrap()
            ));
            check_hypre!(HYPRE_IJVectorInitialize(out.internal));
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

impl_hypre_drop!(IJVector<'_>, internal, HYPRE_IJVectorDestroy);

/// Vector datatype
///
/// This type is generic, to handle the different hypre implementations
#[derive(Debug)]
pub enum Vector<'h> {
    /// IJ vector, defined algebraically
    IJ(IJVector<'h>),
}

impl Vector<'_> {
    pub(crate) fn get_internal(&self) -> HypreResult<HYPRE_Vector> {
        match self {
            IJ(m) => m.get_internal(),
        }
    }
}
