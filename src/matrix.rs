use crate::error::HypreError;
use crate::matrix::Matrix::{ParCSR, IJ};
use crate::utils::inclusive_bounds;
use crate::{Hypre, HypreResult};
use hypre_sys::{
    HYPRE_BigInt, HYPRE_Complex, HYPRE_IJMatrix, HYPRE_IJMatrixAddToValues, HYPRE_IJMatrixAssemble,
    HYPRE_IJMatrixCreate, HYPRE_IJMatrixDestroy, HYPRE_IJMatrixGetObject, HYPRE_IJMatrixInitialize,
    HYPRE_IJMatrixSetObjectType, HYPRE_Int, HYPRE_Matrix, HYPRE_ParCSRMatrix,
    HYPRE_ParCSRMatrixCreate, HYPRE_ParCSRMatrixDestroy, HYPRE_PARCSR,
};
use mpi::topology::Communicator;
use std::marker::PhantomData;
use std::num::TryFromIntError;
use std::ops::Range;
use std::{ffi::c_void, ptr::null_mut};

use itertools::Itertools;

/// Parallel CSR matrix, owning its hypre handle.
///
/// Not `Clone`: copying the handle would destroy it twice.
#[derive(Debug)]
pub struct CSRMatrix<'h> {
    internal_matrix: HYPRE_ParCSRMatrix,
    _hypre: PhantomData<&'h ()>,
}

impl<'h> CSRMatrix<'h> {
    fn translate_into_hypre_bigints(
        values: &[usize],
    ) -> Result<Vec<HYPRE_BigInt>, TryFromIntError> {
        values
            .iter()
            .map(|&x| x.try_into())
            .collect::<Result<Vec<HYPRE_BigInt>, TryFromIntError>>()
    }

    pub fn new(
        _hypre: &'h Hypre<'_>,
        comm: &'h impl Communicator,
        global_num_rows: usize,
        global_num_cols: usize,
        row_starts: &[usize],
        col_starts: &[usize],
    ) -> HypreResult<Self> {
        let mut out = CSRMatrix {
            internal_matrix: null_mut(),
            _hypre: PhantomData,
        };
        let mut h_row_starts = Self::translate_into_hypre_bigints(row_starts)?;
        let mut h_col_starts = Self::translate_into_hypre_bigints(col_starts)?;
        unsafe {
            check_hypre!(HYPRE_ParCSRMatrixCreate(
                comm.as_raw(),
                global_num_rows.try_into()?,
                global_num_cols.try_into()?,
                h_row_starts.as_mut_ptr(),
                h_col_starts.as_mut_ptr(),
                0,
                0,
                0,
                &mut out.internal_matrix,
            ));
        }
        Ok(out)
    }

    fn get_internal(&self) -> HypreResult<HYPRE_Matrix> {
        Ok(self.internal_matrix as HYPRE_Matrix)
    }
}

impl_hypre_drop!(CSRMatrix<'_>, internal_matrix, HYPRE_ParCSRMatrixDestroy);

#[derive(Debug)]
pub struct IJMatrix<'h> {
    internal_matrix: HYPRE_IJMatrix,
    _hypre: PhantomData<&'h ()>,
}

/// Converts from a [IJMatrix]
///
/// Calls hypre to assemble the matrix in CSR format.
/// Conversion can fail in hypre.
impl<'h> TryFrom<IJMatrix<'h>> for CSRMatrix<'h> {
    type Error = HypreError;

    fn try_from(ij: IJMatrix<'h>) -> Result<Self, Self::Error> {
        let mut parcsr: *mut c_void = null_mut();
        unsafe {
            check_hypre!(HYPRE_IJMatrixAssemble(ij.internal_matrix));
            check_hypre!(HYPRE_IJMatrixGetObject(ij.internal_matrix, &mut parcsr));
            // `ij` is dropped on return, and HYPRE_IJMatrixDestroy would also
            // destroy the ParCSR object. With object type -1 it only frees the
            // IJ interface, so the returned CSRMatrix becomes the sole owner.
            check_hypre!(HYPRE_IJMatrixSetObjectType(ij.internal_matrix, -1));
        }
        Ok(CSRMatrix {
            internal_matrix: parcsr as HYPRE_ParCSRMatrix,
            _hypre: PhantomData,
        })
    }
}

impl<'h> IJMatrix<'h> {
    /// Creates an IJMatrix from a communicator [Communicator] and sizes
    ///
    /// # Arguments
    ///
    /// - comm: MPI communicotor
    /// - rows: global indices of the rows owned by the current MPI rank
    /// - cols: global indices of the cols owned by the current MPI rank
    ///
    /// # Remarks
    ///
    /// Ranges are half-open, as usual in Rust: `0..10` is 10 rows (hypre's
    /// inclusive `(0, 9)`). An empty range is allowed for an MPI rank that owns
    /// no rows. The ranges of all MPI ranks must not overlap and must cover
    /// the whole matrix.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if a range's start is greater than its end.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), hypre_rs::HypreError> {
    /// # extern crate hypre_rs;
    /// # use mpi::initialize;
    /// # use mpi::topology::Communicator;
    /// use hypre_rs::matrix::IJMatrix;
    /// # let universe = mpi::initialize().unwrap();
    /// # let hypre = hypre_rs::initialize(&universe)?;
    /// # let mpi_comm = universe.world();
    /// let global_size: u32 = 100;
    /// let (rank, size) = (mpi_comm.rank() as u32, mpi_comm.size() as u32);
    /// let begin = rank * global_size / size;
    /// let end = (rank + 1) * global_size / size;
    /// let ij_matrix = IJMatrix::new(&hypre, &mpi_comm, begin..end, begin..end)?;
    /// # drop(ij_matrix);
    /// # hypre.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        _hypre: &'h Hypre<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> HypreResult<Self> {
        let (row_lower, row_upper) = inclusive_bounds(&rows)?;
        let (col_lower, col_upper) = inclusive_bounds(&cols)?;
        let mut out = Self {
            internal_matrix: null_mut(),
            _hypre: PhantomData,
        };
        // If a call below fails, dropping `out` destroys whatever was created.
        unsafe {
            check_hypre!(HYPRE_IJMatrixCreate(
                comm.as_raw(),
                row_lower,
                row_upper,
                col_lower,
                col_upper,
                &mut out.internal_matrix,
            ));

            // unwrap is ok as HYPRE_PARCSR is i32 in C
            check_hypre!(HYPRE_IJMatrixSetObjectType(
                out.internal_matrix,
                HYPRE_PARCSR.try_into().unwrap()
            ));
            check_hypre!(HYPRE_IJMatrixInitialize(out.internal_matrix));
        }
        Ok(out)
    }

    /// Add elements to the matrix.
    ///
    /// Elements that do not already exist are created.
    /// This routine also to contribute to any element, even remote.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), hypre_rs::HypreError> {
    /// # extern crate hypre_rs;
    /// # use mpi::initialize;
    /// # use mpi::topology::Communicator;
    /// use hypre_rs::matrix::IJMatrix;
    /// # let universe = mpi::initialize().unwrap();
    /// # let hypre = hypre_rs::initialize(&universe)?;
    /// # let mpi_comm = universe.world();
    /// # let global_size: u32 = 100;
    /// # let (rank, size) = (mpi_comm.rank() as u32, mpi_comm.size() as u32);
    /// # let local_begin = rank * global_size / size;
    /// # let local_end = (rank + 1) * global_size / size;
    /// let mut ij_matrix = IJMatrix::new(&hypre, &mpi_comm, local_begin..local_end, local_begin..local_end)?;
    /// ij_matrix.add_elements::<u32, f64>((local_begin..local_end).map(|id| (id, id, 1.0f64)));
    /// # drop(ij_matrix);
    /// # hypre.finalize()
    /// # }
    /// ```
    pub fn add_elements<Id, V>(&mut self, nnz: impl Iterator<Item = (Id, Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let (mut rows, mut cols, mut values): (Vec<_>, Vec<_>, Vec<_>) = nnz
            .into_iter()
            .map(|nnz| {
                (
                    nnz.0.try_into().unwrap_or_default(),
                    nnz.1.try_into().unwrap_or_default(),
                    nnz.2.try_into().unwrap_or_default(),
                )
            })
            .multiunzip();
        let mut ncols = vec![1 as HYPRE_Int; rows.len()];

        unsafe {
            check_hypre!(HYPRE_IJMatrixAddToValues(
                self.internal_matrix,
                rows.len().try_into()?,
                ncols.as_mut_ptr(),
                cols.as_mut_ptr(),
                rows.as_mut_ptr(),
                values.as_mut_ptr()
            ));
        }
        Ok(())
    }

    fn get_internal(&mut self) -> HypreResult<HYPRE_Matrix> {
        let mut out: HYPRE_Matrix = null_mut();
        unsafe {
            check_hypre!(HYPRE_IJMatrixAssemble(self.internal_matrix));
            let csr_mat_ptr = &mut out as *mut _ as *mut *mut c_void;
            check_hypre!(HYPRE_IJMatrixGetObject(self.internal_matrix, csr_mat_ptr));
        }
        Ok(out)
    }
}

impl_hypre_drop!(IJMatrix<'_>, internal_matrix, HYPRE_IJMatrixDestroy);

/// Matrix datatype
///
/// This type is generic, to handle the different hypre implementations
#[derive(Debug)]
pub enum Matrix<'h> {
    /// IJ matrix, defined algebraically
    IJ(IJMatrix<'h>),
    /// Parallel CSR matrix, behind IJ matrix
    ParCSR(CSRMatrix<'h>),
}

impl Matrix<'_> {
    pub(crate) fn get_internal(&mut self) -> HypreResult<HYPRE_Matrix> {
        match self {
            IJ(m) => m.get_internal(),
            ParCSR(m) => m.get_internal(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dropping a wrapper whose `Create` never succeeded must leave hypre's
    /// global error state clean (`HYPRE_IJMatrixDestroy(NULL)` sets the flag).
    #[test]
    fn drop_null_handle() {
        drop(IJMatrix {
            internal_matrix: null_mut(),
            _hypre: PhantomData,
        });
        drop(CSRMatrix {
            internal_matrix: null_mut(),
            _hypre: PhantomData,
        });
        assert_eq!(unsafe { hypre_sys::HYPRE_GetError() }, 0);
    }
}
