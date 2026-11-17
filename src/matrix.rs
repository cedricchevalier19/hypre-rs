use crate::error::HypreError;
use crate::matrix::Matrix::{ParCSR, IJ};
use crate::HypreResult;
use hypre_sys::{
    HYPRE_BigInt, HYPRE_Complex, HYPRE_IJMatrix, HYPRE_IJMatrixAddToValues, HYPRE_IJMatrixAssemble,
    HYPRE_IJMatrixCreate, HYPRE_IJMatrixDestroy, HYPRE_IJMatrixGetObject, HYPRE_IJMatrixInitialize,
    HYPRE_IJMatrixSetObjectType, HYPRE_Int, HYPRE_Matrix, HYPRE_ParCSRMatrix,
    HYPRE_ParCSRMatrixCreate, HYPRE_ParCSRMatrixDestroy, HYPRE_PARCSR,
};
use mpi::topology::Communicator;
use std::num::TryFromIntError;
use std::{ffi::c_void, ptr::null_mut};

use itertools::Itertools;

#[derive(Debug, Clone)]
pub struct CSRMatrix {
    internal_matrix: HYPRE_ParCSRMatrix,
}

impl CSRMatrix {
    fn translate_into_hypre_bigints(
        values: &[usize],
    ) -> Result<Vec<HYPRE_BigInt>, TryFromIntError> {
        values
            .iter()
            .map(|&x| x.try_into())
            .collect::<Result<Vec<HYPRE_BigInt>, TryFromIntError>>()
    }

    pub fn new(
        comm: &impl Communicator,
        global_num_rows: usize,
        global_num_cols: usize,
        row_starts: &[usize],
        col_starts: &[usize],
    ) -> HypreResult<Self> {
        let mut out = CSRMatrix {
            internal_matrix: null_mut(),
        };
        let mut h_row_starts = Self::translate_into_hypre_bigints(row_starts)?;
        let mut h_col_starts = Self::translate_into_hypre_bigints(col_starts)?;
        unsafe {
            match HYPRE_ParCSRMatrixCreate(
                comm.as_raw(),
                global_num_rows.try_into()?,
                global_num_cols.try_into()?,
                h_row_starts.as_mut_ptr(),
                h_col_starts.as_mut_ptr(),
                0,
                0,
                0,
                &mut out.internal_matrix,
            ) {
                0 => Ok(out),
                x => Err(HypreError::new(x)),
            }
        }
    }

    fn get_internal(&self) -> HypreResult<HYPRE_Matrix> {
        Ok(self.internal_matrix as HYPRE_Matrix)
    }
}

impl Drop for CSRMatrix {
    fn drop(&mut self) {
        unsafe {
            HYPRE_ParCSRMatrixDestroy(self.internal_matrix);
        }
    }
}

#[derive(Debug)]
pub struct IJMatrix {
    internal_matrix: HYPRE_IJMatrix,
}

/// Converts from a [IJMatrix]
///
/// Calls hypre to assemble the matrix in CSR format.
/// Conversion can fail in hypre.
impl TryFrom<IJMatrix> for CSRMatrix {
    type Error = HypreError;

    fn try_from(ij: IJMatrix) -> Result<Self, Self::Error> {
        let mut out = CSRMatrix {
            internal_matrix: null_mut(),
        };
        unsafe {
            check_hypre!(HYPRE_IJMatrixAssemble(ij.internal_matrix));
            let csr_mat_ptr = &mut out.internal_matrix as *mut _ as *mut *mut c_void;
            check_hypre!(HYPRE_IJMatrixGetObject(ij.internal_matrix, csr_mat_ptr));
        }
        Ok(out)
    }
}

impl IJMatrix {
    /// Creates an IJMatrix from a communicator [Communicator] and sizes
    ///
    /// # Arguments
    ///
    /// - comm: MPI communicotor
    /// - rows: `(first, last)` global indices of the rows owned by the current process
    /// - cols: `(first, last)` global indices of the cols owned by the current process
    ///
    /// # Remarks
    ///
    /// Both bounds are **inclusive**, as in hypre: `(0, 9)` is 10 rows. The
    /// ranges of all processes must not overlap and must cover the whole matrix.
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
    /// # hypre_rs::initialize()?;
    /// # let mpi_comm = universe.world();
    /// let global_size: u32 = 100;
    /// let (rank, size) = (mpi_comm.rank() as u32, mpi_comm.size() as u32);
    /// let begin = rank * global_size / size;
    /// let end = (rank + 1) * global_size / size; // exclusive
    /// let ij_matrix = IJMatrix::new(&mpi_comm, (begin, end - 1), (begin, end - 1))?;
    /// # hypre_rs::finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        comm: &impl mpi::topology::Communicator,
        rows: (u32, u32),
        cols: (u32, u32),
    ) -> HypreResult<Self> {
        let mut out = Self {
            internal_matrix: null_mut(),
        };
        unsafe {
            let h_matrix = &mut out.internal_matrix as *mut _ as *mut HYPRE_IJMatrix;
            check_hypre!(HYPRE_IJMatrixCreate(
                comm.as_raw(),
                rows.0.try_into()?,
                rows.1.try_into()?,
                cols.0.try_into()?,
                cols.1.try_into()?,
                h_matrix,
            ));

            // unwrap is ok as HYPRE_PARCSR is i32 in C
            check_hypre!(HYPRE_IJMatrixSetObjectType(
                *h_matrix,
                HYPRE_PARCSR.try_into().unwrap()
            ));
            check_hypre!(HYPRE_IJMatrixInitialize(*h_matrix));
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
    /// # hypre_rs::initialize()?;
    /// # let mpi_comm = universe.world();
    /// # let global_size: u32 = 100;
    /// # let (rank, size) = (mpi_comm.rank() as u32, mpi_comm.size() as u32);
    /// # let local_begin = rank * global_size / size;
    /// # let local_end = (rank + 1) * global_size / size; // exclusive
    /// let mut ij_matrix = IJMatrix::new(&mpi_comm, (local_begin, local_end - 1), (local_begin, local_end - 1))?;
    /// ij_matrix.add_elements::<u32, f64>((local_begin..local_end).map(|id| (id, id, 1.0f64)));
    /// # hypre_rs::finalize()
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

impl Drop for IJMatrix {
    fn drop(&mut self) {
        unsafe {
            HYPRE_IJMatrixDestroy(self.internal_matrix);
        }
    }
}

/// Matrix datatype
///
/// This type is generic, to handle the different hypre implementations
#[derive(Debug)]
pub enum Matrix {
    /// IJ matrix, defined algebraically
    IJ(IJMatrix),
    /// Parallel CSR matrix, behind IJ matrix
    ParCSR(CSRMatrix),
}

impl Matrix {
    pub(crate) fn get_internal(&mut self) -> HypreResult<HYPRE_Matrix> {
        match self {
            IJ(m) => m.get_internal(),
            ParCSR(m) => m.get_internal(),
        }
    }
}
