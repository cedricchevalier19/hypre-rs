use crate::error::HypreError;
use crate::matrix::Matrix::{ParCSR, IJ};
use crate::HypreResult;
use hypre_sys::{
    HYPRE_BigInt, HYPRE_Complex, HYPRE_IJMatrix, HYPRE_IJMatrixAddToValues, HYPRE_IJMatrixAssemble,
    HYPRE_IJMatrixCreate, HYPRE_IJMatrixDestroy, HYPRE_IJMatrixGetObject, HYPRE_IJMatrixInitialize,
    HYPRE_IJMatrixSetObjectType, HYPRE_Int, HYPRE_Matrix, HYPRE_ParCSRMatrix,
    HYPRE_ParCSRMatrixCreate, HYPRE_ParCSRMatrixDestroy,
};
use mpi::topology::Communicator;
use std::num::TryFromIntError;
use std::{ffi::c_void, ptr::null_mut};

use itertools::izip;
use itertools::Itertools;

#[derive(Debug, Clone)]
pub struct CSRMatrix {
    internal_matrix: HYPRE_ParCSRMatrix,
}

#[derive(Debug, Clone, Copy)]
pub struct NNZ<Id, V>
where
    Id: Copy,
    V: Copy,
{
    pub row_id: Id,
    pub col_id: Id,
    pub value: V,
}

impl CSRMatrix {
    fn new(
        comm: impl Communicator,
        global_num_rows: usize,
        global_num_cols: usize,
        row_starts: &[usize],
        col_starts: &[usize],
    ) -> HypreResult<Self> {
        let mut out = CSRMatrix {
            internal_matrix: null_mut(),
        };
        let mut h_row_starts = row_starts
            .iter()
            .map(|&x| x.try_into())
            .collect::<Result<Vec<HYPRE_BigInt>, TryFromIntError>>()?;
        let mut h_col_starts: Vec<HYPRE_BigInt> = col_starts
            .iter()
            .map(|&x| x.try_into())
            .collect::<Result<Vec<HYPRE_BigInt>, TryFromIntError>>()?;
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

    fn get_internal(self) -> HypreResult<HYPRE_Matrix> {
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
    /// - rows: couple of global indices of rows owned by current processor
    /// - cols: couple of global indices of cols owned by current processor
    ///
    /// # Remarks
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
    /// # let mpi_comm = universe.world();
    /// let global_size: usize = 100;
    /// // Cannot panic as global_size is properly represented on usize
    /// let step: usize = (global_size as i64 / mpi_comm.size() as i64).try_into().unwrap();
    /// let begin: usize = mpi_comm.rank() as usize * step;
    /// let end = (begin + step).clamp(0usize, global_size);
    /// let ij_matrix = IJMatrix::new(&mpi_comm, (begin, end), (begin, end))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        comm: &impl mpi::topology::Communicator,
        rows: (usize, usize),
        cols: (usize, usize),
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

            check_hypre!(HYPRE_IJMatrixSetObjectType(*h_matrix, 0));
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
    /// use hypre_rs::matrix::{IJMatrix, NNZ};
    /// # let universe = mpi::initialize().unwrap();
    /// # let mpi_comm = universe.world();
    /// # let global_size: usize = 100;
    /// // Cannot panic as global_size is properly represented on usize
    /// # let step: usize = (global_size as i64 / mpi_comm.size() as i64).try_into().unwrap();
    /// # let local_begin: usize = mpi_comm.rank() as usize * step;
    /// # let local_end = (local_begin + step).clamp(0usize, global_size);
    /// let ij_matrix = IJMatrix::new(&mpi_comm, (local_begin, local_end), (local_begin, local_end))?;
    /// let mut nnz = Vec::<NNZ::<i32, f64>>::with_capacity(local_end - local_begin);
    /// for id in local_begin..local_end {
    ///     nnz.push(NNZ::<i32, f64>{row_id: id as i32, col_id: id as i32, value: 1.0});
    /// }
    /// ij_matrix.add_elements::<i32, f64>(nnz)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_elements<Id, V>(self, nnz: Vec<NNZ<Id, V>>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_Int>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_Int>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let mut rows = vec![0 as HYPRE_BigInt; nnz.len()];
        let mut cols = vec![0 as HYPRE_BigInt; nnz.len()];
        let mut ncols = vec![1 as HYPRE_Int; nnz.len()];
        let mut values = vec![0 as HYPRE_Complex; nnz.len()];

        izip!(&mut rows, &mut cols, &mut values, nnz).try_for_each(|(row, col, val, nz)| {
            *row = nz.row_id.try_into()?;
            *col = nz.col_id.try_into()?;
            *val = nz.value.try_into()?;
            Ok::<(), HypreError>(())
        })?;

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
    pub(crate) fn get_internal(self) -> HypreResult<HYPRE_Matrix> {
        match self {
            IJ(m) => <IJMatrix as TryInto<CSRMatrix>>::try_into(m)?.get_internal(),
            ParCSR(m) => m.get_internal(),
        }
    }
}
