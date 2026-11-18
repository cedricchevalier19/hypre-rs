use crate::error::HypreError;
use crate::matrix::Matrix::{ParCSR, IJ};
use crate::utils::inclusive_bounds;
use crate::{HypreContext, HypreResult};
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
        _hypre: &'h HypreContext<'_>,
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
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # extern crate hypre_rs;
    /// # use mpi::initialize;
    /// # use mpi::topology::Communicator;
    /// use hypre_rs::matrix::IJMatrix;
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// # let mpi_comm = universe.world();
    /// let global_size: u32 = 100;
    /// let (rank, size) = (mpi_comm.rank() as u32, mpi_comm.size() as u32);
    /// let begin = rank * global_size / size;
    /// let end = (rank + 1) * global_size / size;
    /// let ij_matrix = IJMatrix::new(&ctx, &mpi_comm, begin..end, begin..end)?;
    /// # drop(ij_matrix);
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        _hypre: &'h HypreContext<'_>,
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
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # extern crate hypre_rs;
    /// # use mpi::initialize;
    /// # use mpi::topology::Communicator;
    /// use hypre_rs::matrix::IJMatrix;
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// # let mpi_comm = universe.world();
    /// # let global_size: u32 = 100;
    /// # let (rank, size) = (mpi_comm.rank() as u32, mpi_comm.size() as u32);
    /// # let local_begin = rank * global_size / size;
    /// # let local_end = (rank + 1) * global_size / size;
    /// let mut ij_matrix = IJMatrix::new(&ctx, &mpi_comm, local_begin..local_end, local_begin..local_end)?;
    /// ij_matrix.add_elements::<u32, f64>((local_begin..local_end).map(|id| (id, id, 1.0f64)))?;
    /// # drop(ij_matrix);
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_elements<Id, V>(&mut self, nnz: impl Iterator<Item = (Id, Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let (rows, cols, values): (Vec<_>, Vec<_>, Vec<_>) =
            convert_entries(nnz)?.into_iter().multiunzip();
        self.add_to_values(&rows, &cols, &values)
    }

    /// Adds `values[k]` to entry `(rows[k], cols[k])` for every `k`.
    fn add_to_values(
        &mut self,
        rows: &[HYPRE_BigInt],
        cols: &[HYPRE_BigInt],
        values: &[HYPRE_Complex],
    ) -> HypreResult<()> {
        debug_assert!(rows.len() == cols.len() && rows.len() == values.len());
        // One column per "row" entry: hypre reads entry k as
        // (rows[k], cols[k], values[k]).
        let mut ncols = vec![1 as HYPRE_Int; rows.len()];
        unsafe {
            check_hypre!(HYPRE_IJMatrixAddToValues(
                self.internal_matrix,
                rows.len().try_into()?,
                ncols.as_mut_ptr(),
                rows.as_ptr(),
                cols.as_ptr(),
                values.as_ptr()
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

/// Converts `(row, col, value)` entries to hypre's types.
///
/// # Errors
///
/// The conversion error of the first entry that does not fit.
fn convert_entries<Id, V>(
    nnz: impl Iterator<Item = (Id, Id, V)>,
) -> HypreResult<Vec<(HYPRE_BigInt, HYPRE_BigInt, HYPRE_Complex)>>
where
    Id: Copy + TryInto<HYPRE_BigInt>,
    V: Copy + TryInto<HYPRE_Complex>,
    HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
    HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
{
    nnz.map(|(row, col, value)| Ok((row.try_into()?, col.try_into()?, value.try_into()?)))
        .collect()
}

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

/// Square matrix that is symmetric by construction.
///
/// hypre always stores both triangles, so entries are given for the lower
/// triangle only (`col <= row`) and each off-diagonal entry `(i, j, v)` is also
/// added at `(j, i)`. Solvers that need a symmetric matrix, such as
/// [PCGSolver](crate::solvers::PCGSolver), only accept this type.
///
/// # Example
///
/// The 1D Laplacian `tridiag(-1, 2, -1)`:
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use mpi::topology::Communicator;
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
/// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
/// let mut laplacian = ctx.symmetric_matrix(&comm, rows.clone())?;
/// // Diagonal and sub-diagonal; the super-diagonal is added by symmetry.
/// laplacian.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
/// laplacian.add_elements(rows.filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
/// # drop(laplacian);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SymmetricMatrix<'h> {
    /// Symmetric: only filled through [SymmetricMatrix::add_elements].
    matrix: Matrix<'h>,
}

impl<'h> SymmetricMatrix<'h> {
    /// Creates an empty symmetric matrix, stored as an [IJMatrix].
    ///
    /// `rows` is the half-open range of global rows owned by the current MPI
    /// rank, as for [IJMatrix::new]; the matrix is square, so the same range
    /// is used for the columns.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if `rows.start > rows.end`, or the error
    /// returned by hypre.
    pub fn new(
        hypre: &'h HypreContext<'_>,
        comm: &'h impl Communicator,
        rows: Range<u32>,
    ) -> HypreResult<Self> {
        Ok(Self {
            matrix: IJ(IJMatrix::new(hypre, comm, rows.clone(), rows)?),
        })
    }

    /// Adds `(row, col, value)` entries of the lower triangle (`col <= row`).
    ///
    /// Each off-diagonal entry is also added at `(col, row)`, so an entry of
    /// the upper triangle must not be given as well: it would be counted
    /// twice. As for [IJMatrix::add_elements], values are added to existing
    /// ones and entries may belong to other MPI ranks; the mirrored entry of
    /// a local row often does.
    ///
    /// # Errors
    ///
    /// - [HypreError::NotLowerTriangular] if an entry has `col > row`,
    /// - [HypreError::NotIJMatrix] after [SymmetricMatrix::into_csr],
    /// - [HypreError::ConversionError] if an index or value does not fit
    ///   hypre's types,
    ///
    /// in which cases nothing is added, or the error returned by hypre.
    pub fn add_elements<Id, V>(&mut self, nnz: impl Iterator<Item = (Id, Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let IJ(ij) = &mut self.matrix else {
            return Err(HypreError::NotIJMatrix);
        };
        let entries = convert_entries(nnz)?;
        if entries.iter().any(|&(row, col, _)| col > row) {
            return Err(HypreError::NotLowerTriangular);
        }
        let (rows, cols, values): (Vec<_>, Vec<_>, Vec<_>) = entries
            .iter()
            .copied()
            .chain(
                entries
                    .iter()
                    .filter(|&&(row, col, _)| row != col)
                    .map(|&(row, col, value)| (col, row, value)),
            )
            .multiunzip();
        ij.add_to_values(&rows, &cols, &values)
    }

    /// Converts the matrix to the [CSRMatrix] format; see
    /// `TryFrom<IJMatrix> for CSRMatrix`.
    ///
    /// Assembles the matrix, so it is collective. The matrix cannot be
    /// modified afterwards. Converting a matrix that is already in CSR format
    /// does nothing.
    pub fn into_csr(self) -> HypreResult<Self> {
        let matrix = match self.matrix {
            IJ(ij) => ParCSR(CSRMatrix::try_from(ij)?),
            csr @ ParCSR(_) => csr,
        };
        Ok(Self { matrix })
    }

    pub(crate) fn get_internal(&mut self) -> HypreResult<HYPRE_Matrix> {
        self.matrix.get_internal()
    }
}

/// Forgets that the matrix is symmetric.
impl<'h> From<SymmetricMatrix<'h>> for Matrix<'h> {
    fn from(symmetric: SymmetricMatrix<'h>) -> Self {
        symmetric.matrix
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
