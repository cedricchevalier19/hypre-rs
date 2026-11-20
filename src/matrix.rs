use crate::error::HypreError;
use crate::utils::inclusive_bounds;
use crate::{HypreContext, HypreResult};
use hypre_sys::{
    HYPRE_BigInt, HYPRE_Complex, HYPRE_IJMatrix, HYPRE_IJMatrixAddToValues, HYPRE_IJMatrixAssemble,
    HYPRE_IJMatrixCreate, HYPRE_IJMatrixDestroy, HYPRE_IJMatrixGetObject, HYPRE_IJMatrixInitialize,
    HYPRE_IJMatrixSetObjectType, HYPRE_Int, HYPRE_ParCSRMatrix, HYPRE_ParCSRMatrixCreate,
    HYPRE_ParCSRMatrixDestroy, HYPRE_ParCSRMatrixGetComm, HYPRE_ParCSRMatrixGetDims,
    HYPRE_ParCSRMatrixGetLocalRange, HYPRE_ParCSRMatrixGetRow, HYPRE_ParCSRMatrixRestoreRow,
    HYPRE_PARCSR,
};
use mpi::collective::{CommunicatorCollectives, SystemOperation};
use mpi::datatype::{Partition, PartitionMut};
use mpi::raw::FromRaw;
use mpi::topology::{Communicator, SimpleCommunicator};
use mpi::Count;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::num::TryFromIntError;
use std::ops::{Deref, Range};
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
}

impl sealed::AsParCSR for CSRMatrix<'_> {
    fn as_parcsr(&self) -> HypreResult<HYPRE_ParCSRMatrix> {
        Ok(self.internal_matrix)
    }
}

impl Matrix for CSRMatrix<'_> {}

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
}

impl<'h> IJMatrix<'h> {
    /// Assembles the matrix into a [CSRMatrix], the format hypre's solvers
    /// use; same as `CSRMatrix::try_from`.
    ///
    /// An `IJMatrix` is not a [Matrix]: it must be assembled before a solver
    /// can use it, and cannot be modified afterwards. Assembly exchanges the
    /// entries added for rows owned by other MPI ranks, so it is collective.
    ///
    /// ```compile_fail
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use hypre_rs::solvers::{LinearSolver, Solver};
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// let mut a = ctx.ij_matrix(&comm, 0..1, 0..1)?;
    /// a.add_elements([(0u32, 0u32, 1.0)].into_iter())?;
    /// let b = ctx.ij_vector(&comm, 0..1)?;
    /// let mut x = ctx.ij_vector(&comm, 0..1)?;
    /// let solver = Solver::GMRES(ctx.gmres_solver(&comm, Default::default())?);
    /// solver.solve(&a, &b, &mut x)?; // `a` is not assembled
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// The error returned by hypre.
    pub fn assemble(self) -> HypreResult<CSRMatrix<'h>> {
        CSRMatrix::try_from(self)
    }
}

impl_hypre_drop!(IJMatrix<'_>, internal_matrix, HYPRE_IJMatrixDestroy);

/// A `(row, col, value)` matrix entry, in hypre's types.
type Entry = (HYPRE_BigInt, HYPRE_BigInt, HYPRE_Complex);

/// Matrix entries indexed by `(row, col)`.
type EntryMap = HashMap<(HYPRE_BigInt, HYPRE_BigInt), HYPRE_Complex>;

/// Converts `(row, col, value)` entries to hypre's types.
///
/// # Errors
///
/// The conversion error of the first entry that does not fit.
fn convert_entries<Id, V>(nnz: impl Iterator<Item = (Id, Id, V)>) -> HypreResult<Vec<Entry>>
where
    Id: Copy + TryInto<HYPRE_BigInt>,
    V: Copy + TryInto<HYPRE_Complex>,
    HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
    HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
{
    nnz.map(|(row, col, value)| Ok((row.try_into()?, col.try_into()?, value.try_into()?)))
        .collect()
}

pub(crate) mod sealed {
    use crate::HypreResult;
    use hypre_sys::HYPRE_ParCSRMatrix;

    /// Gives hypre's solvers the ParCSR object of an assembled matrix. Not
    /// nameable outside the crate, so only the crate's types implement
    /// [Matrix].
    ///
    /// [Matrix]: super::Matrix
    pub trait AsParCSR {
        /// Returns the ParCSR object.
        fn as_parcsr(&self) -> HypreResult<HYPRE_ParCSRMatrix>;
    }
}

/// An assembled distributed matrix, which hypre's solvers accept.
///
/// Implemented by [CSRMatrix] and by a [SymmetricMatrix] of one. An
/// [IJMatrix] is not a `Matrix`: [IJMatrix::assemble] it first. The trait is
/// sealed: other types cannot implement it. It is object safe, so
/// `&dyn Matrix` works when the storage is chosen at runtime.
pub trait Matrix: sealed::AsParCSR {
    /// Returns whether the matrix is symmetric: square, with
    /// `|a_ij - a_ji| <= tol * max(|a_ij|, |a_ji|)` for all `i`, `j`.
    ///
    /// `tol = 0.0` asks for exact symmetry. A missing entry counts as 0.
    ///
    /// Exchanges every off-diagonal entry with the MPI rank that owns its
    /// transposed entry, so it is collective and costs about as much as a
    /// transposition.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidParameterPositive] if `tol` is negative or NaN,
    /// or the error returned by hypre.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use mpi::topology::Communicator;
    /// use hypre_rs::Matrix;
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
    /// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
    /// let mut ij = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
    /// // Upper bidiagonal: not symmetric.
    /// ij.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
    /// ij.add_elements(rows.filter(|&i| i < 9).map(|i| (i, i + 1, -1.0)))?;
    /// let matrix = ij.assemble()?;
    /// assert!(!matrix.is_symmetric(0.0)?);
    /// # drop(matrix);
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    fn is_symmetric(&self, tol: f64) -> HypreResult<bool> {
        check_tolerance(tol)?;
        let parcsr = self.as_parcsr()?;

        let (mut num_rows, mut num_cols) = (0, 0);
        let (mut row_start, mut row_end, mut col_start, mut col_end) = (0, 0, 0, 0);
        let mut raw_comm = unsafe { mpi::ffi::RSMPI_COMM_NULL };
        unsafe {
            check_hypre!(HYPRE_ParCSRMatrixGetDims(
                parcsr,
                &mut num_rows,
                &mut num_cols
            ));
            check_hypre!(HYPRE_ParCSRMatrixGetLocalRange(
                parcsr,
                &mut row_start,
                &mut row_end,
                &mut col_start,
                &mut col_end,
            ));
            check_hypre!(HYPRE_ParCSRMatrixGetComm(parcsr, &mut raw_comm));
        }
        // The dimensions are global: every rank returns here.
        if num_rows != num_cols {
            return Ok(false);
        }

        // Local entries, read row by row (row_end is inclusive).
        let mut local = EntryMap::new();
        for row in row_start..=row_end {
            let mut size: HYPRE_Int = 0;
            let mut cols: *mut HYPRE_BigInt = null_mut();
            let mut values: *mut HYPRE_Complex = null_mut();
            unsafe {
                check_hypre!(HYPRE_ParCSRMatrixGetRow(
                    parcsr,
                    row,
                    &mut size,
                    &mut cols,
                    &mut values
                ));
                let len = usize::try_from(size)?;
                if len > 0 {
                    let cols = std::slice::from_raw_parts(cols, len);
                    let values = std::slice::from_raw_parts(values, len);
                    local.extend(cols.iter().zip(values).map(|(&c, &v)| ((row, c), v)));
                }
                check_hypre!(HYPRE_ParCSRMatrixRestoreRow(
                    parcsr,
                    row,
                    &mut size,
                    &mut cols,
                    &mut values
                ));
            }
        }

        // A duplicate of hypre's communicator, freed when dropped. Duplicating
        // is collective, like the rest of the check.
        let comm = unsafe {
            let mut dup = mpi::ffi::RSMPI_COMM_NULL;
            if mpi::ffi::MPI_Comm_dup(raw_comm, &mut dup) != 0 {
                return Err(HypreError::MpiError);
            }
            SimpleCommunicator::from_raw(dup)
        };
        let num_ranks = usize::try_from(comm.size())?;

        let mut starts = vec![HYPRE_BigInt::default(); num_ranks];
        comm.all_gather_into(&row_start, &mut starts[..]);
        let by_rank = transposed_by_owner(&local, &starts);
        let send_counts: Vec<Count> = by_rank
            .iter()
            .map(|entries| Count::try_from(entries.len()))
            .collect::<Result<_, _>>()?;
        let mut recv_counts = vec![0 as Count; num_ranks];
        comm.all_to_all_into(&send_counts[..], &mut recv_counts[..]);

        let (send_indices, send_values): (Vec<_>, Vec<_>) = by_rank
            .into_iter()
            .flatten()
            .map(|(row, col, value)| ([row, col], value))
            .unzip();
        let send_indices = send_indices.concat();
        let num_received = recv_counts.iter().map(|&c| c as usize).sum::<usize>();
        let mut recv_indices = vec![HYPRE_BigInt::default(); 2 * num_received];
        let mut recv_values = vec![HYPRE_Complex::default(); num_received];
        {
            let pairs = |counts: &[Count]| counts.iter().map(|&c| 2 * c).collect::<Vec<_>>();
            let (send_pairs, recv_pairs) = (pairs(&send_counts), pairs(&recv_counts));
            comm.all_to_all_varcount_into(
                &Partition::new(
                    &send_indices[..],
                    &send_pairs[..],
                    displacements(&send_pairs),
                ),
                &mut PartitionMut::new(
                    &mut recv_indices[..],
                    &recv_pairs[..],
                    displacements(&recv_pairs),
                ),
            );
            comm.all_to_all_varcount_into(
                &Partition::new(
                    &send_values[..],
                    &send_counts[..],
                    displacements(&send_counts),
                ),
                &mut PartitionMut::new(
                    &mut recv_values[..],
                    &recv_counts[..],
                    displacements(&recv_counts),
                ),
            );
        }
        // The transpose of the local rows.
        let transposed: EntryMap = recv_indices
            .iter()
            .copied()
            .step_by(2)
            .zip(recv_indices.iter().copied().skip(1).step_by(2))
            .zip(recv_values)
            .collect();

        let locally_symmetric = entries_match(&local, &transposed, tol);

        let mut symmetric = false;
        comm.all_reduce_into(
            &locally_symmetric,
            &mut symmetric,
            SystemOperation::logical_and(),
        );
        Ok(symmetric)
    }
}

/// Square matrix that is symmetric, stored as an `M`.
///
/// hypre always stores both triangles. A `SymmetricMatrix<IJMatrix>`
/// (made by [SymmetricMatrix::new] or
/// [HypreContext::symmetric_matrix]) is symmetric by construction: entries
/// are given for the lower triangle only (`col <= row`) and each
/// off-diagonal entry `(i, j, v)` is also added at `(j, i)`.
/// [SymmetricMatrix::assemble] then turns it into a
/// `SymmetricMatrix<CSRMatrix>`. Assembled matrices become symmetric ones
/// after a runtime check ([SymmetricMatrix::try_from]), and any matrix on
/// the caller's word ([SymmetricMatrix::from_matrix_unchecked]).
///
/// Solvers that need a symmetric matrix, such as
/// [PCGSolver](crate::solvers::PCGSolver), only accept this type. Once
/// assembled, it is also a [Matrix], so every solver accepts it. It
/// dereferences to `M` (read only: changing `M` could break the
/// symmetry).
///
/// # Example
///
/// The 1D Laplacian `tridiag(-1, 2, -1)`:
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use mpi::topology::Communicator;
/// use hypre_rs::Matrix;
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
/// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
/// let mut laplacian = ctx.symmetric_matrix(&comm, rows.clone())?;
/// // Diagonal and sub-diagonal; the super-diagonal is added by symmetry.
/// laplacian.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
/// laplacian.add_elements(rows.filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
/// let laplacian = laplacian.assemble()?;
/// assert!(laplacian.is_symmetric(0.0)?);
/// # drop(laplacian);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SymmetricMatrix<M: ?Sized> {
    /// Symmetric: only filled through [SymmetricMatrix::add_elements], or
    /// checked (or vouched for) on conversion.
    matrix: M,
}

impl<'h> SymmetricMatrix<IJMatrix<'h>> {
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
            matrix: IJMatrix::new(hypre, comm, rows.clone(), rows)?,
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
        let (rows, cols, values): (Vec<_>, Vec<_>, Vec<_>) = mirror_lower(convert_entries(nnz)?)?
            .into_iter()
            .multiunzip();
        self.matrix.add_to_values(&rows, &cols, &values)
    }

    /// Assembles the matrix into a [CSRMatrix]; see [IJMatrix::assemble].
    ///
    /// Solvers need an assembled matrix. Assembly is collective. A
    /// `SymmetricMatrix<CSRMatrix>` has no `add_elements`: it cannot be
    /// modified afterwards.
    ///
    /// ```compile_fail
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// let mut matrix = ctx.symmetric_matrix(&comm, 0..1)?;
    /// matrix.add_elements([(0u32, 0u32, 1.0)].into_iter())?;
    /// let mut csr = matrix.assemble()?;
    /// csr.add_elements([(0u32, 0u32, 1.0)].into_iter())?; // no such method
    /// # Ok(())
    /// # }
    /// ```
    pub fn assemble(self) -> HypreResult<SymmetricMatrix<CSRMatrix<'h>>> {
        Ok(SymmetricMatrix {
            matrix: self.matrix.assemble()?,
        })
    }
}

impl<M: Matrix> SymmetricMatrix<M> {
    /// Converts `matrix` if it is symmetric up to `tol`; see
    /// [Matrix::is_symmetric], which is collective.
    ///
    /// The matrix keeps the entries it has: within the tolerance, `a_ij` and
    /// `a_ji` may still differ slightly.
    ///
    /// # Errors
    ///
    /// [HypreError::NotSymmetric] if it is not (the matrix is dropped; call
    /// [Matrix::is_symmetric] first to keep it), or the errors of
    /// [Matrix::is_symmetric].
    pub fn try_from_with_tolerance(matrix: M, tol: f64) -> HypreResult<Self> {
        if matrix.is_symmetric(tol)? {
            Ok(Self { matrix })
        } else {
            Err(HypreError::NotSymmetric)
        }
    }
}

impl<M> SymmetricMatrix<M> {
    /// Converts `matrix` without checking that it is symmetric.
    ///
    /// Use it when the matrix is known to be symmetric and the check of
    /// [SymmetricMatrix::try_from] (collective, about the cost of a
    /// transposition) is not worth it, or for an [IJMatrix] that is not
    /// assembled yet. It neither communicates nor assembles the matrix.
    ///
    /// This is not `unsafe`: an asymmetric matrix cannot cause undefined
    /// behavior. But solvers then work on a matrix that breaks their
    /// assumptions: CG, for example, may not converge, or converge to a wrong
    /// solution without reporting an error.
    ///
    /// [SymmetricMatrix::add_elements] keeps adding mirrored entries, so it
    /// keeps the matrix as symmetric as it was.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use mpi::topology::Communicator;
    /// use hypre_rs::SymmetricMatrix;
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
    /// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
    /// // Diagonal, so symmetric: no need to check it.
    /// let mut ij = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
    /// ij.add_elements(rows.map(|i| (i, i, 2.0)))?;
    /// let diagonal = SymmetricMatrix::from_matrix_unchecked(ij);
    /// # drop(diagonal);
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_matrix_unchecked(matrix: M) -> Self {
        Self { matrix }
    }

    /// Returns the matrix, forgetting that it is symmetric.
    pub fn into_inner(self) -> M {
        self.matrix
    }
}

/// Checks that the matrix is exactly symmetric, at runtime.
///
/// This is collective; see [Matrix::is_symmetric] and
/// [SymmetricMatrix::try_from_with_tolerance].
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use mpi::topology::Communicator;
/// use hypre_rs::SymmetricMatrix;
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
/// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
/// // tridiag(-1, 2, -1), with both triangles given.
/// let mut ij = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
/// for i in rows {
///     ij.add_elements([(i, i, 2.0)].into_iter())?;
///     if i > 0 {
///         ij.add_elements([(i, i - 1, -1.0), (i - 1, i, -1.0)].into_iter())?;
///     }
/// }
/// let laplacian = SymmetricMatrix::try_from(ij.assemble()?)?;
/// # drop(laplacian);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
// Not generic: `impl<M: Matrix> TryFrom<M>` conflicts with core's blanket
// `TryFrom<U> for T where U: Into<T>`. Add one impl per assembled matrix type.
impl<'h> TryFrom<CSRMatrix<'h>> for SymmetricMatrix<CSRMatrix<'h>> {
    type Error = HypreError;

    fn try_from(matrix: CSRMatrix<'h>) -> Result<Self, Self::Error> {
        Self::try_from_with_tolerance(matrix, 0.0)
    }
}

/// Read-only access to the stored matrix. There is no `DerefMut`: changing
/// the matrix (for example adding entries to its [IJMatrix]) could break the
/// symmetry. Use [SymmetricMatrix::add_elements] instead.
impl<M: ?Sized> Deref for SymmetricMatrix<M> {
    type Target = M;

    fn deref(&self) -> &M {
        &self.matrix
    }
}

impl<M: Matrix + ?Sized> sealed::AsParCSR for SymmetricMatrix<M> {
    fn as_parcsr(&self) -> HypreResult<HYPRE_ParCSRMatrix> {
        self.matrix.as_parcsr()
    }
}

/// A symmetric matrix is a [Matrix]: every solver accepts it.
impl<M: Matrix + ?Sized> Matrix for SymmetricMatrix<M> {}

/// Checks a relative tolerance: it must be a non-negative number.
fn check_tolerance(tol: f64) -> HypreResult<()> {
    if tol.is_nan() || tol < 0.0 {
        return Err(HypreError::InvalidParameterPositive);
    }
    Ok(())
}

/// Returns the lower-triangle `entries` followed by the transposed copy of
/// each off-diagonal one.
///
/// # Errors
///
/// [HypreError::NotLowerTriangular] if an entry has `col > row`.
fn mirror_lower(mut entries: Vec<Entry>) -> HypreResult<Vec<Entry>> {
    if entries.iter().any(|&(row, col, _)| col > row) {
        return Err(HypreError::NotLowerTriangular);
    }
    let mirrored: Vec<Entry> = entries
        .iter()
        .filter(|&&(row, col, _)| row != col)
        .map(|&(row, col, value)| (col, row, value))
        .collect();
    entries.extend(mirrored);
    Ok(entries)
}

/// Returns the MPI rank that owns `row`, given the first row of every rank
/// (in rank order, non-decreasing).
///
/// Row `j` belongs to the last rank whose first row is `<= j`: ranks that
/// own no rows share the next rank's first row. `row` must be at least
/// `starts[0]`.
fn row_owner(starts: &[HYPRE_BigInt], row: HYPRE_BigInt) -> usize {
    starts.partition_point(|&start| start <= row) - 1
}

/// Groups the transpose of `local` by owning rank: entry `a_ij` becomes
/// `(j, i, a_ij)`, sent to the owner of row `j`.
fn transposed_by_owner(local: &EntryMap, starts: &[HYPRE_BigInt]) -> Vec<Vec<Entry>> {
    let mut by_rank = vec![Vec::new(); starts.len()];
    for (&(row, col), &value) in local {
        by_rank[row_owner(starts, col)].push((col, row, value));
    }
    by_rank
}

/// Exclusive prefix sum of `counts`: where each rank's block starts in an
/// MPI buffer.
fn displacements(counts: &[Count]) -> Vec<Count> {
    counts
        .iter()
        .scan(0, |offset, &count| {
            let displ = *offset;
            *offset += count;
            Some(displ)
        })
        .collect()
}

/// Whether `|a - b| <= tol * max(|a|, |b|)` for the entries `a` of `local`
/// and `b` of `transposed` at every key of either, a missing entry counting
/// as 0.
fn entries_match(local: &EntryMap, transposed: &EntryMap, tol: f64) -> bool {
    local.keys().chain(transposed.keys()).all(|key| {
        let a = local.get(key).copied().unwrap_or_default();
        let b = transposed.get(key).copied().unwrap_or_default();
        (a - b).abs() <= tol * a.abs().max(b.abs())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dropping a wrapper whose `Create` never succeeded must leave hypre's
    /// global error state clean (`HYPRE_IJMatrixDestroy(NULL)` sets the flag).
    #[test]
    fn drop_null_handle() {
        let _lock = crate::utils::hypre_error_lock();
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

    #[test]
    fn convert_entries_to_hypre_types() -> HypreResult<()> {
        assert_eq!(
            convert_entries([(1u32, 2u32, 3.0f64), (4, 5, -6.0)].into_iter())?,
            vec![(1, 2, 3.0), (4, 5, -6.0)]
        );
        assert_eq!(
            convert_entries(std::iter::empty::<(u32, u32, f64)>())?,
            vec![]
        );
        // An index that does not fit in HYPRE_BigInt, in either position.
        for entry in [(u64::MAX, 0, 1.0), (0, u64::MAX, 1.0)] {
            assert!(matches!(
                convert_entries([(0u64, 0u64, 1.0f64), entry].into_iter()),
                Err(HypreError::ConversionError(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn translate_into_hypre_bigints() {
        assert_eq!(
            CSRMatrix::translate_into_hypre_bigints(&[0, 5, 10]).ok(),
            Some(vec![0, 5, 10])
        );
        assert!(CSRMatrix::translate_into_hypre_bigints(&[0, usize::MAX]).is_err());
    }

    #[test]
    fn mirror_lower_adds_transposed_off_diagonal_entries() -> HypreResult<()> {
        assert_eq!(mirror_lower(vec![])?, vec![]);
        // Diagonal entries are not duplicated.
        assert_eq!(mirror_lower(vec![(0, 0, 2.0)])?, vec![(0, 0, 2.0)]);
        assert_eq!(
            mirror_lower(vec![(1, 1, 2.0), (1, 0, -1.0), (5, 2, 3.0)])?,
            vec![
                (1, 1, 2.0),
                (1, 0, -1.0),
                (5, 2, 3.0),
                (0, 1, -1.0),
                (2, 5, 3.0)
            ]
        );
        assert!(matches!(
            mirror_lower(vec![(1, 0, -1.0), (0, 1, -1.0)]),
            Err(HypreError::NotLowerTriangular)
        ));
        Ok(())
    }

    #[test]
    fn row_owner_with_empty_ranks() {
        // Rows 0..3 on rank 0, none on rank 1, 3..7 on rank 2, 7.. on rank 3.
        let starts = [0, 3, 3, 7];
        let owners: Vec<usize> = (0..10).map(|row| row_owner(&starts, row)).collect();
        assert_eq!(owners, [0, 0, 0, 2, 2, 2, 2, 3, 3, 3]);
        // Trailing ranks without rows start at the global number of rows,
        // which no row reaches.
        let starts = [0, 4, 4];
        assert_eq!(row_owner(&starts, 3), 0);
        // A single rank owns everything.
        assert_eq!(row_owner(&[0], 1000), 0);
    }

    #[test]
    fn transposed_by_owner_groups_by_rank_of_column() {
        // Rank 0 owns rows 0..2, rank 1 owns rows 2...
        let starts = [0, 2];
        let local = EntryMap::from([((0, 0), 1.0), ((0, 3), 2.0), ((1, 2), 3.0)]);
        let mut by_rank = transposed_by_owner(&local, &starts);
        for entries in &mut by_rank {
            entries.sort_by_key(|&(row, col, _)| (row, col));
        }
        assert_eq!(
            by_rank,
            vec![vec![(0, 0, 1.0)], vec![(2, 1, 3.0), (3, 0, 2.0)]]
        );
    }

    #[test]
    fn displacements_are_exclusive_prefix_sums() {
        assert_eq!(displacements(&[]), Vec::<Count>::new());
        assert_eq!(displacements(&[4]), vec![0]);
        assert_eq!(displacements(&[2, 0, 3, 1]), vec![0, 2, 2, 5]);
    }

    #[test]
    fn entries_match_with_tolerance() {
        let local = EntryMap::from([((0, 1), 1.0), ((1, 1), 2.0)]);
        assert!(entries_match(&local, &local, 0.0));
        assert!(entries_match(&EntryMap::new(), &EntryMap::new(), 0.0));

        // Slightly different values: only accepted with a tolerance.
        let close = EntryMap::from([((0, 1), 1.0 + 1e-10), ((1, 1), 2.0)]);
        assert!(!entries_match(&local, &close, 0.0));
        assert!(entries_match(&local, &close, 1e-8));
        assert!(entries_match(&close, &local, 1e-8));

        // A missing entry counts as 0, on either side.
        let missing = EntryMap::from([((1, 1), 2.0)]);
        assert!(!entries_match(&local, &missing, 0.5));
        assert!(!entries_match(&missing, &local, 0.5));
        let explicit_zero = EntryMap::from([((0, 1), 0.0), ((1, 1), 2.0)]);
        assert!(entries_match(&explicit_zero, &missing, 0.0));
        // A relative tolerance of 1 accepts anything.
        assert!(entries_match(&local, &missing, 1.0));
    }

    #[test]
    fn check_tolerance_rejects_negative_and_nan() {
        for tol in [0.0, 1e-12, 1.0, f64::INFINITY] {
            assert!(check_tolerance(tol).is_ok(), "{tol}");
        }
        for tol in [-1e-12, -1.0, f64::NAN, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    check_tolerance(tol),
                    Err(HypreError::InvalidParameterPositive)
                ),
                "{tol}"
            );
        }
    }
}
