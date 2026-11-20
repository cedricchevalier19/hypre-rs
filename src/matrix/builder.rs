//! Builders of [CSRMatrix]es: one for matrices whose entries are all given
//! by the MPI rank that owns their row, one for matrices that also take
//! entries for rows owned by other ranks, and one for symmetric matrices.

use super::{convert_entries, mirror_lower, CSRMatrix, Entry, SymmetricMatrix};
use crate::error::HypreError;
use crate::utils::inclusive_bounds;
use crate::{HypreContext, HypreResult};
use hypre_sys::*;
use mpi::ffi::MPI_Comm;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::null_mut;

/// hypre's IJ matrix, created but not necessarily initialized; destroyed on
/// drop.
#[derive(Debug)]
struct IjMatrix<'h> {
    internal: HYPRE_IJMatrix,
    rows: Range<u32>,
    _hypre: PhantomData<&'h ()>,
}

impl<'h> IjMatrix<'h> {
    /// Creates an IJ matrix owning `rows`, whose diagonal block has the
    /// columns `cols`. It must be initialized before entries are given.
    fn create(comm: MPI_Comm, rows: Range<u32>, cols: Range<u32>) -> HypreResult<Self> {
        let (row_lower, row_upper) = inclusive_bounds(&rows)?;
        let (col_lower, col_upper) = inclusive_bounds(&cols)?;
        let mut out = Self {
            internal: null_mut(),
            rows,
            _hypre: PhantomData,
        };
        // If a call below fails, dropping `out` destroys whatever was created.
        unsafe {
            check_hypre!(HYPRE_IJMatrixCreate(
                comm,
                row_lower,
                row_upper,
                col_lower,
                col_upper,
                &mut out.internal,
            ));
            check_hypre!(HYPRE_IJMatrixSetObjectType(
                out.internal,
                HYPRE_PARCSR.try_into()?
            ));
        }
        Ok(out)
    }

    /// Expects at most `max` entries for rows of other ranks; before
    /// [IjMatrix::initialize].
    fn set_max_off_rank(&mut self, max: usize) -> HypreResult<()> {
        unsafe {
            check_hypre!(HYPRE_IJMatrixSetMaxOffProcElmts(
                self.internal,
                max.try_into()?
            ));
        }
        Ok(())
    }

    /// Preallocates `diag[k]` entries in the diagonal block and `offd[k]` in
    /// the off-diagonal block of local row `k`; before
    /// [IjMatrix::initialize].
    fn set_diag_offd_sizes(&mut self, diag: &[HYPRE_Int], offd: &[HYPRE_Int]) -> HypreResult<()> {
        unsafe {
            check_hypre!(HYPRE_IJMatrixSetDiagOffdSizes(
                self.internal,
                diag.as_ptr(),
                offd.as_ptr()
            ));
        }
        Ok(())
    }

    fn initialize(&mut self) -> HypreResult<()> {
        unsafe {
            check_hypre!(HYPRE_IJMatrixInitialize(self.internal));
        }
        Ok(())
    }

    /// Adds (with `HYPRE_IJMatrixAddToValues`) or sets (with
    /// `HYPRE_IJMatrixSetValues`, local rows only) every entry.
    fn insert(&mut self, entries: &[Entry], add: bool) -> HypreResult<()> {
        let rows: Vec<HYPRE_BigInt> = entries.iter().map(|e| e.0).collect();
        let cols: Vec<HYPRE_BigInt> = entries.iter().map(|e| e.1).collect();
        let values: Vec<HYPRE_Complex> = entries.iter().map(|e| e.2).collect();
        // One column per "row" entry: hypre reads entry k as
        // (rows[k], cols[k], values[k]).
        let mut ncols = vec![1 as HYPRE_Int; rows.len()];
        let n = rows.len().try_into()?;
        unsafe {
            if add {
                check_hypre!(HYPRE_IJMatrixAddToValues(
                    self.internal,
                    n,
                    ncols.as_mut_ptr(),
                    rows.as_ptr(),
                    cols.as_ptr(),
                    values.as_ptr()
                ));
            } else {
                check_hypre!(HYPRE_IJMatrixSetValues(
                    self.internal,
                    n,
                    ncols.as_mut_ptr(),
                    rows.as_ptr(),
                    cols.as_ptr(),
                    values.as_ptr()
                ));
            }
        }
        Ok(())
    }

    /// Assembles the matrix and hands its ParCSR object over.
    fn assemble(self) -> HypreResult<CSRMatrix<'h>> {
        let mut parcsr: *mut c_void = null_mut();
        unsafe {
            check_hypre!(HYPRE_IJMatrixAssemble(self.internal));
            check_hypre!(HYPRE_IJMatrixGetObject(self.internal, &mut parcsr));
            // `self` is dropped on return, and HYPRE_IJMatrixDestroy would
            // also destroy the ParCSR object. With object type -1 it only
            // frees the IJ interface, so the returned CSRMatrix becomes the
            // sole owner.
            check_hypre!(HYPRE_IJMatrixSetObjectType(self.internal, -1));
        }
        Ok(CSRMatrix {
            internal_matrix: parcsr as HYPRE_ParCSRMatrix,
            _hypre: PhantomData,
        })
    }
}

impl_hypre_drop!(IjMatrix<'_>, internal, HYPRE_IJMatrixDestroy);

/// Whether a batch of entries replaces or adds to the current values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Set,
    Add,
}

/// Builds a [CSRMatrix] whose entries are all given by the MPI rank that
/// owns their row.
///
/// Entries are kept in order until [LocalMatrixBuilder::assemble], which
/// counts them to preallocate each row exactly (diagonal and off-diagonal
/// blocks), tells hypre to expect no entry for other ranks, and hands them
/// over. Entries can be set (replacing the current value) or added. An
/// entry may be in any column, but its row must be local: use a
/// [DistributedMatrixBuilder] otherwise.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// let n = 10;
/// let rows = hypre_rs::distribute_rows(&comm, n);
/// // tridiag(-1, 2, -1), each rank giving its own rows.
/// let mut a = ctx.local_matrix_builder(&comm, rows.clone(), rows.clone())?;
/// for i in rows.clone() {
///     a.add([(i, i, 2.0)])?;
///     if i > 0 {
///         a.add([(i, i - 1, -1.0)])?;
///     }
///     if i + 1 < n {
///         a.add([(i, i + 1, -1.0)])?;
///     }
/// }
/// let a = a.assemble()?;
/// # drop(a);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
///
/// Solvers need an assembled matrix:
///
/// ```compile_fail
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// let mut a = ctx.local_matrix_builder(&comm, 0..1, 0..1)?;
/// a.add([(0u32, 0u32, 1.0)])?;
/// let b = ctx.local_vector_builder(&comm, 0..1)?.assemble()?;
/// let mut x = ctx.local_vector_builder(&comm, 0..1)?.assemble()?;
/// let solver = ctx.gmres_solver(&comm, Default::default())?;
/// solver.solve(&a, &b, &mut x)?; // `a` is not assembled
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct LocalMatrixBuilder<'h> {
    comm: MPI_Comm,
    rows: Range<u32>,
    cols: Range<u32>,
    /// Entries in the order they were given, grouped by mode.
    batches: Vec<(Mode, Vec<Entry>)>,
    _hypre: PhantomData<&'h ()>,
}

impl<'h> LocalMatrixBuilder<'h> {
    /// Creates a builder of a matrix owning `rows` on the current MPI rank.
    ///
    /// `rows` is the half-open range of global rows owned by this rank (see
    /// [distribute_rows](crate::distribute_rows)), and `cols` the columns of
    /// its diagonal block: the range of the vectors the matrix multiplies
    /// that this rank owns, `rows` for a square matrix. Either may be empty.
    /// The ranges of all ranks must not overlap and must cover the matrix.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if a range's start is greater than its end.
    pub fn new(
        _hypre: &'h HypreContext<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> HypreResult<Self> {
        inclusive_bounds(&rows)?;
        inclusive_bounds(&cols)?;
        Ok(Self {
            comm: comm.as_raw(),
            rows,
            cols,
            batches: Vec::new(),
            _hypre: PhantomData,
        })
    }

    /// Rows owned by the current MPI rank.
    pub fn local_range(&self) -> Range<u32> {
        self.rows.clone()
    }

    /// Converts `entries` and checks that their rows are local.
    fn local_entries<Id, V>(
        &self,
        entries: impl IntoIterator<Item = (Id, Id, V)>,
    ) -> HypreResult<Vec<Entry>>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let entries = convert_entries(entries)?;
        let is_local = |row: HYPRE_BigInt| u32::try_from(row).is_ok_and(|r| self.rows.contains(&r));
        match entries.iter().find(|&&(row, _, _)| !is_local(row)) {
            Some(&(row, _, _)) => Err(HypreError::NotLocalRow(row)),
            None => Ok(entries),
        }
    }

    fn push(&mut self, mode: Mode, entries: Vec<Entry>) {
        match self.batches.last_mut() {
            Some((last, batch)) if *last == mode => batch.extend(entries),
            _ => self.batches.push((mode, entries)),
        }
    }

    /// Sets entry `(row, col)` to `value` for every `(row, col, value)` of
    /// `entries`, replacing what was set or added there before; when an
    /// entry appears several times, the last value wins.
    ///
    /// # Errors
    ///
    /// [HypreError::NotLocalRow] if a row is not owned by this rank, or
    /// [HypreError::ConversionError] if an index or value does not fit
    /// hypre's types. Nothing is changed then.
    pub fn set<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let entries = self.local_entries(entries)?;
        self.push(Mode::Set, entries);
        Ok(())
    }

    /// Adds `value` to entry `(row, col)` for every `(row, col, value)` of
    /// `entries`.
    ///
    /// # Errors
    ///
    /// See [LocalMatrixBuilder::set].
    pub fn add<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let entries = self.local_entries(entries)?;
        self.push(Mode::Add, entries);
        Ok(())
    }

    /// Assembles the matrix into a [CSRMatrix], the format hypre's solvers
    /// use.
    ///
    /// hypre's assembly is collective, even with no entries for other ranks:
    /// every rank must call it.
    ///
    /// # Errors
    ///
    /// The error returned by hypre.
    pub fn assemble(self) -> HypreResult<CSRMatrix<'h>> {
        // Count the entries of each local row in the diagonal block (local
        // columns) and in the off-diagonal one. Repeated entries are
        // counted again: an overestimate, which hypre accepts.
        let mut diag = vec![0 as HYPRE_Int; self.rows.len()];
        let mut offd = vec![0 as HYPRE_Int; self.rows.len()];
        for (_, batch) in &self.batches {
            for &(row, col, _) in batch {
                // Rows were checked to be local when the entries were given.
                let k = usize::try_from(u32::try_from(row)? - self.rows.start)?;
                if u32::try_from(col).is_ok_and(|c| self.cols.contains(&c)) {
                    diag[k] += 1;
                } else {
                    offd[k] += 1;
                }
            }
        }

        let mut matrix = IjMatrix::create(self.comm, self.rows.clone(), self.cols.clone())?;
        matrix.set_diag_offd_sizes(&diag, &offd)?;
        matrix.set_max_off_rank(0)?;
        matrix.initialize()?;
        for (mode, batch) in &self.batches {
            matrix.insert(batch, *mode == Mode::Add)?;
        }
        matrix.assemble()
    }
}

/// Builds a [CSRMatrix] from entries in any row, including rows owned by
/// other MPI ranks, which [DistributedMatrixBuilder::assemble] sends to
/// their owners.
///
/// Values can only be added: hypre cannot set a value on another rank.
/// Finite-element matrices are built this way, each element adding to the
/// rows of its nodes. When every row is local, a [LocalMatrixBuilder] does
/// less work.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// let rows = hypre_rs::distribute_rows(&comm, 10);
/// // Element (i - 1, i) adds the stiffness [[1, -1], [-1, 1]]; row i - 1
/// // may be owned by the previous rank.
/// let mut a = ctx.distributed_matrix_builder(&comm, rows.clone(), rows.clone())?;
/// for i in rows.clone().filter(|&i| i > 0) {
///     a.add([(i - 1, i - 1, 1.0), (i - 1, i, -1.0), (i, i - 1, -1.0), (i, i, 1.0)])?;
/// }
/// let a = a.assemble()?;
/// # drop(a);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct DistributedMatrixBuilder<'h> {
    matrix: IjMatrix<'h>,
}

impl<'h> DistributedMatrixBuilder<'h> {
    /// Creates a builder of a matrix owning `rows` on the current MPI rank;
    /// see [LocalMatrixBuilder::new] for `rows` and `cols`.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if a range's start is greater than its
    /// end, or the error returned by hypre.
    pub fn new(
        hypre: &'h HypreContext<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> HypreResult<Self> {
        Self::create(hypre, comm, rows, cols, None)
    }

    /// Same as [DistributedMatrixBuilder::new], telling hypre to expect at
    /// most `max_off_rank_entries` entries for rows of other ranks from this
    /// one, which hypre recommends for efficiency.
    ///
    /// # Errors
    ///
    /// See [DistributedMatrixBuilder::new].
    pub fn with_off_rank_capacity(
        hypre: &'h HypreContext<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
        cols: Range<u32>,
        max_off_rank_entries: usize,
    ) -> HypreResult<Self> {
        Self::create(hypre, comm, rows, cols, Some(max_off_rank_entries))
    }

    fn create(
        _hypre: &'h HypreContext<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
        cols: Range<u32>,
        max_off_rank: Option<usize>,
    ) -> HypreResult<Self> {
        let mut matrix = IjMatrix::create(comm.as_raw(), rows, cols)?;
        if let Some(max) = max_off_rank {
            matrix.set_max_off_rank(max)?;
        }
        matrix.initialize()?;
        Ok(Self { matrix })
    }

    /// Rows owned by the current MPI rank.
    pub fn local_range(&self) -> Range<u32> {
        self.matrix.rows.clone()
    }

    /// Adds `value` to entry `(row, col)` for every `(row, col, value)` of
    /// `entries`; the row may belong to any rank.
    ///
    /// # Errors
    ///
    /// [HypreError::ConversionError] if an index or value does not fit
    /// hypre's types (nothing is added then), or the error returned by hypre.
    pub fn add<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        self.matrix.insert(&convert_entries(entries)?, true)
    }

    /// Assembles the matrix into a [CSRMatrix], sending the entries added
    /// for other ranks' rows to their owners, so it is collective.
    ///
    /// # Errors
    ///
    /// The error returned by hypre.
    pub fn assemble(self) -> HypreResult<CSRMatrix<'h>> {
        self.matrix.assemble()
    }
}

/// Builds a [SymmetricMatrix] from the entries of its lower triangle.
///
/// hypre stores both triangles, so each off-diagonal entry `(i, j, v)` is
/// also added at `(j, i)`, which often belongs to another MPI rank: like a
/// [DistributedMatrixBuilder], it sends those entries to their owners when
/// assembled.
///
/// # Example
///
/// The 1D Laplacian `tridiag(-1, 2, -1)`:
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use hypre_rs::Matrix;
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// let rows = hypre_rs::distribute_rows(&comm, 10);
/// let mut laplacian = ctx.symmetric_matrix_builder(&comm, rows.clone())?;
/// // Diagonal and sub-diagonal; the super-diagonal is added by symmetry.
/// laplacian.add(rows.clone().map(|i| (i, i, 2.0)))?;
/// laplacian.add(rows.filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
/// let laplacian = laplacian.assemble()?;
/// assert!(laplacian.is_symmetric(0.0)?);
/// # drop(laplacian);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SymmetricMatrixBuilder<'h> {
    matrix: DistributedMatrixBuilder<'h>,
}

impl<'h> SymmetricMatrixBuilder<'h> {
    /// Creates a builder of a square symmetric matrix owning `rows` on the
    /// current MPI rank; the same range is used for the columns (see
    /// [LocalMatrixBuilder::new]).
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if `rows.start > rows.end`, or the error
    /// returned by hypre.
    pub fn new(
        hypre: &'h HypreContext<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
    ) -> HypreResult<Self> {
        Ok(Self {
            matrix: DistributedMatrixBuilder::new(hypre, comm, rows.clone(), rows)?,
        })
    }

    /// Rows owned by the current MPI rank.
    pub fn local_range(&self) -> Range<u32> {
        self.matrix.local_range()
    }

    /// Adds `(row, col, value)` entries of the lower triangle (`col <= row`).
    ///
    /// Each off-diagonal entry is also added at `(col, row)`, so an entry of
    /// the upper triangle must not be given as well: it would be counted
    /// twice. Values are added to existing ones, and rows may belong to any
    /// rank.
    ///
    /// # Errors
    ///
    /// - [HypreError::NotLowerTriangular] if an entry has `col > row`,
    /// - [HypreError::ConversionError] if an index or value does not fit
    ///   hypre's types,
    ///
    /// in which cases nothing is added, or the error returned by hypre.
    pub fn add<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let entries = mirror_lower(convert_entries(entries)?)?;
        self.matrix.matrix.insert(&entries, true)
    }

    /// Assembles the matrix into a [SymmetricMatrix] of a [CSRMatrix], the
    /// format hypre's solvers use, so it is collective.
    ///
    /// # Errors
    ///
    /// The error returned by hypre.
    pub fn assemble(self) -> HypreResult<SymmetricMatrix<CSRMatrix<'h>>> {
        Ok(SymmetricMatrix {
            matrix: self.matrix.assemble()?,
        })
    }
}
