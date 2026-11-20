//! Refill builders: new values for an assembled matrix, with the same
//! sparsity pattern.
//!
//! Nonlinear solvers (Newton's method, for example) assemble a matrix with
//! the same pattern again and again, only with new values. A refill builder
//! consumes the assembled matrix, takes the new values, and gives back an
//! assembled matrix, reusing hypre's structure instead of building a new
//! one. Assembled matrices stay immutable: their values only change through
//! a refill builder, which no solver can be using meanwhile.

use super::builder::insert;
use super::{convert_entries, mirror_lower, CSRMatrix, Entry, SymmetricMatrix};
use crate::error::HypreError;
use crate::HypreResult;
use hypre_sys::*;

impl<'h> CSRMatrix<'h> {
    /// Turns the matrix into a [MatrixRefillBuilder] with all values set to
    /// 0, to give it new values with the same sparsity pattern; for
    /// example a new Jacobian in a Newton iteration.
    ///
    /// # Errors
    ///
    /// The error returned by hypre (the matrix is dropped then).
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use hypre_rs::prelude::*;
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// let rows = distribute_rows(&comm, 10);
    /// let mut a = ctx.local_matrix_builder(&comm, rows.clone(), rows.clone())?;
    /// a.add(rows.clone().map(|i| (i, i, 1.0)))?;
    /// let a = a.assemble()?;
    ///
    /// // Same pattern, new values: a = 2 I.
    /// let mut a = a.refill()?;
    /// a.add(rows.clone().map(|i| (i, i, 2.0)))?;
    /// let a = a.assemble()?;
    ///
    /// let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |_| 4.0)?;
    /// let mut x = ParVector::zeros(&ctx, &comm, rows)?;
    /// ctx.gmres_solver(&comm, Default::default())?.solve(&a, &b, &mut x)?;
    /// assert!(x.local_values()?.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    /// # drop((a, b, x));
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn refill(self) -> HypreResult<MatrixRefillBuilder<'h>> {
        unsafe {
            check_hypre!(HYPRE_IJMatrixSetConstantValues(self.ij, 0.0));
        }
        Ok(self.update())
    }

    /// Turns the matrix into a [MatrixRefillBuilder] that keeps the current
    /// values, to set or add new ones on top, with the same sparsity
    /// pattern.
    pub fn update(self) -> MatrixRefillBuilder<'h> {
        MatrixRefillBuilder { matrix: self }
    }
}

/// Converts `entries` and checks that their rows are in `rows`.
fn local_entries<Id, V>(
    rows: &std::ops::Range<u32>,
    entries: impl IntoIterator<Item = (Id, Id, V)>,
) -> HypreResult<Vec<Entry>>
where
    Id: Copy + TryInto<HYPRE_BigInt>,
    V: Copy + TryInto<HYPRE_Complex>,
    HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
    HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
{
    let entries = convert_entries(entries)?;
    let is_local = |row: HYPRE_BigInt| u32::try_from(row).is_ok_and(|r| rows.contains(&r));
    match entries.iter().find(|&&(row, _, _)| !is_local(row)) {
        Some(&(row, _, _)) => Err(HypreError::NotLocalRow(row)),
        None => Ok(entries),
    }
}

/// New values for an assembled [CSRMatrix], with the same sparsity
/// pattern; made by [CSRMatrix::refill] (from zeros) or
/// [CSRMatrix::update] (from the current values).
///
/// Entries must already be in the pattern: hypre returns an error for a new
/// one (from `add`/`set` for a local row, from `assemble` for a row of
/// another rank). Values can be added to any row, including rows owned by
/// other MPI ranks, and set in local rows.
#[derive(Debug)]
pub struct MatrixRefillBuilder<'h> {
    matrix: CSRMatrix<'h>,
}

impl<'h> MatrixRefillBuilder<'h> {
    /// Rows owned by the current MPI rank.
    pub fn local_range(&self) -> std::ops::Range<u32> {
        self.matrix.local_range()
    }

    /// Sets entry `(row, col)` to `value` for every `(row, col, value)` of
    /// `entries`; rows must be local.
    ///
    /// # Errors
    ///
    /// [HypreError::NotLocalRow] if a row is not owned by this rank, or
    /// [HypreError::ConversionError] if an index or value does not fit
    /// hypre's types (nothing is changed then), or the error returned by
    /// hypre (for example for an entry outside the pattern).
    pub fn set<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let entries = local_entries(&self.matrix.rows, entries)?;
        insert(self.matrix.ij, &entries, false)
    }

    /// Adds `value` to entry `(row, col)` for every `(row, col, value)` of
    /// `entries`; the row may belong to any rank.
    ///
    /// # Errors
    ///
    /// [HypreError::ConversionError] if an index or value does not fit
    /// hypre's types (nothing is added then), or the error returned by hypre
    /// (for example for a local entry outside the pattern).
    pub fn add<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        insert(self.matrix.ij, &convert_entries(entries)?, true)
    }

    /// Assembles the matrix again, sending the entries added for other
    /// ranks' rows to their owners, so it is collective. hypre's matrix
    /// object is reused.
    ///
    /// # Errors
    ///
    /// The error returned by hypre, for example for an entry outside the
    /// pattern added from another rank.
    pub fn assemble(self) -> HypreResult<CSRMatrix<'h>> {
        unsafe {
            check_hypre!(HYPRE_IJMatrixAssemble(self.matrix.ij));
        }
        Ok(self.matrix)
    }
}

impl<'h> SymmetricMatrix<CSRMatrix<'h>> {
    /// Turns the matrix into a [SymmetricMatrixRefillBuilder] with all
    /// values set to 0, to give it new symmetric values with the same
    /// sparsity pattern.
    ///
    /// # Errors
    ///
    /// The error returned by hypre (the matrix is dropped then).
    pub fn refill(self) -> HypreResult<SymmetricMatrixRefillBuilder<'h>> {
        Ok(SymmetricMatrixRefillBuilder {
            matrix: self.matrix.refill()?,
        })
    }

    /// Turns the matrix into a [SymmetricMatrixRefillBuilder] that keeps the
    /// current values, to add new symmetric ones on top.
    pub fn update(self) -> SymmetricMatrixRefillBuilder<'h> {
        SymmetricMatrixRefillBuilder {
            matrix: self.matrix.update(),
        }
    }
}

/// New values for an assembled [SymmetricMatrix], with the same sparsity
/// pattern, given by their lower triangle like in a
/// [SymmetricMatrixBuilder](super::SymmetricMatrixBuilder); made by
/// [SymmetricMatrix::refill] or [SymmetricMatrix::update].
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use hypre_rs::prelude::*;
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// let rows = distribute_rows(&comm, 10);
/// let lower = |scale: f64| {
///     let rows = rows.clone();
///     rows.flat_map(move |i| {
///         let mut row = vec![(i, i, 2.0 * scale)];
///         if i > 0 {
///             row.push((i, i - 1, -scale));
///         }
///         row
///     })
/// };
/// let mut a = ctx.symmetric_matrix_builder(&comm, rows.clone())?;
/// a.add(lower(1.0))?;
/// let a = a.assemble()?;
///
/// // Twice the matrix, still symmetric.
/// let mut a = a.refill()?;
/// a.add(lower(2.0))?;
/// let a = a.assemble()?;
/// assert!(a.is_symmetric(0.0)?);
/// # drop(a);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SymmetricMatrixRefillBuilder<'h> {
    matrix: MatrixRefillBuilder<'h>,
}

impl<'h> SymmetricMatrixRefillBuilder<'h> {
    /// Rows owned by the current MPI rank.
    pub fn local_range(&self) -> std::ops::Range<u32> {
        self.matrix.local_range()
    }

    /// Adds `(row, col, value)` entries of the lower triangle (`col <= row`),
    /// each off-diagonal one also at `(col, row)`; see
    /// [SymmetricMatrixBuilder::add](super::SymmetricMatrixBuilder::add).
    ///
    /// # Errors
    ///
    /// [HypreError::NotLowerTriangular] if an entry has `col > row`,
    /// [HypreError::ConversionError] if an index or value does not fit
    /// hypre's types (nothing is added then), or the error returned by hypre
    /// (for example for an entry outside the pattern).
    pub fn add<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let entries = mirror_lower(convert_entries(entries)?)?;
        insert(self.matrix.matrix.ij, &entries, true)
    }

    /// Assembles the matrix again; see [MatrixRefillBuilder::assemble].
    ///
    /// # Errors
    ///
    /// See [MatrixRefillBuilder::assemble].
    pub fn assemble(self) -> HypreResult<SymmetricMatrix<CSRMatrix<'h>>> {
        Ok(SymmetricMatrix {
            matrix: self.matrix.assemble()?,
        })
    }
}
