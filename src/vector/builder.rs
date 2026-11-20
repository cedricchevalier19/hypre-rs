//! Builders of [ParVector]s: one for vectors whose entries are all given
//! by the MPI rank that owns them, one for vectors that also take entries
//! for rows owned by other ranks.

use super::ParVector;
use crate::error::HypreError;
use crate::utils::inclusive_bounds;
use crate::{HypreContext, HypreResult};
use hypre_sys::*;
use mpi::ffi::MPI_Comm;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::null_mut;

/// hypre's IJ vector, initialized; destroyed on drop.
#[derive(Debug)]
struct IjVector<'h> {
    internal: HYPRE_IJVector,
    rows: Range<u32>,
    _hypre: PhantomData<&'h ()>,
}

impl<'h> IjVector<'h> {
    /// Creates and initializes an IJ vector owning `rows`.
    // No `HYPRE_IJVectorSetMaxOffProcElmts`: with a positive value, the next
    // off-rank `AddToValues` crashes (hypre 3.2.1 when set before
    // `Initialize`, 2.26 when set after). hypre's default grows the off-rank
    // buffer as needed.
    fn new(comm: MPI_Comm, rows: Range<u32>) -> HypreResult<Self> {
        let (lower, upper) = inclusive_bounds(&rows)?;
        let mut out = Self {
            internal: null_mut(),
            rows,
            _hypre: PhantomData,
        };
        // If a call below fails, dropping `out` destroys whatever was created.
        unsafe {
            check_hypre!(HYPRE_IJVectorCreate(comm, lower, upper, &mut out.internal));
            check_hypre!(HYPRE_IJVectorSetObjectType(
                out.internal,
                HYPRE_PARCSR.try_into()?
            ));
            check_hypre!(HYPRE_IJVectorInitialize(out.internal));
        }
        Ok(out)
    }

    /// Adds `values[k]` to entry `indices[k]`, which may be on another rank.
    fn add(&mut self, indices: &[HYPRE_BigInt], values: &[HYPRE_Complex]) -> HypreResult<()> {
        add_values(self.internal, indices, values)
    }

    /// Sets entry `indices[k]`, which must be local, to `values[k]`.
    fn set(&mut self, indices: &[HYPRE_BigInt], values: &[HYPRE_Complex]) -> HypreResult<()> {
        set_values(self.internal, indices, values)
    }

    /// Assembles the vector into a [ParVector], which takes the IJ handle
    /// over (to refill the values later).
    fn assemble(mut self) -> HypreResult<ParVector<'h>> {
        let mut par: *mut c_void = null_mut();
        unsafe {
            check_hypre!(HYPRE_IJVectorAssemble(self.internal));
            check_hypre!(HYPRE_IJVectorGetObject(self.internal, &mut par));
        }
        // From here on the ParVector destroys the handle, not `self`.
        let ij = std::mem::replace(&mut self.internal, null_mut());
        Ok(ParVector {
            ij,
            par: par as HYPRE_ParVector,
            rows: self.rows.clone(),
            _hypre: PhantomData,
        })
    }
}

/// Adds `values[k]` to entry `indices[k]` of `ij`, which may be on another
/// rank.
pub(super) fn add_values(
    ij: HYPRE_IJVector,
    indices: &[HYPRE_BigInt],
    values: &[HYPRE_Complex],
) -> HypreResult<()> {
    unsafe {
        check_hypre!(HYPRE_IJVectorAddToValues(
            ij,
            indices.len().try_into()?,
            indices.as_ptr(),
            values.as_ptr(),
        ));
    }
    Ok(())
}

/// Sets entry `indices[k]` of `ij`, which must be local, to `values[k]`.
pub(super) fn set_values(
    ij: HYPRE_IJVector,
    indices: &[HYPRE_BigInt],
    values: &[HYPRE_Complex],
) -> HypreResult<()> {
    unsafe {
        check_hypre!(HYPRE_IJVectorSetValues(
            ij,
            indices.len().try_into()?,
            indices.as_ptr(),
            values.as_ptr(),
        ));
    }
    Ok(())
}

impl_hypre_drop!(IjVector<'_>, internal, HYPRE_IJVectorDestroy);

/// Converts `(index, value)` entries to hypre's types.
pub(super) fn convert<Id, V>(
    entries: impl IntoIterator<Item = (Id, V)>,
) -> HypreResult<(Vec<HYPRE_BigInt>, Vec<HYPRE_Complex>)>
where
    Id: TryInto<HYPRE_BigInt>,
    V: TryInto<HYPRE_Complex>,
    HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
    HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
{
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for (index, value) in entries {
        indices.push(index.try_into()?);
        values.push(value.try_into()?);
    }
    Ok((indices, values))
}

/// Builds a [ParVector] whose entries are all given by the MPI rank that
/// owns them.
///
/// Values live in a local array until [LocalVectorBuilder::assemble], which
/// hands them to hypre in a single local call.
/// Entries can be set (replacing the current value) or added. Entries for
/// rows of other ranks are rejected: use a [DistributedVectorBuilder] for
/// those.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// let rows = hypre_rs::distribute_rows(&comm, 10);
/// let mut b = ctx.local_vector_builder(&comm, rows.clone())?;
/// b.set(rows.clone().map(|i| (i, 1.0)))?;
/// b.add(rows.clone().map(|i| (i, f64::from(i))))?;
/// let b = b.assemble()?;
/// assert!(rows.zip(b.local_values()?).all(|(i, v)| v == 1.0 + f64::from(i)));
/// # drop(b);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
///
/// Solvers need an assembled vector:
///
/// ```compile_fail
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// let mut a = ctx.local_matrix_builder(&comm, 0..1, 0..1)?;
/// a.add([(0u32, 0u32, 1.0)])?;
/// let a = a.assemble()?;
/// let b = ctx.local_vector_builder(&comm, 0..1)?.assemble()?;
/// let mut x = ctx.local_vector_builder(&comm, 0..1)?;
/// let solver = ctx.gmres_solver(&comm, Default::default())?;
/// solver.solve(&a, &b, &mut x)?; // `x` is not assembled
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct LocalVectorBuilder<'h> {
    comm: MPI_Comm,
    rows: Range<u32>,
    /// `values[k]` is row `rows.start + k`.
    values: Vec<HYPRE_Complex>,
    _hypre: PhantomData<&'h ()>,
}

impl<'h> LocalVectorBuilder<'h> {
    /// Creates a builder of a vector owning `rows` on the current MPI rank,
    /// with all entries 0.
    ///
    /// `rows` is the half-open range of global rows owned by this rank (see
    /// [distribute_rows](crate::distribute_rows)); it may be empty. The
    /// ranges of all ranks must not overlap and must cover the vector.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if `rows.start > rows.end`.
    pub fn new(
        _hypre: &'h HypreContext<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
    ) -> HypreResult<Self> {
        inclusive_bounds(&rows)?;
        Ok(Self {
            comm: comm.as_raw(),
            values: vec![0.0; rows.len()],
            rows,
            _hypre: PhantomData,
        })
    }

    /// Rows owned by the current MPI rank.
    pub fn local_range(&self) -> Range<u32> {
        self.rows.clone()
    }

    /// Positions in `values` of `indices`.
    ///
    /// # Errors
    ///
    /// [HypreError::NotLocalRow] for the first index this rank does not own.
    fn positions(&self, indices: &[HYPRE_BigInt]) -> HypreResult<Vec<usize>> {
        indices
            .iter()
            .map(|&index| match u32::try_from(index) {
                Ok(row) if self.rows.contains(&row) => Ok(usize::try_from(row - self.rows.start)?),
                _ => Err(HypreError::NotLocalRow(index)),
            })
            .collect()
    }

    /// Sets entry `index` to `value` for every `(index, value)` of
    /// `entries`; when an index appears several times, the last value wins.
    ///
    /// # Errors
    ///
    /// [HypreError::NotLocalRow] if an index is not owned by this rank, or
    /// [HypreError::ConversionError] if an index or value does not fit
    /// hypre's types. Nothing is changed then.
    pub fn set<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, V)>) -> HypreResult<()>
    where
        Id: TryInto<HYPRE_BigInt>,
        V: TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let (indices, values) = convert(entries)?;
        for (position, value) in self.positions(&indices)?.into_iter().zip(values) {
            self.values[position] = value;
        }
        Ok(())
    }

    /// Adds `value` to entry `index` for every `(index, value)` of
    /// `entries`.
    ///
    /// # Errors
    ///
    /// See [LocalVectorBuilder::set].
    pub fn add<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, V)>) -> HypreResult<()>
    where
        Id: TryInto<HYPRE_BigInt>,
        V: TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let (indices, values) = convert(entries)?;
        for (position, value) in self.positions(&indices)?.into_iter().zip(values) {
            self.values[position] += value;
        }
        Ok(())
    }

    /// Assembles the vector into a [ParVector], the format hypre's solvers
    /// use.
    ///
    /// hypre's assembly is collective, even with no entries for other ranks:
    /// every rank must call it.
    ///
    /// # Errors
    ///
    /// The error returned by hypre.
    pub fn assemble(self) -> HypreResult<ParVector<'h>> {
        let mut vector = IjVector::new(self.comm, self.rows.clone())?;
        let indices = self
            .rows
            .clone()
            .map(HYPRE_BigInt::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        vector.set(&indices, &self.values)?;
        vector.assemble()
    }
}

/// Builds a [ParVector] from entries for any row, including rows owned by
/// other MPI ranks, which [DistributedVectorBuilder::assemble] sends to
/// their owners.
///
/// Values can only be added: hypre cannot set a value on another rank.
/// Finite-element load vectors are built this way, each element adding to
/// the rows of its nodes. When every entry is local, a
/// [LocalVectorBuilder] does less work.
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
/// // Element (i - 1, i) adds 0.5 to both of its nodes, one of which may be
/// // owned by the previous rank.
/// let mut b = ctx.distributed_vector_builder(&comm, rows.clone())?;
/// for i in rows.clone().filter(|&i| i > 0) {
///     b.add([(i - 1, 0.5), (i, 0.5)])?;
/// }
/// let b = b.assemble()?;
/// let expected = |i: u32| if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
/// assert!(rows.zip(b.local_values()?).all(|(i, v)| v == expected(i)));
/// # drop(b);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct DistributedVectorBuilder<'h> {
    vector: IjVector<'h>,
}

impl<'h> DistributedVectorBuilder<'h> {
    /// Creates a builder of a vector owning `rows` on the current MPI rank,
    /// with all entries 0; see [LocalVectorBuilder::new] for `rows`.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if `rows.start > rows.end`, or the error
    /// returned by hypre.
    pub fn new(
        _hypre: &'h HypreContext<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
    ) -> HypreResult<Self> {
        Ok(Self {
            vector: IjVector::new(comm.as_raw(), rows)?,
        })
    }

    /// Rows owned by the current MPI rank.
    pub fn local_range(&self) -> Range<u32> {
        self.vector.rows.clone()
    }

    /// Adds `value` to entry `index` for every `(index, value)` of
    /// `entries`; the index may belong to any rank.
    ///
    /// # Errors
    ///
    /// [HypreError::ConversionError] if an index or value does not fit
    /// hypre's types (nothing is added then), or the error returned by hypre.
    pub fn add<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, V)>) -> HypreResult<()>
    where
        Id: TryInto<HYPRE_BigInt>,
        V: TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let (indices, values) = convert(entries)?;
        self.vector.add(&indices, &values)
    }

    /// Assembles the vector into a [ParVector], sending the entries added
    /// for other ranks' rows to their owners, so it is collective.
    ///
    /// # Errors
    ///
    /// The error returned by hypre.
    pub fn assemble(self) -> HypreResult<ParVector<'h>> {
        self.vector.assemble()
    }
}
