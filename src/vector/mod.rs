//! Distributed vectors: [ParVector], assembled and read-only, and the
//! builders that make one.

use crate::error::HypreError;
use crate::{HypreContext, HypreResult};
use hypre_sys::*;
use std::marker::PhantomData;
use std::ops::Range;

mod builder;
mod refill;
pub use builder::{DistributedVectorBuilder, LocalVectorBuilder};
pub use refill::VectorRefillBuilder;

/// Assembled distributed vector, which hypre's solvers use; made by a
/// [LocalVectorBuilder] or a [DistributedVectorBuilder], or directly by
/// [ParVector::zeros] and [ParVector::from_fn]. It cannot be modified,
/// except by a solver writing its solution into it, but its values can be
/// replaced: [ParVector::refill] and [ParVector::update] turn it into a
/// [VectorRefillBuilder].
///
/// Not `Clone`: copying the handle would destroy it twice.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// use hypre_rs::ParVector;
///
/// let rows = hypre_rs::distribute_rows(&comm, 10);
/// let mut v = ctx.local_vector_builder(&comm, rows.clone())?;
/// v.set(rows.clone().map(|i| (i, f64::from(i))))?;
/// let v: ParVector<'_> = v.assemble()?;
/// // Or in one step: ParVector::from_fn(&ctx, &comm, rows.clone(), f64::from)?
/// let first = v.get_values(rows.start..rows.start + 1)?;
/// assert!(rows.is_empty() || first == [f64::from(rows.start)]);
/// assert_eq!(v.local_values()?.len(), rows.len());
/// # drop(v);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ParVector<'h> {
    /// hypre's IJ interface, kept to refill the values; destroying it also
    /// destroys `par`.
    ij: HYPRE_IJVector,
    par: HYPRE_ParVector,
    /// Rows owned by this MPI rank.
    rows: Range<u32>,
    _hypre: PhantomData<&'h ()>,
}

impl<'h> ParVector<'h> {
    /// Creates an assembled vector of zeros, owning `rows` on the current
    /// MPI rank (see [LocalVectorBuilder::new]); for example an initial
    /// guess.
    ///
    /// Collective, like [LocalVectorBuilder::assemble].
    ///
    /// # Errors
    ///
    /// See [ParVector::from_fn].
    pub fn zeros(
        hypre: &'h HypreContext<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
    ) -> HypreResult<Self> {
        Self::from_fn(hypre, comm, rows, |_| 0.0)
    }

    /// Creates an assembled vector whose entry `i` is `f(i)`, for the rows
    /// `rows` owned by the current MPI rank (see [LocalVectorBuilder::new]).
    ///
    /// Same as setting the values in a [LocalVectorBuilder] and assembling
    /// it, so it is collective.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if `rows.start > rows.end`, or the error
    /// returned by hypre.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use hypre_rs::ParVector;
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// let rows = hypre_rs::distribute_rows(&comm, 10);
    /// let squares = ParVector::from_fn(&ctx, &comm, rows.clone(), |i| f64::from(i * i))?;
    /// let values = squares.local_values()?;
    /// assert!(rows.zip(values).all(|(i, v)| v == f64::from(i * i)));
    /// # drop(squares);
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_fn(
        hypre: &'h HypreContext<'_>,
        comm: &'h impl mpi::topology::Communicator,
        rows: Range<u32>,
        mut f: impl FnMut(u32) -> HYPRE_Complex,
    ) -> HypreResult<Self> {
        let mut vector = LocalVectorBuilder::new(hypre, comm, rows.clone())?;
        vector.set(rows.map(|i| (i, f(i))))?;
        vector.assemble()
    }

    /// Rows owned by the current MPI rank.
    pub fn local_range(&self) -> Range<u32> {
        self.rows.clone()
    }

    /// Returns the values of all the rows owned by the current MPI rank, in
    /// order: `local_values()?[k]` is row `local_range().start + k`.
    ///
    /// This only reads local data, so it is not collective.
    ///
    /// # Errors
    ///
    /// The error returned by hypre.
    pub fn local_values(&self) -> HypreResult<Vec<HYPRE_Complex>> {
        self.get_values(self.local_range())
    }

    /// Returns the values of the rows in `rows`, which must be owned by the
    /// current MPI rank.
    ///
    /// This only reads local data, so it is not collective.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if `rows.start > rows.end`, or the error
    /// returned by hypre (for example for a row that is not owned locally).
    pub fn get_values(&self, rows: Range<u32>) -> HypreResult<Vec<HYPRE_Complex>> {
        if rows.start > rows.end {
            return Err(HypreError::InvalidRange);
        }
        let mut indices = rows
            .map(HYPRE_BigInt::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let mut values: Vec<HYPRE_Complex> = vec![Default::default(); indices.len()];
        unsafe {
            check_hypre!(HYPRE_ParVectorGetValues(
                self.par,
                indices.len().try_into()?,
                indices.as_mut_ptr(),
                values.as_mut_ptr(),
            ));
        }
        Ok(values)
    }

    /// The ParVector object, for hypre's solvers.
    pub(crate) fn as_raw(&self) -> HYPRE_ParVector {
        self.par
    }
}

impl_hypre_drop!(ParVector<'_>, ij, HYPRE_IJVectorDestroy);
