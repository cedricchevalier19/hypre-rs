use crate::error::HypreError;
use crate::utils::inclusive_bounds;
use crate::{HypreContext, HypreResult};
use hypre_sys::*;
use itertools::Itertools;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::null_mut;

/// Distributed vector under construction, filled with
/// [IJVector::add_elements].
///
/// Solvers need an assembled vector: [IJVector::assemble] turns it into a
/// [ParVector].
///
/// ```compile_fail
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// let mut a = ctx.ij_matrix(&comm, 0..1, 0..1)?;
/// a.add_elements([(0u32, 0u32, 1.0)].into_iter())?;
/// let a = a.assemble()?;
/// let b = ctx.ij_vector(&comm, 0..1)?.assemble()?;
/// let mut x = ctx.ij_vector(&comm, 0..1)?;
/// let solver = ctx.gmres_solver(&comm, Default::default())?;
/// solver.solve(&a, &b, &mut x)?; // `x` is not assembled
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct IJVector<'h> {
    internal: HYPRE_IJVector,
    _hypre: PhantomData<&'h ()>,
}

impl<'h> IJVector<'h> {
    /// Creates an IJVector from a communicator `comm` and sizes
    ///
    /// `rows` is the half-open range of global indices owned by the current MPI
    /// rank (`0..10` is 10 rows); it may be empty.
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if `rows.start > rows.end`.
    pub fn new(
        _hypre: &'h HypreContext<'_>,
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

    /// Assembles the vector into a [ParVector], the format hypre's solvers
    /// use.
    ///
    /// Assembly adds the entries given for rows owned by other MPI ranks to
    /// those ranks' rows, so it is collective. Entries cannot be added
    /// afterwards.
    ///
    /// # Errors
    ///
    /// The error returned by hypre.
    pub fn assemble(self) -> HypreResult<ParVector<'h>> {
        let mut par: *mut c_void = null_mut();
        unsafe {
            check_hypre!(HYPRE_IJVectorAssemble(self.internal));
            check_hypre!(HYPRE_IJVectorGetObject(self.internal, &mut par));
            // `self` is dropped on return, and HYPRE_IJVectorDestroy would
            // also destroy the ParVector. With object type -1 it only frees
            // the IJ interface, so the returned ParVector becomes the sole
            // owner.
            check_hypre!(HYPRE_IJVectorSetObjectType(self.internal, -1));
        }
        Ok(ParVector {
            internal: par as HYPRE_ParVector,
            _hypre: PhantomData,
        })
    }

    /// Adds `value` to entry `index` for every `(index, value)` of `nnz`.
    ///
    /// # Errors
    ///
    /// [HypreError::ConversionError] if an index or value does not fit
    /// hypre's types (nothing is added then), or the error returned by hypre.
    pub fn add_elements<Id, V>(&mut self, nnz: impl Iterator<Item = (Id, V)>) -> HypreResult<()>
    where
        Id: Copy + TryInto<HYPRE_BigInt>,
        V: Copy + TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let (indices, values): (Vec<HYPRE_BigInt>, Vec<HYPRE_Complex>) = nnz
            .map(|(index, value)| Ok((index.try_into()?, value.try_into()?)))
            .collect::<HypreResult<Vec<_>>>()?
            .into_iter()
            .multiunzip();

        unsafe {
            check_hypre!(HYPRE_IJVectorAddToValues(
                self.internal,
                indices.len().try_into()?,
                indices.as_ptr(),
                values.as_ptr(),
            ));
        }
        Ok(())
    }

    /// Returns the values of the rows in `rows`, which must be owned by the
    /// current MPI rank.
    ///
    /// This only reads local data, so it is not collective. Contributions
    /// that other MPI ranks added to these rows are not included: they are
    /// only added by [IJVector::assemble] (see [ParVector::get_values]).
    ///
    /// # Errors
    ///
    /// [HypreError::InvalidRange] if `rows.start > rows.end`, or the error
    /// returned by hypre (for example for a row that is not owned locally).
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use mpi::topology::Communicator;
    /// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    /// # let comm = universe.world();
    /// # let ctx = hypre_rs::initialize(&universe)?;
    /// # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
    /// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
    /// let mut vector = ctx.ij_vector(&comm, rows.clone())?;
    /// vector.add_elements(rows.clone().map(|i| (i, f64::from(i))))?;
    /// let values = vector.get_values(rows.clone())?;
    /// assert!(values.iter().zip(rows).all(|(&v, i)| v == f64::from(i)));
    /// # drop(vector);
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_values(&self, rows: Range<u32>) -> HypreResult<Vec<HYPRE_Complex>> {
        if rows.start > rows.end {
            return Err(HypreError::InvalidRange);
        }
        let indices = rows
            .map(HYPRE_BigInt::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let mut values: Vec<HYPRE_Complex> = vec![Default::default(); indices.len()];
        unsafe {
            check_hypre!(HYPRE_IJVectorGetValues(
                self.internal,
                indices.len().try_into()?,
                indices.as_ptr(),
                values.as_mut_ptr(),
            ));
        }
        Ok(values)
    }
}

impl_hypre_drop!(IJVector<'_>, internal, HYPRE_IJVectorDestroy);

/// Assembled distributed vector, which hypre's solvers use; made by
/// [IJVector::assemble].
///
/// Not `Clone`: copying the handle would destroy it twice.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use mpi::topology::Communicator;
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// # let (rank, size) = (comm.rank() as u32, comm.size() as u32);
/// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
/// let mut vector = ctx.ij_vector(&comm, rows.clone())?;
/// vector.add_elements(rows.clone().map(|i| (i, f64::from(i))))?;
/// let vector = vector.assemble()?;
/// let values = vector.get_values(rows.clone())?;
/// assert!(values.iter().zip(rows).all(|(&v, i)| v == f64::from(i)));
/// # drop(vector);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ParVector<'h> {
    internal: HYPRE_ParVector,
    _hypre: PhantomData<&'h ()>,
}

impl ParVector<'_> {
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
                self.internal,
                indices.len().try_into()?,
                indices.as_mut_ptr(),
                values.as_mut_ptr(),
            ));
        }
        Ok(values)
    }

    /// The ParVector object, for hypre's solvers.
    pub(crate) fn as_raw(&self) -> HYPRE_ParVector {
        self.internal
    }
}

impl_hypre_drop!(ParVector<'_>, internal, HYPRE_ParVectorDestroy);
