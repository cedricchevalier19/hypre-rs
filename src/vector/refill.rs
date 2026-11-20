//! Refill builder: new values for an assembled vector.
//!
//! Nonlinear solvers compute a new right-hand side (a residual, for
//! example) at every iteration. A refill builder consumes the assembled
//! vector, takes the new values, and gives back an assembled vector,
//! reusing hypre's object. Assembled vectors stay immutable: their values
//! only change through a refill builder, which no solver can be using
//! meanwhile.

use super::builder::{add_values, convert, set_values};
use super::ParVector;
use crate::error::HypreError;
use crate::HypreResult;
use hypre_sys::*;
use std::ops::Range;

impl<'h> ParVector<'h> {
    /// Turns the vector into a [VectorRefillBuilder] with all values set to
    /// 0, to give it new values; for example a new residual in a Newton
    /// iteration.
    ///
    /// # Errors
    ///
    /// The error returned by hypre (the vector is dropped then).
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
    /// let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |_| 1.0)?;
    ///
    /// let mut b = b.refill()?;
    /// b.set(rows.clone().map(|i| (i, f64::from(i))))?;
    /// let b = b.assemble()?;
    /// assert!(rows.zip(b.local_values()?).all(|(i, v)| v == f64::from(i)));
    /// # drop(b);
    /// # ctx.finalize()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn refill(self) -> HypreResult<VectorRefillBuilder<'h>> {
        unsafe {
            check_hypre!(HYPRE_ParVectorSetConstantValues(self.par, 0.0));
        }
        Ok(self.update())
    }

    /// Turns the vector into a [VectorRefillBuilder] that keeps the current
    /// values, to set or add new ones on top.
    pub fn update(self) -> VectorRefillBuilder<'h> {
        VectorRefillBuilder { vector: self }
    }
}

/// New values for an assembled [ParVector]; made by [ParVector::refill]
/// (from zeros) or [ParVector::update] (from the current values).
///
/// Values can be added to any row, including rows owned by other MPI ranks,
/// and set in local rows.
#[derive(Debug)]
pub struct VectorRefillBuilder<'h> {
    vector: ParVector<'h>,
}

impl<'h> VectorRefillBuilder<'h> {
    /// Rows owned by the current MPI rank.
    pub fn local_range(&self) -> Range<u32> {
        self.vector.local_range()
    }

    /// Sets entry `index` to `value` for every `(index, value)` of
    /// `entries`; indices must be local.
    ///
    /// # Errors
    ///
    /// [HypreError::NotLocalRow] if an index is not owned by this rank, or
    /// [HypreError::ConversionError] if an index or value does not fit
    /// hypre's types (nothing is changed then), or the error returned by
    /// hypre.
    pub fn set<Id, V>(&mut self, entries: impl IntoIterator<Item = (Id, V)>) -> HypreResult<()>
    where
        Id: TryInto<HYPRE_BigInt>,
        V: TryInto<HYPRE_Complex>,
        HypreError: From<<Id as TryInto<HYPRE_BigInt>>::Error>,
        HypreError: From<<V as TryInto<HYPRE_Complex>>::Error>,
    {
        let (indices, values) = convert(entries)?;
        let rows = &self.vector.rows;
        let is_local = |index: HYPRE_BigInt| u32::try_from(index).is_ok_and(|r| rows.contains(&r));
        if let Some(&index) = indices.iter().find(|&&index| !is_local(index)) {
            return Err(HypreError::NotLocalRow(index));
        }
        set_values(self.vector.ij, &indices, &values)
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
        add_values(self.vector.ij, &indices, &values)
    }

    /// Assembles the vector again, sending the entries added for other
    /// ranks' rows to their owners, so it is collective. hypre's vector
    /// object is reused.
    ///
    /// # Errors
    ///
    /// The error returned by hypre.
    pub fn assemble(self) -> HypreResult<ParVector<'h>> {
        unsafe {
            check_hypre!(HYPRE_IJVectorAssemble(self.vector.ij));
        }
        Ok(self.vector)
    }
}
