use crate::{HypreError, HypreResult};
use hypre_sys::HYPRE_BigInt;
use std::ops::Range;

macro_rules! check_hypre {
    ( $res:expr) => {{
        let res = $res;
        if res != 0 {
            return Err(HypreError::new(res));
        }
    }};
}

/// Implements [Drop] for a wrapper owning a raw hypre handle.
///
/// `impl_hypre_drop!(Type, field, HYPRE_XxxDestroy)` destroys `self.field`
/// with the given hypre function. A null handle (`Create` failed or never
/// ran) is skipped. `drop` cannot report errors, so a failing destroy has its
/// error code cleared from hypre's global error state rather than leaking into
/// the next, unrelated, hypre call.
macro_rules! impl_hypre_drop {
    ($ty:ty, $field:ident, $destroy:path) => {
        impl Drop for $ty {
            /// Calls hypre's destroy function on the owned handle.
            fn drop(&mut self) {
                if !self.$field.is_null() {
                    let code = unsafe { $destroy(self.$field) };
                    $crate::utils::discard_error(code);
                }
            }
        }
    };
}

/// Clears a hypre error code that cannot be reported to the caller.
pub(crate) fn discard_error(code: hypre_sys::HYPRE_Int) {
    if code != 0 {
        // Building the error clears hypre's global error flag.
        let _ = HypreError::new(code);
    }
}

/// Converts a half-open Rust range into hypre's inclusive `(lower, upper)` bounds.
///
/// An empty range `n..n` becomes `(n, n - 1)`, which hypre accepts for an
/// MPI rank that owns no rows.
// `HYPRE_BigInt` is `i64` or `i32` depending on how hypre was built
// (`--enable-bigint`). The conversions below are infallible only in the first
// case, where clippy flags them.
#[allow(clippy::unnecessary_fallible_conversions)]
pub(crate) fn inclusive_bounds(range: &Range<u32>) -> HypreResult<(HYPRE_BigInt, HYPRE_BigInt)> {
    if range.start > range.end {
        return Err(HypreError::InvalidRange);
    }
    let lower: HYPRE_BigInt = range.start.try_into()?;
    let end: HYPRE_BigInt = range.end.try_into()?;
    Ok((lower, end - 1))
}

/// Distributes `n` global rows over the MPI ranks of `comm` in balanced,
/// contiguous blocks, and returns the half-open range owned by this rank.
///
/// Rank `r` of `size` gets rows `r * n / size .. (r + 1) * n / size`: block
/// sizes differ by at most one, and ranks beyond `n` get an empty range.
/// The result is what [IJMatrix::new](crate::matrix::IJMatrix::new),
/// [IJVector::new](crate::IJVector::new) and the context's factories expect.
/// It is not collective: every rank computes its own range.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// # let comm = universe.world();
/// # let ctx = hypre_rs::initialize(&universe)?;
/// let rows = hypre_rs::distribute_rows(&comm, 100);
/// let x = hypre_rs::ParVector::zeros(&ctx, &comm, rows)?;
/// # drop(x);
/// # ctx.finalize()?;
/// # Ok(())
/// # }
/// ```
pub fn distribute_rows(comm: &impl mpi::topology::Communicator, n: u32) -> Range<u32> {
    // An MPI rank and size are never negative.
    balanced_block(comm.rank().unsigned_abs(), comm.size().unsigned_abs(), n)
}

/// Rows `rank * n / size .. (rank + 1) * n / size`, computed in 64 bits so
/// that `rank * n` cannot overflow.
fn balanced_block(rank: u32, size: u32, n: u32) -> Range<u32> {
    let bound = |r: u32| {
        let b = u64::from(r) * u64::from(n) / u64::from(size);
        // b <= n because r <= size.
        u32::try_from(b).unwrap_or(n)
    };
    bound(rank)..bound(rank + 1)
}

/// Serializes the unit tests that read or change hypre's global error flag:
/// tests of one binary run in parallel threads.
#[cfg(test)]
pub(crate) fn hypre_error_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A failed test poisons the lock; the others can still run.
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_blocks_cover_all_rows() {
        for (n, size) in [(10, 1), (10, 3), (100, 7), (2, 5), (0, 4), (u32::MAX, 3)] {
            let blocks: Vec<_> = (0..size).map(|r| balanced_block(r, size, n)).collect();
            // Contiguous, starting at 0 and ending at n.
            assert_eq!(blocks[0].start, 0);
            assert_eq!(blocks[blocks.len() - 1].end, n);
            assert!(blocks.windows(2).all(|w| w[0].end == w[1].start));
            // Balanced: sizes differ by at most one.
            let lens: Vec<_> = blocks.iter().map(|b| b.len()).collect();
            let (min, max) = (lens.iter().min(), lens.iter().max());
            assert!(
                matches!((min, max), (Some(a), Some(b)) if b - a <= 1),
                "{lens:?}"
            );
        }
    }

    #[test]
    fn inclusive_bounds_conversion() -> HypreResult<()> {
        assert_eq!(inclusive_bounds(&(0..10))?, (0, 9));
        assert_eq!(inclusive_bounds(&(5..6))?, (5, 5));
        assert_eq!(inclusive_bounds(&(0..0))?, (0, -1));
        assert!(matches!(
            inclusive_bounds(&Range { start: 5, end: 3 }),
            Err(HypreError::InvalidRange)
        ));
        // A reversed empty range is rejected too, not only a "negative" one.
        assert!(matches!(
            inclusive_bounds(&Range { start: 1, end: 0 }),
            Err(HypreError::InvalidRange)
        ));
        Ok(())
    }

    #[test]
    fn inclusive_bounds_largest_range() {
        // u32::MAX fits in a 64-bit HYPRE_BigInt, not in a 32-bit one.
        // Widened to i128 to compare the same way for both sizes.
        let bounds = inclusive_bounds(&(0..u32::MAX)).map(|(l, u)| (i128::from(l), i128::from(u)));
        if size_of::<HYPRE_BigInt>() == 8 {
            assert_eq!(bounds.ok(), Some((0, i128::from(u32::MAX) - 1)));
        } else {
            assert!(matches!(bounds, Err(HypreError::ConversionError(_))));
        }
    }

    #[test]
    fn discard_error_clears_hypre_error_flag() {
        let _lock = hypre_error_lock();
        discard_error(0);
        assert_eq!(unsafe { hypre_sys::HYPRE_GetError() }, 0);

        // Destroying a null handle is a hypre error that sets the flag.
        let code = unsafe { hypre_sys::HYPRE_IJVectorDestroy(std::ptr::null_mut()) };
        assert_ne!(code, 0);
        assert_ne!(unsafe { hypre_sys::HYPRE_GetError() }, 0);
        discard_error(code);
        assert_eq!(unsafe { hypre_sys::HYPRE_GetError() }, 0);
    }
}
