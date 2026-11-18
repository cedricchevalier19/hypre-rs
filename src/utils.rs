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
        let bounds = inclusive_bounds(&(0..u32::MAX));
        if size_of::<HYPRE_BigInt>() == 8 {
            assert_eq!(bounds.ok(), Some((0, HYPRE_BigInt::from(u32::MAX) - 1)));
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
