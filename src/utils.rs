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

pub fn initialize() -> HypreResult<()> {
    use hypre_sys::HYPRE_Initialize;
    unsafe {
        check_hypre!(HYPRE_Initialize());
    }
    Ok(())
}

pub fn finalize() -> HypreResult<()> {
    use hypre_sys::HYPRE_Finalize;
    unsafe {
        check_hypre!(HYPRE_Finalize());
    }
    Ok(())
}

/// Converts a half-open Rust range into hypre's inclusive `(lower, upper)` bounds.
///
/// An empty range `n..n` becomes `(n, n - 1)`, which hypre accepts for an
/// MPI rank that owns no rows.
pub(crate) fn inclusive_bounds(range: &Range<u32>) -> HypreResult<(HYPRE_BigInt, HYPRE_BigInt)> {
    if range.start > range.end {
        return Err(HypreError::InvalidRange);
    }
    let lower: HYPRE_BigInt = range.start.try_into()?;
    let end: HYPRE_BigInt = range.end.try_into()?;
    Ok((lower, end - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inclusive_bounds_conversion() {
        assert_eq!(inclusive_bounds(&(0..10)).unwrap(), (0, 9));
        assert_eq!(inclusive_bounds(&(5..6)).unwrap(), (5, 5));
        assert_eq!(inclusive_bounds(&(0..0)).unwrap(), (0, -1));
        assert!(matches!(
            inclusive_bounds(&Range { start: 5, end: 3 }),
            Err(HypreError::InvalidRange)
        ));
    }
}
