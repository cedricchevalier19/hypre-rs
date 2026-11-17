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
/// An empty range `n..n` becomes `(n, n - 1)`, which hypre accepts for a
/// process that owns no rows.
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
