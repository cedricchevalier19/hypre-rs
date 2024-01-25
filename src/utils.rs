use crate::{HypreError, HypreResult};
use hypre_sys::{HYPRE_Finalize, HYPRE_Initialize};

macro_rules! check_hypre {
    ( $res:expr) => {{
        let res = $res;
        if res != 0 {
            return Err(HypreError::new(res));
        }
    }};
}

pub fn initialize() -> HypreResult<()> {
    unsafe {
        check_hypre!(HYPRE_Initialize());
    }
    Ok(())
}

pub fn finalize() -> HypreResult<()> {
    unsafe {
        check_hypre!(HYPRE_Finalize());
    }
    Ok(())
}
