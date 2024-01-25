use crate::{HypreError, HypreResult};

macro_rules! check_hypre {
    ( $res:expr) => {{
        let res = $res;
        if res != 0 {
            return Err(HypreError::new(res));
        }
    }};
}

pub fn initialize() -> HypreResult<()> {
    #[cfg(target_os = "macos")]
    {
        use hypre_sys::HYPRE_Initialize;
        unsafe {
            check_hypre!(HYPRE_Initialize());
        }
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
