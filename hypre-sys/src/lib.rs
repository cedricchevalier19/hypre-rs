#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use mpi_sys::*;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Fallback for hypre < 2.29, which only provides `HYPRE_Init`.
///
/// # Safety
/// Same contract as the C function: call once, before any other hypre call.
#[cfg(not(hypre_has_initialize))]
pub unsafe fn HYPRE_Initialize() -> HYPRE_Int {
    HYPRE_Init()
}
