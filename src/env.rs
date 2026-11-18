//! hypre's global state: initialization, finalization and the [Hypre] token.

use crate::{HypreError, HypreResult};
use hypre_sys::{HYPRE_Finalize, HYPRE_Initialize};
use mpi::environment::Universe;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

/// Whether a [Hypre] token is alive (hypre is initialized).
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Proof that hypre is initialized, borrowing the MPI [Universe].
///
/// Every hypre object borrows this token, so the borrow checker guarantees
/// that:
///
/// - all hypre objects are dropped before hypre is finalized, and
/// - hypre is finalized before MPI (dropping the `Universe` calls
///   `MPI_Finalize`).
///
/// Only one token can exist at a time. It is neither `Send` nor `Sync`:
/// hypre objects are created and used on the thread that initialized hypre.
///
/// # Example
///
/// ```
/// # use mpi::topology::Communicator;
/// use hypre_rs::matrix::IJMatrix;
///
/// let universe = mpi::initialize().unwrap();
/// let hypre = hypre_rs::initialize(&universe)?;
/// let comm = universe.world();
/// let (rank, size) = (comm.rank() as u32, comm.size() as u32);
/// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
///
/// let matrix = IJMatrix::new(&hypre, &comm, rows.clone(), rows)?;
/// drop(matrix);
/// hypre.finalize()?;
/// # Ok::<(), hypre_rs::HypreError>(())
/// ```
///
/// Finalizing hypre while an object still uses it does not compile:
///
/// ```compile_fail,E0505
/// # use mpi::topology::Communicator;
/// # use hypre_rs::matrix::IJMatrix;
/// let universe = mpi::initialize().unwrap();
/// let hypre = hypre_rs::initialize(&universe)?;
/// let comm = universe.world();
/// let matrix = IJMatrix::new(&hypre, &comm, 0..0, 0..0)?;
/// hypre.finalize()?; // error: `hypre` is still borrowed by `matrix`
/// drop(matrix);
/// # Ok::<(), hypre_rs::HypreError>(())
/// ```
///
/// Nor does finalizing MPI while hypre is initialized:
///
/// ```compile_fail,E0505
/// let universe = mpi::initialize().unwrap();
/// let hypre = hypre_rs::initialize(&universe)?;
/// drop(universe); // error: `universe` is still borrowed by `hypre`
/// hypre.finalize()?;
/// # Ok::<(), hypre_rs::HypreError>(())
/// ```
#[derive(Debug)]
pub struct Hypre<'u> {
    /// Borrows the universe; the raw pointer makes the token `!Send + !Sync`.
    _universe: PhantomData<(&'u Universe, *const ())>,
}

/// Initializes hypre (`HYPRE_Initialize`) and returns the token that hypre
/// objects borrow.
///
/// # Errors
///
/// [HypreError::AlreadyInitialized] if another [Hypre] token is alive, or the
/// error returned by hypre.
pub fn initialize(_universe: &Universe) -> HypreResult<Hypre<'_>> {
    if INITIALIZED.swap(true, Ordering::SeqCst) {
        return Err(HypreError::AlreadyInitialized);
    }
    let code = unsafe { HYPRE_Initialize() };
    if code != 0 {
        INITIALIZED.store(false, Ordering::SeqCst);
        return Err(HypreError::new(code));
    }
    Ok(Hypre {
        _universe: PhantomData,
    })
}

impl Hypre<'_> {
    /// Finalizes hypre (`HYPRE_Finalize`), reporting its error.
    ///
    /// Dropping the token also finalizes hypre, but ignores the error.
    pub fn finalize(self) -> HypreResult<()> {
        let code = Self::finalize_raw();
        std::mem::forget(self);
        match code {
            0 => Ok(()),
            code => Err(HypreError::new(code)),
        }
    }

    fn finalize_raw() -> hypre_sys::HYPRE_Int {
        let code = unsafe { HYPRE_Finalize() };
        INITIALIZED.store(false, Ordering::SeqCst);
        code
    }
}

impl Drop for Hypre<'_> {
    fn drop(&mut self) {
        crate::utils::discard_error(Self::finalize_raw());
    }
}
