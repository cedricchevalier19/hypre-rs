//! The [HypreContext]: hypre initialization, finalization and object creation.

use crate::matrix::{CSRMatrix, IJMatrix};
use crate::solvers::{PCGSolver, PCGSolverConfig};
use crate::vector::IJVector;
use crate::{HypreError, HypreResult};
use hypre_sys::{HYPRE_Finalize, HYPRE_Initialize};
use mpi::environment::Universe;
use mpi::topology::Communicator;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

/// Whether a [HypreContext] is alive (hypre is initialized).
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// A hypre context: proof that hypre is initialized, borrowing the MPI
/// [Universe].
///
/// Every hypre object borrows this context and its MPI communicator, so the
/// borrow checker guarantees that:
///
/// - all hypre objects are dropped before hypre is finalized,
/// - hypre is finalized before MPI (dropping the `Universe` calls
///   `MPI_Finalize`), and
/// - a communicator outlives the hypre objects that use it.
///
/// Objects are created either with the factory methods of the context
/// ([HypreContext::ij_matrix], [HypreContext::ij_vector],
/// [HypreContext::csr_matrix], [HypreContext::pcg_solver]) or with their `new`
/// constructor.
///
/// Only one context can exist at a time. It is neither `Send` nor `Sync`:
/// hypre objects are created and used on the thread that initialized hypre.
///
/// # Example
///
/// ```
/// # use mpi::topology::Communicator;
/// use hypre_rs::matrix::IJMatrix;
///
/// let universe = mpi::initialize().unwrap();
/// let comm = universe.world();
/// let ctx = hypre_rs::initialize(&universe)?;
/// let (rank, size) = (comm.rank() as u32, comm.size() as u32);
/// let rows = (rank * 10 / size)..((rank + 1) * 10 / size);
///
/// let matrix = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
/// // Same as the factory method above.
/// let other = IJMatrix::new(&ctx, &comm, rows.clone(), rows)?;
/// drop((matrix, other));
/// ctx.finalize()?;
/// # Ok::<(), hypre_rs::HypreError>(())
/// ```
///
/// Finalizing hypre while an object still uses it does not compile:
///
/// ```compile_fail,E0505
/// # use mpi::topology::Communicator;
/// # use hypre_rs::matrix::IJMatrix;
/// let universe = mpi::initialize().unwrap();
/// let comm = universe.world();
/// let ctx = hypre_rs::initialize(&universe)?;
/// let matrix = IJMatrix::new(&ctx, &comm, 0..0, 0..0)?;
/// ctx.finalize()?; // error: `ctx` is still borrowed by `matrix`
/// drop(matrix);
/// # Ok::<(), hypre_rs::HypreError>(())
/// ```
///
/// Nor does freeing a communicator still used by a hypre object:
///
/// ```compile_fail,E0505
/// # use mpi::topology::Communicator;
/// let universe = mpi::initialize().unwrap();
/// let ctx = hypre_rs::initialize(&universe)?;
/// let comm = universe.world().duplicate();
/// let vector = ctx.ij_vector(&comm, 0..0)?;
/// drop(comm); // error: `comm` is still borrowed by `vector`
/// drop(vector);
/// # Ok::<(), hypre_rs::HypreError>(())
/// ```
///
/// Nor does finalizing MPI while hypre is initialized:
///
/// ```compile_fail,E0505
/// let universe = mpi::initialize().unwrap();
/// let ctx = hypre_rs::initialize(&universe)?;
/// drop(universe); // error: `universe` is still borrowed by `ctx`
/// ctx.finalize()?;
/// # Ok::<(), hypre_rs::HypreError>(())
/// ```
#[derive(Debug)]
pub struct HypreContext<'u> {
    /// Borrows the universe; the raw pointer makes the context `!Send + !Sync`.
    _universe: PhantomData<(&'u Universe, *const ())>,
}

/// Initializes hypre (`HYPRE_Initialize`) and returns the context that hypre
/// objects borrow.
///
/// # Errors
///
/// [HypreError::AlreadyInitialized] if another [HypreContext] is alive, or the
/// error returned by hypre.
pub fn initialize(_universe: &Universe) -> HypreResult<HypreContext<'_>> {
    if INITIALIZED.swap(true, Ordering::SeqCst) {
        return Err(HypreError::AlreadyInitialized);
    }
    let code = unsafe { HYPRE_Initialize() };
    if code != 0 {
        INITIALIZED.store(false, Ordering::SeqCst);
        return Err(HypreError::new(code));
    }
    Ok(HypreContext {
        _universe: PhantomData,
    })
}

impl HypreContext<'_> {
    /// Finalizes hypre (`HYPRE_Finalize`), reporting its error.
    ///
    /// Dropping the context also finalizes hypre, but ignores the error.
    pub fn finalize(self) -> HypreResult<()> {
        let code = Self::finalize_raw();
        std::mem::forget(self);
        match code {
            0 => Ok(()),
            code => Err(HypreError::new(code)),
        }
    }

    /// Creates an [IJMatrix]; see [IJMatrix::new].
    pub fn ij_matrix<'a>(
        &'a self,
        comm: &'a impl Communicator,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> HypreResult<IJMatrix<'a>> {
        IJMatrix::new(self, comm, rows, cols)
    }

    /// Creates a [CSRMatrix]; see [CSRMatrix::new].
    pub fn csr_matrix<'a>(
        &'a self,
        comm: &'a impl Communicator,
        global_num_rows: usize,
        global_num_cols: usize,
        row_starts: &[usize],
        col_starts: &[usize],
    ) -> HypreResult<CSRMatrix<'a>> {
        CSRMatrix::new(
            self,
            comm,
            global_num_rows,
            global_num_cols,
            row_starts,
            col_starts,
        )
    }

    /// Creates an [IJVector]; see [IJVector::new].
    pub fn ij_vector<'a>(
        &'a self,
        comm: &'a impl Communicator,
        rows: Range<u32>,
    ) -> HypreResult<IJVector<'a>> {
        IJVector::new(self, comm, rows)
    }

    /// Creates a [PCGSolver]; see [PCGSolver::new].
    pub fn pcg_solver<'a>(
        &'a self,
        comm: &'a impl Communicator,
        config: PCGSolverConfig,
    ) -> HypreResult<PCGSolver<'a>> {
        PCGSolver::new(self, comm, config)
    }

    fn finalize_raw() -> hypre_sys::HYPRE_Int {
        let code = unsafe { HYPRE_Finalize() };
        INITIALIZED.store(false, Ordering::SeqCst);
        code
    }
}

impl Drop for HypreContext<'_> {
    fn drop(&mut self) {
        crate::utils::discard_error(Self::finalize_raw());
    }
}
