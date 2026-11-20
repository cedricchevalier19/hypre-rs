//! The [HypreContext]: hypre initialization, finalization and object creation.

use crate::matrix::{DistributedMatrixBuilder, LocalMatrixBuilder, SymmetricMatrixBuilder};
use crate::preconditioners::{
    AMGPreconditioner, AMGPreconditionerConfig, SymmetricAMGPreconditioner,
    SymmetricAMGPreconditionerConfig,
};
use crate::solvers::{
    AMGSolver, AMGSolverConfig, BiCGSTABSolver, BiCGSTABSolverConfig, COGMRESSolver,
    COGMRESSolverConfig, FlexGMRESSolver, FlexGMRESSolverConfig, GMRESSolver, GMRESSolverConfig,
    LGMRESSolver, LGMRESSolverConfig, PCGSolver, PCGSolverConfig,
};
use crate::vector::{DistributedVectorBuilder, LocalVectorBuilder};
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
/// ([HypreContext::local_matrix_builder],
/// [HypreContext::distributed_matrix_builder],
/// [HypreContext::symmetric_matrix_builder],
/// [HypreContext::local_vector_builder],
/// [HypreContext::distributed_vector_builder],
/// [HypreContext::pcg_solver], [HypreContext::gmres_solver] and the other
/// solvers) or with their `new`
/// constructor.
///
/// Only one context can exist at a time. It is neither `Send` nor `Sync`:
/// hypre objects are created and used on the thread that initialized hypre.
///
/// # Example
///
/// ```
/// use hypre_rs::matrix::DistributedMatrixBuilder;
///
/// let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// let comm = universe.world();
/// let ctx = hypre_rs::initialize(&universe)?;
/// let rows = hypre_rs::distribute_rows(&comm, 10);
///
/// let matrix = ctx.distributed_matrix_builder(&comm, rows.clone(), rows.clone())?;
/// // Same as the factory method above.
/// let other = DistributedMatrixBuilder::new(&ctx, &comm, rows.clone(), rows)?;
/// drop((matrix, other));
/// ctx.finalize()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Finalizing hypre while an object still uses it does not compile:
///
/// ```compile_fail,E0505
/// # use hypre_rs::matrix::DistributedMatrixBuilder;
/// let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// let comm = universe.world();
/// let ctx = hypre_rs::initialize(&universe)?;
/// let matrix = DistributedMatrixBuilder::new(&ctx, &comm, 0..0, 0..0)?;
/// ctx.finalize()?; // error: `ctx` is still borrowed by `matrix`
/// drop(matrix);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Nor does freeing a communicator still used by a hypre object:
///
/// ```compile_fail,E0505
/// let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// let ctx = hypre_rs::initialize(&universe)?;
/// let comm = universe.world().duplicate();
/// let vector = ctx.distributed_vector_builder(&comm, 0..0)?;
/// drop(comm); // error: `comm` is still borrowed by `vector`
/// drop(vector);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Nor does finalizing MPI while hypre is initialized:
///
/// ```compile_fail,E0505
/// let universe = mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
/// let ctx = hypre_rs::initialize(&universe)?;
/// drop(universe); // error: `universe` is still borrowed by `ctx`
/// ctx.finalize()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
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

    /// Creates a [LocalMatrixBuilder]; see [LocalMatrixBuilder::new].
    pub fn local_matrix_builder<'a>(
        &'a self,
        comm: &'a impl Communicator,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> HypreResult<LocalMatrixBuilder<'a>> {
        LocalMatrixBuilder::new(self, comm, rows, cols)
    }

    /// Creates a [DistributedMatrixBuilder]; see
    /// [DistributedMatrixBuilder::new].
    pub fn distributed_matrix_builder<'a>(
        &'a self,
        comm: &'a impl Communicator,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> HypreResult<DistributedMatrixBuilder<'a>> {
        DistributedMatrixBuilder::new(self, comm, rows, cols)
    }

    /// Creates a [SymmetricMatrixBuilder]; see [SymmetricMatrixBuilder::new].
    pub fn symmetric_matrix_builder<'a>(
        &'a self,
        comm: &'a impl Communicator,
        rows: Range<u32>,
    ) -> HypreResult<SymmetricMatrixBuilder<'a>> {
        SymmetricMatrixBuilder::new(self, comm, rows)
    }

    /// Creates a [LocalVectorBuilder]; see [LocalVectorBuilder::new].
    pub fn local_vector_builder<'a>(
        &'a self,
        comm: &'a impl Communicator,
        rows: Range<u32>,
    ) -> HypreResult<LocalVectorBuilder<'a>> {
        LocalVectorBuilder::new(self, comm, rows)
    }

    /// Creates a [DistributedVectorBuilder]; see
    /// [DistributedVectorBuilder::new].
    pub fn distributed_vector_builder<'a>(
        &'a self,
        comm: &'a impl Communicator,
        rows: Range<u32>,
    ) -> HypreResult<DistributedVectorBuilder<'a>> {
        DistributedVectorBuilder::new(self, comm, rows)
    }

    /// Creates a [PCGSolver]; see [PCGSolver::new].
    pub fn pcg_solver<'a>(
        &'a self,
        comm: &'a impl Communicator,
        config: PCGSolverConfig,
    ) -> HypreResult<PCGSolver<'a>> {
        PCGSolver::new(self, comm, config)
    }

    /// Creates a [GMRESSolver]; see [GMRESSolver::new].
    pub fn gmres_solver<'a>(
        &'a self,
        comm: &'a impl Communicator,
        config: GMRESSolverConfig,
    ) -> HypreResult<GMRESSolver<'a>> {
        GMRESSolver::new(self, comm, config)
    }

    /// Creates a [FlexGMRESSolver]; see [FlexGMRESSolver::new].
    pub fn flex_gmres_solver<'a>(
        &'a self,
        comm: &'a impl Communicator,
        config: FlexGMRESSolverConfig,
    ) -> HypreResult<FlexGMRESSolver<'a>> {
        FlexGMRESSolver::new(self, comm, config)
    }

    /// Creates a [LGMRESSolver]; see [LGMRESSolver::new].
    pub fn lgmres_solver<'a>(
        &'a self,
        comm: &'a impl Communicator,
        config: LGMRESSolverConfig,
    ) -> HypreResult<LGMRESSolver<'a>> {
        LGMRESSolver::new(self, comm, config)
    }

    /// Creates a [COGMRESSolver]; see [COGMRESSolver::new].
    pub fn cogmres_solver<'a>(
        &'a self,
        comm: &'a impl Communicator,
        config: COGMRESSolverConfig,
    ) -> HypreResult<COGMRESSolver<'a>> {
        COGMRESSolver::new(self, comm, config)
    }

    /// Creates a [BiCGSTABSolver]; see [BiCGSTABSolver::new].
    pub fn bicgstab_solver<'a>(
        &'a self,
        comm: &'a impl Communicator,
        config: BiCGSTABSolverConfig,
    ) -> HypreResult<BiCGSTABSolver<'a>> {
        BiCGSTABSolver::new(self, comm, config)
    }

    /// Creates an [AMGSolver]; see [AMGSolver::new].
    pub fn amg_solver<'a>(
        &'a self,
        comm: &'a impl Communicator,
        config: AMGSolverConfig,
    ) -> HypreResult<AMGSolver<'a>> {
        AMGSolver::new(self, comm, config)
    }

    /// Creates a [SymmetricAMGPreconditioner]; see
    /// [SymmetricAMGPreconditioner::new].
    pub fn symmetric_amg_preconditioner<'a>(
        &'a self,
        comm: &'a impl Communicator,
        config: SymmetricAMGPreconditionerConfig,
    ) -> HypreResult<SymmetricAMGPreconditioner<'a>> {
        SymmetricAMGPreconditioner::new(self, comm, config)
    }

    /// Creates an [AMGPreconditioner]; see [AMGPreconditioner::new].
    pub fn amg_preconditioner<'a>(
        &'a self,
        comm: &'a impl Communicator,
        config: AMGPreconditionerConfig,
    ) -> HypreResult<AMGPreconditioner<'a>> {
        AMGPreconditioner::new(self, comm, config)
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
