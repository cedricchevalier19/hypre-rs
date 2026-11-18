//! Creating hypre objects: vectors, IJ and ParCSR matrices, including empty
//! and invalid ranges.

use hypre_rs::matrix::{CSRMatrix, IJMatrix};
use hypre_rs::vector::IJVector;
use hypre_rs::HypreError;
use mpi::topology::Communicator;

#[test]
fn create_objects() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let (rank, size) = (comm.rank() as u32, comm.size() as u32);
        let rows = (rank * 10 / size)..((rank + 1) * 10 / size);

        // add_elements adds to the current values; reading needs no assembly.
        let mut v = IJVector::new(&ctx, &comm, rows.clone())?;
        v.add_elements(rows.clone().map(|i| (i, f64::from(i))))?;
        v.add_elements(rows.clone().map(|i| (i, 0.5)))?;
        let expected: Vec<f64> = rows.clone().map(|i| f64::from(i) + 0.5).collect();
        assert_eq!(v.get_values(rows.clone())?, expected);
        assert_eq!(v.get_values(rows.start..rows.start)?, Vec::<f64>::new());

        // Only rank 0 owns rows: the others have empty ranges.
        let first_only = if rank == 0 { 0..4 } else { 4..4 };
        let mut w = ctx.ij_vector(&comm, first_only.clone())?;
        w.add_elements(first_only.clone().map(|i| (i, 1.0)))?;
        assert_eq!(w.get_values(first_only.clone())?.len(), first_only.len());
        let m = ctx.ij_matrix(&comm, first_only.clone(), first_only)?;
        drop((v, w, m));

        // A range whose start is after its end is rejected before calling hypre.
        #[allow(clippy::reversed_empty_ranges)]
        let reversed = 5..3;
        assert!(matches!(
            IJVector::new(&ctx, &comm, reversed.clone()),
            Err(HypreError::InvalidRange)
        ));
        assert!(matches!(
            IJMatrix::new(&ctx, &comm, reversed.clone(), 0..1),
            Err(HypreError::InvalidRange)
        ));
        assert!(matches!(
            IJMatrix::new(&ctx, &comm, 0..1, reversed.clone()),
            Err(HypreError::InvalidRange)
        ));
        assert!(matches!(
            ctx.symmetric_matrix(&comm, reversed.clone()),
            Err(HypreError::InvalidRange)
        ));
        let mut v = ctx.ij_vector(&comm, rows.clone())?;
        assert!(matches!(
            v.get_values(reversed),
            Err(HypreError::InvalidRange)
        ));

        // An index that does not fit in HYPRE_BigInt is an error, and nothing
        // is added (it used to be replaced by 0).
        assert!(matches!(
            v.add_elements([(0, 1.0), (u64::MAX, 1.0)].into_iter()),
            Err(HypreError::ConversionError(_))
        ));
        assert!(v.get_values(rows.clone())?.iter().all(|&x| x == 0.0));
        drop(v);

        // Symmetric matrices take lower-triangle entries only; a rejected
        // call adds nothing.
        let mut s = ctx.symmetric_matrix(&comm, rows.clone())?;
        s.add_elements(rows.clone().map(|i| (i, i, 1.0)))?;
        assert!(matches!(
            s.add_elements([(1u32, 0u32, 1.0), (0, 1, 1.0)].into_iter()),
            Err(HypreError::NotLowerTriangular)
        ));
        assert!(matches!(
            s.add_elements([(0u64, 0u64, 1.0), (u64::MAX, 0, 1.0)].into_iter()),
            Err(HypreError::ConversionError(_))
        ));
        drop(s);

        // An empty ParCSR matrix, partitioned like `rows`.
        let starts = [rows.start as usize, rows.end as usize];
        let csr = CSRMatrix::new(&ctx, &comm, 10, 10, &starts, &starts)?;
        drop(csr);
        // A size that does not fit in HYPRE_BigInt.
        assert!(matches!(
            ctx.csr_matrix(&comm, usize::MAX, 10, &starts, &starts),
            Err(HypreError::ConversionError(_))
        ));
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
