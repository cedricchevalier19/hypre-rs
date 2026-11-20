//! Creating hypre objects with the builders, including empty and invalid
//! ranges, entries for other ranks' rows and values that do not fit.

use hypre_rs::matrix::{CSRMatrix, DistributedMatrixBuilder, LocalMatrixBuilder};
use hypre_rs::vector::{DistributedVectorBuilder, LocalVectorBuilder};
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
        let (rank, size) = (comm.rank(), comm.size());
        let rows = hypre_rs::distribute_rows(&comm, 10);

        // A local vector: `add` adds, `set` replaces, the last value wins.
        let mut v = LocalVectorBuilder::new(&ctx, &comm, rows.clone())?;
        assert_eq!(v.local_range(), rows);
        v.add(rows.clone().map(|i| (i, f64::from(i))))?;
        v.add(rows.clone().map(|i| (i, 0.5)))?;
        v.set(rows.clone().filter(|i| i % 2 == 0).map(|i| (i, -1.0)))?;
        v.set(rows.clone().filter(|i| i % 4 == 0).map(|i| (i, -2.0)))?;
        let v = v.assemble()?;
        let expected = |i: u32| match (i % 4, i % 2) {
            (0, _) => -2.0,
            (_, 0) => -1.0,
            _ => f64::from(i) + 0.5,
        };
        assert!(rows
            .clone()
            .zip(v.local_values()?)
            .all(|(i, x)| x == expected(i)));
        assert_eq!(v.get_values(rows.start..rows.start)?, Vec::<f64>::new());
        drop(v);

        // Local builders reject rows of other ranks, and change nothing.
        let other = if rows.end < 10 { rows.end } else { 0 };
        if !rows.contains(&other) {
            let mut v = ctx.local_vector_builder(&comm, rows.clone())?;
            assert!(matches!(
                v.add([(rows.start, 1.0), (other, 1.0)]),
                Err(HypreError::NotLocalRow(row)) if u32::try_from(row) == Ok(other)
            ));
            let mut m = ctx.local_matrix_builder(&comm, rows.clone(), rows.clone())?;
            assert!(matches!(
                m.set([(other, other, 1.0)]),
                Err(HypreError::NotLocalRow(_))
            ));
            drop((v, m));
        }
        // Every rank must assemble together (assembly is collective).
        let v = ctx.local_vector_builder(&comm, rows.clone())?.assemble()?;
        assert!(v.local_values()?.iter().all(|&x| x == 0.0));
        drop(v);

        // A distributed vector: every rank adds 1 to every row.
        let mut d = DistributedVectorBuilder::new(&ctx, &comm, rows.clone())?;
        assert_eq!(d.local_range(), rows);
        d.add((0..10u32).map(|i| (i, 1.0)))?;
        let d = d.assemble()?;
        assert!(d.local_values()?.iter().all(|&x| x == f64::from(size)));
        drop(d);

        // Only rank 0 owns rows: the others have empty ranges.
        let first_only = if rank == 0 { 0..4 } else { 4..4 };
        let mut w = ctx.local_vector_builder(&comm, first_only.clone())?;
        w.add(first_only.clone().map(|i| (i, 1.0)))?;
        let w = w.assemble()?;
        assert_eq!(w.local_values()?.len(), first_only.len());
        let mut m = ctx.local_matrix_builder(&comm, first_only.clone(), first_only.clone())?;
        m.add(first_only.clone().map(|i| (i, i, 1.0)))?;
        let m = m.assemble()?;
        drop((w, m));

        // A range whose start is after its end is rejected before calling hypre.
        #[allow(clippy::reversed_empty_ranges)]
        let reversed = 5..3;
        assert!(matches!(
            LocalVectorBuilder::new(&ctx, &comm, reversed.clone()),
            Err(HypreError::InvalidRange)
        ));
        assert!(matches!(
            DistributedVectorBuilder::new(&ctx, &comm, reversed.clone()),
            Err(HypreError::InvalidRange)
        ));
        assert!(matches!(
            LocalMatrixBuilder::new(&ctx, &comm, reversed.clone(), 0..1),
            Err(HypreError::InvalidRange)
        ));
        assert!(matches!(
            DistributedMatrixBuilder::new(&ctx, &comm, 0..1, reversed.clone()),
            Err(HypreError::InvalidRange)
        ));
        assert!(matches!(
            ctx.symmetric_matrix_builder(&comm, reversed.clone()),
            Err(HypreError::InvalidRange)
        ));

        // An index that does not fit in HYPRE_BigInt is an error, and nothing
        // is added.
        let mut v = ctx.local_vector_builder(&comm, rows.clone())?;
        assert!(matches!(
            v.add([(u64::from(rows.start), 1.0), (u64::MAX, 1.0)]),
            Err(HypreError::ConversionError(_))
        ));
        let v = v.assemble()?;
        assert!(v.local_values()?.iter().all(|&x| x == 0.0));
        assert!(matches!(
            v.get_values(reversed),
            Err(HypreError::InvalidRange)
        ));
        drop(v);

        // Symmetric matrices take lower-triangle entries only; a rejected
        // call adds nothing.
        let mut s = ctx.symmetric_matrix_builder(&comm, rows.clone())?;
        assert_eq!(s.local_range(), rows);
        s.add(rows.clone().map(|i| (i, i, 1.0)))?;
        assert!(matches!(
            s.add([(1u32, 0u32, 1.0), (0, 1, 1.0)]),
            Err(HypreError::NotLowerTriangular)
        ));
        assert!(matches!(
            s.add([(0u64, 0u64, 1.0), (u64::MAX, 0, 1.0)]),
            Err(HypreError::ConversionError(_))
        ));
        drop(s.assemble()?);

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
