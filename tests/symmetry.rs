//! Runtime symmetry check: `Matrix::is_symmetric` and
//! `TryFrom<CSRMatrix> for SymmetricMatrix`, on any number of MPI ranks.

use hypre_rs::matrix::{CSRMatrix, LocalMatrixBuilder};
use hypre_rs::solvers::PCGSolverConfigBuilder;
use hypre_rs::{HypreContext, HypreError, Matrix, ParVector, SymmetricMatrix};
use mpi::topology::{Communicator, SimpleCommunicator};
use std::ops::Range;

/// Global number of rows; not a multiple of 2 or 3, so ranks get uneven rows.
const N: u32 = 10;

/// tridiag(-1, 2, -1) with both triangles given, plus `extra` entries,
/// assembled. The
/// coupling between the first local row and the previous one is added by
/// this rank to a row owned by the previous rank.
fn laplacian<'a>(
    ctx: &'a HypreContext<'_>,
    comm: &'a SimpleCommunicator,
    rows: Range<u32>,
    extra: &[(u32, u32, f64)],
) -> Result<CSRMatrix<'a>, HypreError> {
    // Row i - 1 may belong to the previous rank: a distributed builder.
    let mut ij = ctx.distributed_matrix_builder(comm, rows.clone(), rows.clone())?;
    for i in rows.clone() {
        ij.add([(i, i, 2.0)])?;
        if i > 0 {
            ij.add([(i, i - 1, -1.0), (i - 1, i, -1.0)])?;
        }
    }
    // Extra entries are added once, by the rank owning their row.
    ij.add(
        extra
            .iter()
            .copied()
            .filter(|(row, _, _)| rows.contains(row)),
    )?;
    ij.assemble()
}

#[test]
fn symmetry_check() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let rank = comm.rank();
        let rows = hypre_rs::distribute_rows(&comm, N);

        // Symmetric: converts, and CG solves it. A * [1, ..., 1] = [1, 0, ..., 0, 1].
        let a = SymmetricMatrix::try_from(laplacian(&ctx, &comm, rows.clone(), &[])?)?;
        let mut b = ctx.local_vector_builder(&comm, rows.clone())?;
        b.add(
            rows.clone()
                .map(|i| (i, if i == 0 || i == N - 1 { 1.0 } else { 0.0 })),
        )?;
        let b = b.assemble()?;
        let mut x = ParVector::zeros(&ctx, &comm, rows.clone())?;
        let config = PCGSolverConfigBuilder::default()
            .tol(1e-12)
            .max_iters(N as usize)
            .two_norm(true)
            .build()?;
        let solver = ctx.pcg_solver(&comm, config)?;
        let status = solver.solve(&a, &b, &mut x)?;
        assert!(status.converged, "{status}");
        for (i, v) in rows.clone().zip(x.local_values()?) {
            assert!((v - 1.0).abs() < 1e-8, "x[{i}] = {v}, expected 1");
        }
        drop((a, x, solver));

        // A diagonal matrix built by a local builder.
        let mut ij = ctx.local_matrix_builder(&comm, rows.clone(), rows.clone())?;
        ij.add(rows.clone().map(|i| (i, i, 2.0)))?;
        let csr: CSRMatrix<'_> = ij.assemble()?;
        assert!(csr.is_symmetric(0.0)?);
        drop(csr);

        // An entry without its transposed one, across ranks when size > 1.
        let m = laplacian(&ctx, &comm, rows.clone(), &[(N - 1, 0, 1.0)])?;
        assert!(!m.is_symmetric(0.0)?);
        // A missing transposed entry counts as 0: no tolerance below 1 hides it.
        assert!(!m.is_symmetric(0.5)?);
        assert!(matches!(
            SymmetricMatrix::try_from(m),
            Err(HypreError::NotSymmetric)
        ));

        // The unchecked conversion accepts it anyway, and gives it back.
        let m = laplacian(&ctx, &comm, rows.clone(), &[(N - 1, 0, 1.0)])?;
        let m = SymmetricMatrix::from_matrix_unchecked(m).into_inner();
        assert!(!m.is_symmetric(0.0)?);
        drop(m);

        // Both entries, but slightly different values: tolerance matters.
        let pair = [(N - 1, 0, 1.0), (0, N - 1, 1.0 + 1e-10)];
        let m = laplacian(&ctx, &comm, rows.clone(), &pair)?;
        assert!(!m.is_symmetric(0.0)?);
        assert!(m.is_symmetric(1e-8)?);
        drop(m);
        let m = laplacian(&ctx, &comm, rows.clone(), &pair)?;
        assert!(SymmetricMatrix::try_from_with_tolerance(m, 1e-8).is_ok());

        // An explicit zero without its transposed entry is still symmetric.
        let m = laplacian(&ctx, &comm, rows.clone(), &[(N - 1, 0, 0.0)])?;
        assert!(m.is_symmetric(0.0)?);

        // Invalid tolerances.
        for tol in [-1.0, f64::NAN] {
            assert!(matches!(
                m.is_symmetric(tol),
                Err(HypreError::InvalidParameterPositive)
            ));
        }
        drop(m);

        // Not square: N rows, 2 * N columns.
        let cols = hypre_rs::distribute_rows(&comm, 2 * N);
        let mut ij = LocalMatrixBuilder::new(&ctx, &comm, rows.clone(), cols)?;
        ij.add(rows.clone().map(|i| (i, i, 1.0)))?;
        assert!(!ij.assemble()?.is_symmetric(0.0)?);

        // A matrix built from its lower triangle is exactly symmetric. Once
        // assembled, it is a Matrix, also as a trait object.
        let mut s = ctx.symmetric_matrix_builder(&comm, rows.clone())?;
        s.add(rows.clone().map(|i| (i, i, 2.0)))?;
        s.add(rows.clone().filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
        s.add(rows.clone().filter(|&i| i >= 3).map(|i| (i, i - 3, 0.25)))?;
        let s = s.assemble()?;
        assert!(s.is_symmetric(0.0)?);
        let as_matrix: &dyn Matrix = &s;
        assert!(as_matrix.is_symmetric(0.0)?);
        drop(s);

        // Only rank 0 owns rows: the others have nothing to send.
        let first_only = if rank == 0 { 0..4 } else { 4..4 };
        let m = laplacian(&ctx, &comm, first_only, &[])?;
        assert!(SymmetricMatrix::try_from(m).is_ok());
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
