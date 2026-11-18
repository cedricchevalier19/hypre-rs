//! Runtime symmetry check: `Matrix::is_symmetric` and
//! `TryFrom<Matrix> for SymmetricMatrix`, on any number of MPI ranks.

use hypre_rs::matrix::{CSRMatrix, IJMatrix};
use hypre_rs::solvers::{PCGSolverConfigBuilder, SymmetricLinearSolver, SymmetricSolver};
use hypre_rs::{HypreContext, HypreError, Matrix, SymmetricMatrix, Vector};
use mpi::topology::{Communicator, SimpleCommunicator};
use std::ops::Range;

/// Global number of rows; not a multiple of 2 or 3, so ranks get uneven rows.
const N: u32 = 10;

/// tridiag(-1, 2, -1) with both triangles given, plus `extra` entries. The
/// coupling between the first local row and the previous one is added by
/// this rank to a row owned by the previous rank.
fn laplacian<'a>(
    ctx: &'a HypreContext<'_>,
    comm: &'a SimpleCommunicator,
    rows: Range<u32>,
    extra: &[(u32, u32, f64)],
) -> Result<Matrix<'a>, HypreError> {
    let mut ij = ctx.ij_matrix(comm, rows.clone(), rows.clone())?;
    for i in rows.clone() {
        ij.add_elements([(i, i, 2.0)].into_iter())?;
        if i > 0 {
            ij.add_elements([(i, i - 1, -1.0), (i - 1, i, -1.0)].into_iter())?;
        }
    }
    // Extra entries are added once, by the rank owning their row.
    ij.add_elements(
        extra
            .iter()
            .copied()
            .filter(|(row, _, _)| rows.contains(row)),
    )?;
    Ok(Matrix::IJ(ij))
}

#[test]
fn symmetry_check() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let (rank, size) = (comm.rank() as u32, comm.size() as u32);
        let rows = (rank * N / size)..((rank + 1) * N / size);

        // Symmetric: converts, and CG solves it. A * [1, ..., 1] = [1, 0, ..., 0, 1].
        let mut a = SymmetricMatrix::try_from(laplacian(&ctx, &comm, rows.clone(), &[])?)?;
        let mut b = ctx.ij_vector(&comm, rows.clone())?;
        b.add_elements(
            rows.clone()
                .map(|i| (i, if i == 0 || i == N - 1 { 1.0 } else { 0.0 })),
        )?;
        let mut x = ctx.ij_vector(&comm, rows.clone())?;
        x.add_elements(rows.clone().map(|i| (i, 0.0)))?;
        let mut x = Vector::IJ(x);
        let config = PCGSolverConfigBuilder::default()
            .tol(1e-12)
            .max_iters(N as usize)
            .two_norm(true)
            .build()?;
        let solver = SymmetricSolver::CG(ctx.pcg_solver(&comm, config)?);
        let status = solver.solve(&mut a, &Vector::IJ(b), &mut x)?;
        assert!(status.converged, "{status}");
        let Vector::IJ(x) = &x;
        for (i, v) in rows.clone().zip(x.get_values(rows.clone())?) {
            assert!((v - 1.0).abs() < 1e-8, "x[{i}] = {v}, expected 1");
        }
        drop((a, x, solver));

        // The check also works on the CSR format.
        let mut ij = laplacian(&ctx, &comm, rows.clone(), &[])?;
        assert!(ij.is_symmetric(0.0)?);
        let Matrix::IJ(ij) = ij else { unreachable!() };
        let mut csr = Matrix::ParCSR(CSRMatrix::try_from(ij)?);
        assert!(csr.is_symmetric(0.0)?);
        drop(csr);

        // An entry without its transposed one, across ranks when size > 1.
        let mut m = laplacian(&ctx, &comm, rows.clone(), &[(N - 1, 0, 1.0)])?;
        assert!(!m.is_symmetric(0.0)?);
        // A missing transposed entry counts as 0: no tolerance below 1 hides it.
        assert!(!m.is_symmetric(0.5)?);
        assert!(matches!(
            SymmetricMatrix::try_from(m),
            Err(HypreError::NotSymmetric)
        ));

        // The unchecked conversion accepts it anyway, and gives it back.
        let m = laplacian(&ctx, &comm, rows.clone(), &[(N - 1, 0, 1.0)])?;
        let mut m = Matrix::from(SymmetricMatrix::from_matrix_unchecked(m));
        assert!(!m.is_symmetric(0.0)?);
        drop(m);

        // Both entries, but slightly different values: tolerance matters.
        let pair = [(N - 1, 0, 1.0), (0, N - 1, 1.0 + 1e-10)];
        let mut m = laplacian(&ctx, &comm, rows.clone(), &pair)?;
        assert!(!m.is_symmetric(0.0)?);
        assert!(m.is_symmetric(1e-8)?);
        drop(m);
        let m = laplacian(&ctx, &comm, rows.clone(), &pair)?;
        assert!(SymmetricMatrix::try_from_with_tolerance(m, 1e-8).is_ok());

        // An explicit zero without its transposed entry is still symmetric.
        let mut m = laplacian(&ctx, &comm, rows.clone(), &[(N - 1, 0, 0.0)])?;
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
        let cols = (2 * rank * N / size)..(2 * (rank + 1) * N / size);
        let mut ij = IJMatrix::new(&ctx, &comm, rows.clone(), cols)?;
        ij.add_elements(rows.clone().map(|i| (i, i, 1.0)))?;
        assert!(!Matrix::IJ(ij).is_symmetric(0.0)?);

        // Only rank 0 owns rows: the others have nothing to send.
        let first_only = if rank == 0 { 0..4 } else { 4..4 };
        let m = laplacian(&ctx, &comm, first_only, &[])?;
        assert!(SymmetricMatrix::try_from(m).is_ok());
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
