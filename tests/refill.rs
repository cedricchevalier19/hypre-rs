//! Refill builders: new values for assembled matrices and vectors with the
//! same pattern, checked through solves whose exact solution is known, on
//! any number of MPI ranks.

use hypre_rs::prelude::*;
use mpi::topology::{Communicator, SimpleCommunicator};
use std::ops::Range;

/// Global number of rows; not a multiple of 2 or 3, so ranks get uneven rows.
const N: u32 = 50;

/// Entries of row `i` of `scale * tridiag(-1.5, 2, -0.5)`, not symmetric.
fn row(i: u32, scale: f64) -> Vec<(u32, u32, f64)> {
    let mut entries = vec![(i, i, 2.0 * scale)];
    if i > 0 {
        entries.push((i, i - 1, -1.5 * scale));
    }
    if i + 1 < N {
        entries.push((i, i + 1, -0.5 * scale));
    }
    entries
}

/// Solves `a * x = b` with GMRES and checks that `x` is `expected(i)`.
fn check_solve(
    ctx: &HypreContext<'_>,
    comm: &SimpleCommunicator,
    rows: Range<u32>,
    a: &impl Matrix,
    b: &ParVector<'_>,
    expected: impl Fn(u32) -> f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = GMRESSolverConfigBuilder::default()
        .tol(1e-12)
        .k_dim(N as usize)
        .max_iters(10 * N as usize)
        .build()?;
    let mut x = ParVector::zeros(ctx, comm, rows.clone())?;
    let status = ctx.gmres_solver(comm, config)?.solve(a, b, &mut x)?;
    assert!(status.converged, "{status}");
    for (i, v) in rows.zip(x.local_values()?) {
        assert!(
            (v - expected(i)).abs() < 1e-8,
            "x[{i}] = {v}, expected {}",
            expected(i)
        );
    }
    Ok(())
}

#[test]
fn refill() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let rows = distribute_rows(&comm, N);
        let row_sum = |i: u32| row(i, 1.0).iter().map(|e| e.2).sum::<f64>();

        let mut a = ctx.local_matrix_builder(&comm, rows.clone(), rows.clone())?;
        for i in rows.clone() {
            a.add(row(i, 1.0))?;
        }
        let a = a.assemble()?;
        assert_eq!(a.local_range(), rows);
        let b = ParVector::from_fn(&ctx, &comm, rows.clone(), row_sum)?;
        check_solve(&ctx, &comm, rows.clone(), &a, &b, |_| 1.0)?;

        // Refill from zeros with twice the values: x = 1 / 2.
        let mut a = a.refill()?;
        assert_eq!(a.local_range(), rows);
        for i in rows.clone() {
            a.add(row(i, 2.0))?;
        }
        let a = a.assemble()?;
        check_solve(&ctx, &comm, rows.clone(), &a, &b, |_| 0.5)?;

        // Update on top of the current values: 2 A + 2 A = 4 A, x = 1 / 4.
        // The owner of row i adds the super-diagonal entry of row i and the
        // sub-diagonal entry of row i + 1, which may be another rank's; the
        // diagonal is set to 4 * 2 in local rows.
        let mut a = a.update();
        for i in rows.clone().filter(|&i| i + 1 < N) {
            a.add([(i, i + 1, -0.5 * 2.0), (i + 1, i, -1.5 * 2.0)])?;
        }
        a.set(rows.clone().map(|i| (i, i, 8.0)))?;
        let a = a.assemble()?;
        check_solve(&ctx, &comm, rows.clone(), &a, &b, |_| 0.25)?;

        // A new entry outside the pattern is rejected by hypre; the matrix
        // is dropped with the builder.
        let mut a = a.update();
        if let Some(i) = rows.clone().find(|&i| i + 3 < N) {
            assert!(a.add([(i, i + 3, 1.0)]).is_err());
        }
        // Setting another rank's row is rejected before reaching hypre.
        let other = if rows.end < N { rows.end } else { 0 };
        if !rows.contains(&other) {
            assert!(matches!(
                a.set([(other, other, 1.0)]),
                Err(HypreError::NotLocalRow(_))
            ));
        }
        drop(a);

        // Vectors: refill from zeros, set local values, then add on top
        // from every rank (off-rank rows included).
        let mut b = b.refill()?;
        assert_eq!(b.local_range(), rows);
        b.set(rows.clone().map(|i| (i, f64::from(i))))?;
        if !rows.contains(&other) {
            assert!(matches!(
                b.set([(other, 1.0)]),
                Err(HypreError::NotLocalRow(_))
            ));
        }
        let b = b.assemble()?;
        assert!(rows
            .clone()
            .zip(b.local_values()?)
            .all(|(i, v)| v == f64::from(i)));
        let size = f64::from(comm.size());
        let mut b = b.update();
        b.add((0..N).map(|i| (i, 1.0 / size)))?;
        let b = b.assemble()?;
        for (i, v) in rows.clone().zip(b.local_values()?) {
            assert!((v - (f64::from(i) + 1.0)).abs() < 1e-12, "b[{i}] = {v}");
        }
        drop(b);

        // Symmetric matrices: refill keeps them symmetric, and CG solves.
        let lower = |i: u32, scale: f64| {
            let mut entries = vec![(i, i, 2.0 * scale)];
            if i > 0 {
                entries.push((i, i - 1, -scale));
            }
            entries
        };
        let mut s = ctx.symmetric_matrix_builder(&comm, rows.clone())?;
        for i in rows.clone() {
            s.add(lower(i, 1.0))?;
        }
        let s = s.assemble()?;
        let mut s = s.refill()?;
        assert_eq!(s.local_range(), rows);
        for i in rows.clone() {
            s.add(lower(i, 3.0))?;
        }
        let s = s.assemble()?;
        assert!(s.is_symmetric(0.0)?);
        // 3 tridiag(-1, 2, -1) * [1, ..., 1] = [3, 0, ..., 0, 3].
        let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |i| {
            if i == 0 || i == N - 1 {
                3.0
            } else {
                0.0
            }
        })?;
        let mut x = ParVector::zeros(&ctx, &comm, rows.clone())?;
        let config = PCGSolverConfigBuilder::default().tol(1e-12).build()?;
        let status = ctx.pcg_solver(&comm, config)?.solve(&s, &b, &mut x)?;
        assert!(status.converged, "{status}");
        assert!(x.local_values()?.iter().all(|&v| (v - 1.0).abs() < 1e-8));
        // `update` keeps the values: adding the same again doubles them.
        let mut s = s.update();
        for i in rows.clone() {
            s.add(lower(i, 3.0))?;
        }
        let s = s.assemble()?;
        let mut x = x.refill()?.assemble()?;
        let status = ctx.pcg_solver(&comm, config)?.solve(&s, &b, &mut x)?;
        assert!(status.converged, "{status}");
        assert!(x.local_values()?.iter().all(|&v| (v - 0.5).abs() < 1e-8));
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
