//! Matrices built by the local and distributed builders hold the right
//! values, checked through solves whose exact solution is known, on any
//! number of MPI ranks.

use hypre_rs::prelude::*;
use mpi::topology::SimpleCommunicator;
use std::ops::Range;

/// Global number of rows; not a multiple of 2 or 3, so ranks get uneven rows.
const N: u32 = 50;

/// Solves `a * x = b` with GMRES from `x = 0` and returns this rank's
/// values of `x`.
fn solve<'a>(
    ctx: &'a HypreContext<'_>,
    comm: &'a SimpleCommunicator,
    rows: Range<u32>,
    a: &CSRMatrix<'_>,
    b: &ParVector<'_>,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let config = GMRESSolverConfigBuilder::default()
        .tol(1e-12)
        .k_dim(N as usize)
        .max_iters(10 * N as usize)
        .build()?;
    let mut x = ParVector::zeros(ctx, comm, rows)?;
    let status = ctx.gmres_solver(comm, config)?.solve(a, b, &mut x)?;
    assert!(status.converged, "{status}");
    Ok(x.local_values()?)
}

#[test]
fn builders() -> Result<(), Box<dyn std::error::Error>> {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let rows = distribute_rows(&comm, N);

        // A diagonal matrix built with repeated entries and `set` after
        // `add`: d_i = 3 for odd i (1 + 2), and 5 for even i (set, after
        // adding 7). Repeated entries overestimate the preallocation.
        let mut a = ctx.local_matrix_builder(&comm, rows.clone(), rows.clone())?;
        a.add(rows.clone().map(|i| (i, i, 1.0)))?;
        a.add(rows.clone().map(|i| (i, i, 2.0)))?;
        a.add(rows.clone().filter(|i| i % 2 == 0).map(|i| (i, i, 4.0)))?;
        a.set(rows.clone().filter(|i| i % 2 == 0).map(|i| (i, i, 5.0)))?;
        let a = a.assemble()?;
        let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |_| 15.0)?;
        let x = solve(&ctx, &comm, rows.clone(), &a, &b)?;
        for (i, v) in rows.clone().zip(x) {
            let expected = if i % 2 == 0 { 3.0 } else { 5.0 };
            assert!(
                (v - expected).abs() < 1e-8,
                "x[{i}] = {v}, expected {expected}"
            );
        }
        drop((a, b));

        // The same non-symmetric tridiagonal matrix, built row by row by
        // each owner (with off-diagonal-block columns on rank boundaries),
        // and element by element across ranks. Both give x = [1, ..., 1].
        let entries = |i: u32| {
            let mut row = vec![(i, i, 2.0)];
            if i > 0 {
                row.push((i, i - 1, -1.5));
            }
            if i + 1 < N {
                row.push((i, i + 1, -0.5));
            }
            row
        };
        let mut local = ctx.local_matrix_builder(&comm, rows.clone(), rows.clone())?;
        for i in rows.clone() {
            local.add(entries(i))?;
        }
        // Each rank gives the entries of the "elements" (i - 1, i) that end
        // in its rows: entries of row i - 1 may belong to the previous rank.
        let mut distributed = DistributedMatrixBuilder::with_off_rank_capacity(
            &ctx,
            &comm,
            rows.clone(),
            rows.clone(),
            4,
        )?;
        // Diagonal: 1 from row i itself, 1 from the element (i, i + 1),
        // which the last row does not have, so it adds the second 1 itself.
        for i in rows.clone() {
            distributed.add([(i, i, 1.0)])?;
            if i > 0 {
                distributed.add([(i - 1, i - 1, 1.0), (i - 1, i, -0.5), (i, i - 1, -1.5)])?;
            }
            if i + 1 == N {
                distributed.add([(i, i, 1.0)])?;
            }
        }
        // Row sums: 0.5 for row 0, 0 inside, 0.5 for the last row.
        let b = ParVector::from_fn(&ctx, &comm, rows.clone(), |i| {
            entries(i).iter().map(|e| e.2).sum()
        })?;
        for a in [local.assemble()?, distributed.assemble()?] {
            let x = solve(&ctx, &comm, rows.clone(), &a, &b)?;
            for (i, v) in rows.clone().zip(x) {
                assert!((v - 1.0).abs() < 1e-8, "x[{i}] = {v}, expected 1");
            }
        }
        drop(b);

        // A distributed vector equals the local one with the same values:
        // each rank adds its share of every row.
        let size = mpi::topology::Communicator::size(&comm);
        let mut v = ctx.distributed_vector_builder(&comm, rows.clone())?;
        v.add((0..N).map(|i| (i, f64::from(i) / f64::from(size))))?;
        let v = v.assemble()?;
        for (i, x) in rows.clone().zip(v.local_values()?) {
            assert!((x - f64::from(i)).abs() < 1e-12, "v[{i}] = {x}");
        }
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
