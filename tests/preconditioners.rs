//! Preconditioned Krylov solvers, on any number of MPI ranks.

use hypre_rs::matrix::CSRMatrix;
use hypre_rs::preconditioners::{
    AMGPreconditionerConfigBuilder, JacobiPreconditioner, Preconditioner,
    SymmetricAMGPreconditionerConfigBuilder, SymmetricPreconditioner, SymmetricSmoother,
};
use hypre_rs::solvers::{
    BiCGSTABSolverConfigBuilder, COGMRESSolverConfigBuilder, FlexGMRESSolverConfigBuilder,
    GMRESSolverConfigBuilder, IterativeSolverStatus, LGMRESSolverConfigBuilder, LinearSolver,
    PCGSolverConfigBuilder, Solver, SymmetricLinearSolver, SymmetricSolver,
};
use hypre_rs::vector::ParVector;
use hypre_rs::{HypreContext, HypreError, SymmetricMatrix};
use mpi::topology::{Communicator, SimpleCommunicator};
use std::ops::Range;

/// Global number of rows; not a multiple of 2 or 3, so ranks get uneven rows.
const N: u32 = 100;

/// Below and above the diagonal of the convection-diffusion matrix
/// `tridiag(LOWER, 2, UPPER)`, which is not symmetric.
const LOWER: f64 = -1.5;
const UPPER: f64 = -0.5;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// The assembled vector whose entry `i` is `f(i)`.
fn vector<'a>(
    ctx: &'a HypreContext<'_>,
    comm: &'a SimpleCommunicator,
    rows: Range<u32>,
    f: impl Fn(u32) -> f64,
) -> Result<ParVector<'a>, HypreError> {
    let mut v = ctx.ij_vector(comm, rows.clone())?;
    v.add_elements(rows.map(|i| (i, f(i))))?;
    v.assemble()
}

/// Checks that `x = [1, ..., 1]`.
fn check_ones(x: &ParVector<'_>, rows: Range<u32>, what: &str) -> TestResult {
    for (i, v) in rows.clone().zip(x.get_values(rows)?) {
        assert!((v - 1.0).abs() < 1e-6, "{what}: x[{i}] = {v}, expected 1");
    }
    Ok(())
}

/// Solves `a * x = b` with CG, from `x = 0`, and checks that `x = [1, ..., 1]`.
fn cg(
    ctx: &HypreContext<'_>,
    comm: &SimpleCommunicator,
    rows: Range<u32>,
    a: &SymmetricMatrix<CSRMatrix<'_>>,
    preconditioner: Option<SymmetricPreconditioner<'_>>,
) -> TestResult<IterativeSolverStatus> {
    let what = format!("CG with {preconditioner:?}");
    let config = PCGSolverConfigBuilder::default()
        .tol(1e-12)
        .max_iters(10 * N as usize)
        .build()?;
    let mut solver = ctx.pcg_solver(comm, config)?;
    if let Some(p) = preconditioner {
        solver = solver.with_preconditioner(p)?;
    }
    let solver = SymmetricSolver::CG(solver);
    let b = vector(ctx, comm, rows.clone(), |i| {
        if i == 0 || i == N - 1 {
            1.0
        } else {
            0.0
        }
    })?;
    let mut x = vector(ctx, comm, rows.clone(), |_| 0.0)?;
    let status = solver.solve(a, &b, &mut x)?;
    assert!(status.converged, "{what}: {status}");
    check_ones(&x, rows, &what)?;
    Ok(status)
}

/// Every solver of general matrices, each with the preconditioner `make()`.
fn general_solvers<'a>(
    ctx: &'a HypreContext<'_>,
    comm: &'a SimpleCommunicator,
    make: impl Fn() -> Result<Preconditioner<'a>, HypreError>,
) -> Result<Vec<Solver<'a>>, HypreError> {
    let (tol, max_iters) = (1e-12, 10 * N as usize);
    Ok(vec![
        Solver::GMRES(
            ctx.gmres_solver(
                comm,
                GMRESSolverConfigBuilder::default()
                    .tol(tol)
                    .max_iters(max_iters)
                    .k_dim(20usize)
                    .build()?,
            )?
            .with_preconditioner(make()?)?,
        ),
        Solver::FlexGMRES(
            ctx.flex_gmres_solver(
                comm,
                FlexGMRESSolverConfigBuilder::default()
                    .tol(tol)
                    .max_iters(max_iters)
                    .k_dim(20usize)
                    .build()?,
            )?
            .with_preconditioner(make()?)?,
        ),
        Solver::LGMRES(
            ctx.lgmres_solver(
                comm,
                LGMRESSolverConfigBuilder::default()
                    .tol(tol)
                    .max_iters(max_iters)
                    .k_dim(20usize)
                    .build()?,
            )?
            .with_preconditioner(make()?)?,
        ),
        Solver::COGMRES(
            ctx.cogmres_solver(
                comm,
                COGMRESSolverConfigBuilder::default()
                    .tol(tol)
                    .max_iters(max_iters)
                    .k_dim(20usize)
                    .build()?,
            )?
            .with_preconditioner(make()?)?,
        ),
        Solver::BiCGSTAB(
            ctx.bicgstab_solver(
                comm,
                BiCGSTABSolverConfigBuilder::default()
                    .tol(tol)
                    .max_iters(max_iters)
                    .build()?,
            )?
            .with_preconditioner(make()?)?,
        ),
    ])
}

#[test]
fn preconditioners() -> TestResult {
    // Keep the universe alive: dropping it calls MPI_Finalize.
    let universe =
        mpi::initialize().ok_or("MPI was already initialized (or finalized) in this process")?;
    let comm = universe.world();
    let ctx = hypre_rs::initialize(&universe)?;
    {
        let (rank, size) = (comm.rank() as u32, comm.size() as u32);
        let rows = (rank * N / size)..((rank + 1) * N / size);

        // CG on the 1D Laplacian tridiag(-1, 2, -1).
        let mut a = ctx.symmetric_matrix(&comm, rows.clone())?;
        a.add_elements(rows.clone().map(|i| (i, i, 2.0)))?;
        a.add_elements(rows.clone().filter(|&i| i > 0).map(|i| (i, i - 1, -1.0)))?;
        let a = a.assemble()?;

        let plain = cg(&ctx, &comm, rows.clone(), &a, None)?;
        cg(
            &ctx,
            &comm,
            rows.clone(),
            &a,
            Some(JacobiPreconditioner.into()),
        )?;
        // AMG makes CG converge in a few iterations, whatever the smoother.
        let default_amg = ctx.symmetric_amg_preconditioner(&comm, Default::default())?;
        let mut amg_configs = vec![SymmetricAMGPreconditionerConfigBuilder::default()];
        for smoother in [
            SymmetricSmoother::Jacobi,
            SymmetricSmoother::HybridSymmetricGaussSeidel,
            SymmetricSmoother::L1SymmetricGaussSeidel,
            SymmetricSmoother::Chebyshev,
            SymmetricSmoother::L1Jacobi,
        ] {
            let mut config = SymmetricAMGPreconditionerConfigBuilder::default();
            config.relax_type(smoother);
            if smoother == SymmetricSmoother::Jacobi {
                // Unweighted Jacobi does not smooth the 1D Laplacian.
                config.relax_wt(0.5);
            }
            amg_configs.push(config);
        }
        // Two W-cycles per application.
        amg_configs.push(
            *SymmetricAMGPreconditionerConfigBuilder::default()
                .cycle_type(2u32)
                .max_iters(2usize),
        );
        for config in amg_configs {
            let amg = ctx.symmetric_amg_preconditioner(&comm, config.build()?)?;
            let status = cg(&ctx, &comm, rows.clone(), &a, Some(amg.into()))?;
            assert!(
                status.num_iters < 20 && status.num_iters < plain.num_iters,
                "{config:?}: {status}, without preconditioner: {plain}"
            );
        }

        // hypre's default configuration is overridden: one cycle, no tolerance.
        let current = default_amg.current_config()?;
        assert_eq!(current.max_iters, Some(1));

        // A new preconditioner replaces the previous one.
        let solver = ctx
            .pcg_solver(&comm, Default::default())?
            .with_preconditioner(JacobiPreconditioner)?
            .with_preconditioner(default_amg)?;
        assert!(matches!(
            solver.preconditioner(),
            Some(SymmetricPreconditioner::AMG(_))
        ));
        drop(solver);

        // A symmetric matrix also works with the general solvers.
        let b = vector(&ctx, &comm, rows.clone(), |i| {
            if i == 0 || i == N - 1 {
                1.0
            } else {
                0.0
            }
        })?;
        for solver in general_solvers(&ctx, &comm, || {
            Ok(ctx
                .symmetric_amg_preconditioner(&comm, Default::default())?
                .into())
        })? {
            let mut x = vector(&ctx, &comm, rows.clone(), |_| 0.0)?;
            let status = solver.solve(&a, &b, &mut x)?;
            assert!(status.converged, "{solver:?}: {status}");
            check_ones(&x, rows.clone(), &format!("{solver:?}"))?;
        }
        drop((a, b));

        // The non-symmetric convection-diffusion matrix, with every general
        // solver and preconditioner.
        let mut ij = ctx.ij_matrix(&comm, rows.clone(), rows.clone())?;
        for i in rows.clone() {
            ij.add_elements([(i, i, 2.0)].into_iter())?;
            if i > 0 {
                ij.add_elements([(i, i - 1, LOWER)].into_iter())?;
            }
            if i + 1 < N {
                ij.add_elements([(i, i + 1, UPPER)].into_iter())?;
            }
        }
        let a = ij.assemble()?;
        // A * [1, ..., 1]
        let b = vector(&ctx, &comm, rows.clone(), |i| {
            2.0 + if i > 0 { LOWER } else { 0.0 } + if i + 1 < N { UPPER } else { 0.0 }
        })?;
        let jacobi = || Ok(JacobiPreconditioner.into());
        let symmetric_amg = || {
            Ok(ctx
                .symmetric_amg_preconditioner(&comm, Default::default())?
                .into())
        };
        // Forward hybrid Gauss-Seidel, C points first: not symmetric.
        let amg = || {
            let config = AMGPreconditionerConfigBuilder::default()
                .relax_type(3u32)
                .relax_order(true)
                .build()?;
            Ok(ctx.amg_preconditioner(&comm, config)?.into())
        };
        for solvers in [
            general_solvers(&ctx, &comm, jacobi)?,
            general_solvers(&ctx, &comm, symmetric_amg)?,
            general_solvers(&ctx, &comm, amg)?,
        ] {
            for solver in solvers {
                let mut x = vector(&ctx, &comm, rows.clone(), |_| 0.0)?;
                let status = solver.solve(&a, &b, &mut x)?;
                assert!(status.converged, "{solver:?}: {status}");
                check_ones(&x, rows.clone(), &format!("{solver:?}"))?;
            }
        }

        // The general AMG preconditioner reads back its configuration.
        let config = AMGPreconditionerConfigBuilder::default()
            .max_iters(2usize)
            .coarsen_type(8u32)
            .build()?;
        let current = ctx.amg_preconditioner(&comm, config)?.current_config()?;
        assert_eq!(current.max_iters, Some(2));
        assert_eq!(current.coarsen_type, Some(8));
    }
    // hypre objects must be dropped before hypre is finalized (checked by the compiler)
    ctx.finalize()?;
    Ok(())
}
