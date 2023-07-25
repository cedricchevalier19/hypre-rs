

use crate::solvers::{InternalLinearPreconditioner, SymmetricLinearPreconditioner};
use crate::{HypreError, HypreResult};
use hypre_sys::{
    HYPRE_BoomerAMGCreate, HYPRE_BoomerAMGSetCoarsenType, HYPRE_BoomerAMGSetMaxIter,
    HYPRE_BoomerAMGSetNumSweeps, HYPRE_BoomerAMGSetOldDefault, HYPRE_BoomerAMGSetPrintLevel,
    HYPRE_BoomerAMGSetRelaxType, HYPRE_BoomerAMGSetTol, HYPRE_BoomerAMGSetup, HYPRE_BoomerAMGSolve,
    HYPRE_Int, HYPRE_Matrix, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_PtrToSolverFcn,
    HYPRE_Solver, HYPRE_Vector,
};
use std::num::TryFromIntError;
use std::ptr::null_mut;

#[derive(Debug, Clone)]
pub struct BoomerAMG {
    internal_solver: HYPRE_Solver,
}

impl BoomerAMG {
    pub fn new() -> HypreResult<Self> {
        let mut solver = BoomerAMG {
            internal_solver: null_mut(),
        };
        unsafe {
            let h_solver = &mut solver.internal_solver as *mut _ as *mut HYPRE_Solver;
            check_hypre![HYPRE_BoomerAMGCreate(h_solver)];

            check_hypre![HYPRE_BoomerAMGSetPrintLevel(*h_solver, 1)]; /* print amg solution info */
            check_hypre![HYPRE_BoomerAMGSetCoarsenType(*h_solver, 6)];
            check_hypre![HYPRE_BoomerAMGSetOldDefault(*h_solver)];
            check_hypre![HYPRE_BoomerAMGSetRelaxType(*h_solver, 6)]; /* Sym G.S./Jacobi hybrid */
            check_hypre![HYPRE_BoomerAMGSetNumSweeps(*h_solver, 1)];
            check_hypre![HYPRE_BoomerAMGSetTol(*h_solver, 0.0)]; /* conv. tolerance zero */
            check_hypre![HYPRE_BoomerAMGSetMaxIter(*h_solver, 1)]; /* do only one iteration! */
        }
        Ok(solver)
    }

    unsafe extern "C" fn solve(
        solver: HYPRE_Solver,
        mat: HYPRE_Matrix,
        b: HYPRE_Vector,
        x: HYPRE_Vector,
    ) -> HYPRE_Int {
        unsafe {
            HYPRE_BoomerAMGSolve(
                solver,
                mat as HYPRE_ParCSRMatrix,
                b as HYPRE_ParVector,
                x as HYPRE_ParVector,
            )
        }
    }

    unsafe extern "C" fn setup(
        solver: HYPRE_Solver,
        mat: HYPRE_Matrix,
        b: HYPRE_Vector,
        x: HYPRE_Vector,
    ) -> HYPRE_Int {
        unsafe {
            HYPRE_BoomerAMGSetup(
                solver,
                mat as HYPRE_ParCSRMatrix,
                b as HYPRE_ParVector,
                x as HYPRE_ParVector,
            )
        }
    }
}

impl InternalLinearPreconditioner for BoomerAMG {
    fn get_precond(&self) -> HYPRE_PtrToSolverFcn {
        Some(BoomerAMG::solve)
    }

    fn get_precond_setup(&self) -> HYPRE_PtrToSolverFcn {
        Some(BoomerAMG::setup)
    }

    fn get_internal(&self) -> HYPRE_Solver {
        self.internal_solver
    }
}

impl SymmetricLinearPreconditioner for BoomerAMG {
    fn precond_descriptor(&self) -> &dyn InternalLinearPreconditioner {
        self
    }
}
