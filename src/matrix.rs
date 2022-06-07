use hypre_sys::*;
use std::{ffi::c_void, ptr::null_mut};

pub struct CSRMatrix {
    internal_matrix: HYPRE_ParCSRMatrix,
}

impl CSRMatrix {
    pub fn new(
        comm: MPI_Comm,
        global_num_rows: usize,
        global_num_cols: usize,
        row_starts: &[usize],
        col_starts: &[usize],
    ) -> Self {
        let mut out = CSRMatrix {
            internal_matrix: null_mut(),
        };
        let mut h_row_starts: Vec<HYPRE_BigInt> =
            row_starts.iter().map(|&x| x.try_into().unwrap()).collect();
        let mut h_col_starts: Vec<HYPRE_BigInt> =
            col_starts.iter().map(|&x| x.try_into().unwrap()).collect();
        unsafe {
            HYPRE_ParCSRMatrixCreate(
                comm,
                global_num_rows.try_into().unwrap(),
                global_num_cols.try_into().unwrap(),
                h_row_starts.as_mut_ptr(),
                h_col_starts.as_mut_ptr(),
                0,
                0,
                0,
                &mut out.internal_matrix,
            );
        };

        out
    }
}

impl Drop for CSRMatrix {
    fn drop(&mut self) {
        unsafe {
            HYPRE_ParCSRMatrixDestroy(self.internal_matrix);
        }
    }
}

pub struct IJMatrix {
    internal_matrix: HYPRE_IJMatrix,
}

impl From<IJMatrix> for CSRMatrix {
    fn from(ij: IJMatrix) -> Self {
        let mut out = CSRMatrix {
            internal_matrix: null_mut(),
        };
        unsafe {
            let csr_mat_ptr = &mut out.internal_matrix as *mut _ as *mut *mut c_void;
            HYPRE_IJMatrixGetObject(ij.internal_matrix, csr_mat_ptr);
        }
        out
    }
}

pub struct PCGSolver {
    internal_solver: HYPRE_Solver,
}

impl PCGSolver {
    fn check_hypre_error(mut self, err: i32) -> Option<PCGSolver> {
        if err != 0 {
            None
        } else {
            Some(self)
        }
    }

    pub fn set_tol(mut self, tol: f64) -> Option<PCGSolver> {
        let err = unsafe { HYPRE_PCGSetTol(self.internal_solver, tol.try_into().unwrap()) };
        self.check_hypre_error(err)
    }

    pub fn set_absolute_tol(mut self, tol: f64) -> Option<PCGSolver> {
        let err = unsafe { HYPRE_PCGSetAbsoluteTol(self.internal_solver, tol.try_into().unwrap()) };
        self.check_hypre_error(err)
    }

    pub fn set_residual_tol(mut self, tol: f64) -> Option<PCGSolver> {
        let err = unsafe { HYPRE_PCGSetResidualTol(self.internal_solver, tol.try_into().unwrap()) };
        self.check_hypre_error(err)
    }

    pub fn set_absolute_tol_factor(mut self, tol: f64) -> Option<PCGSolver> {
        let err =
            unsafe { HYPRE_PCGSetAbsoluteTolFactor(self.internal_solver, tol.try_into().unwrap()) };
        self.check_hypre_error(err)
    }

    pub fn set_convergence_factor_tol(mut self, tol: f64) -> Option<PCGSolver> {
        let err = unsafe {
            HYPRE_PCGSetConvergenceFactorTol(self.internal_solver, tol.try_into().unwrap())
        };
        self.check_hypre_error(err)
    }

    pub fn set_max_iter(mut self, max_iter: usize) -> Option<PCGSolver> {
        let err =
            unsafe { HYPRE_PCGSetMaxIter(self.internal_solver, max_iter.try_into().unwrap()) };
        self.check_hypre_error(err)
    }

    pub fn set_two_norm(mut self, act: bool) -> Option<PCGSolver> {
        let err = unsafe { HYPRE_PCGSetTwoNorm(self.internal_solver, act.try_into().unwrap()) };
        self.check_hypre_error(err)
    }

    pub fn set_rel_change(mut self, act: bool) -> Option<PCGSolver> {
        let err = unsafe { HYPRE_PCGSetRelChange(self.internal_solver, act.try_into().unwrap()) };
        self.check_hypre_error(err)
    }

    pub fn set_recompute_residual(mut self, act: bool) -> Option<PCGSolver> {
        let err =
            unsafe { HYPRE_PCGSetRecomputeResidual(self.internal_solver, act.try_into().unwrap()) };
        self.check_hypre_error(err)
    }

    pub fn set_recompute_residualp(mut self, act: bool) -> Option<PCGSolver> {
        let err = unsafe {
            HYPRE_PCGSetRecomputeResidualP(self.internal_solver, act.try_into().unwrap())
        };
        self.check_hypre_error(err)
    }
}
