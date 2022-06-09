use hypre_sys::*;
use mpi::topology::Communicator;
use std::{ffi::c_void, ptr::null_mut};

pub struct CSRMatrix {
    internal_matrix: HYPRE_ParCSRMatrix,
}

impl CSRMatrix {
    pub fn new(
        comm: impl Communicator,
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
                comm.as_raw(),
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
