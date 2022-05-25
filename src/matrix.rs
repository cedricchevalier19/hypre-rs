use std::ptr::null_mut;

use hypre_sys::*;

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