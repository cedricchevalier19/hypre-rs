use crate::error::HypreError;
use crate::matrix::Matrix::{ParCSR, IJ};
use crate::HypreResult;
use hypre_sys::*;
use mpi::topology::Communicator;
use std::{ffi::c_void, ptr::null_mut};

pub struct CSRMatrix {
    internal_matrix: HYPRE_ParCSRMatrix,
}

impl CSRMatrix {
    fn new(
        comm: impl Communicator,
        global_num_rows: usize,
        global_num_cols: usize,
        row_starts: &[usize],
        col_starts: &[usize],
    ) -> HypreResult<Self> {
        let mut out = CSRMatrix {
            internal_matrix: null_mut(),
        };
        let mut h_row_starts: Vec<HYPRE_BigInt> =
            row_starts.iter().map(|&x| x.try_into().unwrap()).collect();
        let mut h_col_starts: Vec<HYPRE_BigInt> =
            col_starts.iter().map(|&x| x.try_into().unwrap()).collect();
        unsafe {
            match HYPRE_ParCSRMatrixCreate(
                comm.as_raw(),
                global_num_rows.try_into().unwrap(),
                global_num_cols.try_into().unwrap(),
                h_row_starts.as_mut_ptr(),
                h_col_starts.as_mut_ptr(),
                0,
                0,
                0,
                &mut out.internal_matrix,
            ) {
                0 => Ok(out),
                x => Err(HypreError::new(x)),
            }
        }
    }

    fn get_internal_matrix(self) -> HypreResult<HYPRE_Matrix> {
        Ok(self.internal_matrix as HYPRE_Matrix)
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

impl TryFrom<IJMatrix> for CSRMatrix {
    type Error = HypreError;

    fn try_from(ij: IJMatrix) -> Result<Self, Self::Error> {
        let mut out = CSRMatrix {
            internal_matrix: null_mut(),
        };
        unsafe {
            check_hypre!(HYPRE_IJMatrixAssemble(ij.internal_matrix));
            let csr_mat_ptr = &mut out.internal_matrix as *mut _ as *mut *mut c_void;
            check_hypre!(HYPRE_IJMatrixGetObject(ij.internal_matrix, csr_mat_ptr));
        }
        Ok(out)
    }
}

impl IJMatrix {
    pub fn new(
        comm: impl mpi::topology::Communicator,
        rows: (usize, usize),
        cols: (usize, usize),
    ) -> HypreResult<Self> {
        let mut out = Self {
            internal_matrix: null_mut(),
        };
        unsafe {
            let h_matrix = &mut out.internal_matrix as *mut _ as *mut HYPRE_IJMatrix;
            check_hypre!(HYPRE_IJMatrixCreate(
                comm.as_raw(),
                rows.0.try_into().unwrap(),
                rows.1.try_into().unwrap(),
                cols.0.try_into().unwrap(),
                cols.1.try_into().unwrap(),
                h_matrix,
            ));

            check_hypre!(HYPRE_IJMatrixSetObjectType(*h_matrix, 0));
            check_hypre!(HYPRE_IJMatrixInitialize(*h_matrix));
        }
        Ok(out)
    }
}

impl Drop for IJMatrix {
    fn drop(&mut self) {
        unsafe {
            HYPRE_IJMatrixDestroy(self.internal_matrix);
        }
    }
}

pub enum Matrix {
    IJ(IJMatrix),
    ParCSR(CSRMatrix),
}

impl Matrix {
    pub(crate) fn get_internal_matrix(self) -> HypreResult<HYPRE_Matrix> {
        match self {
            IJ(m) => <IJMatrix as TryInto<CSRMatrix>>::try_into(m)?.get_internal_matrix(),
            ParCSR(m) => m.get_internal_matrix(),
        }
    }
}
