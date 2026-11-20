//! Unit tests of the solver types.

use super::*;

#[test]
fn status_display() {
    let mut status = IterativeSolverStatus {
        num_iters: 12,
        res_num: 0.5,
        converged: true,
    };
    assert_eq!(
        status.to_string(),
        "Solver has converged in 12 iterations, reaching a relative residual of 5.000e-1"
    );
    status.converged = false;
    assert_eq!(
        status.to_string(),
        "Solver has NOT converged in 12 iterations, reaching a relative residual of 5.000e-1"
    );
}

#[test]
fn solve_result_keeps_the_status_of_a_convergence_failure() {
    // HypreError::new clears hypre's global error flag, shared with other
    // unit tests.
    let _lock = crate::utils::hypre_error_lock();
    let status = IterativeSolverStatus {
        num_iters: 3,
        res_num: 0.25,
        converged: true,
    };
    assert!(matches!(solve_result(0, status), Ok(s) if s.converged && s.num_iters == 3));
    let conv = hypre_sys::HYPRE_ERROR_CONV as hypre_sys::HYPRE_Int;
    assert!(matches!(
        solve_result(conv, status),
        Err(HypreError::NotConverged(s)) if !s.converged && s.num_iters == 3 && s.res_num == 0.25
    ));
    let generic = hypre_sys::HYPRE_ERROR_GENERIC as hypre_sys::HYPRE_Int;
    assert!(matches!(
        solve_result(generic, status),
        Err(HypreError::HypreGenericError)
    ));
    // A convergence failure combined with another error is not one.
    assert!(matches!(
        solve_result(conv | generic, status),
        Err(HypreError::UnknowError)
    ));
}
