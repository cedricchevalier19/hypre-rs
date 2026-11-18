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
        "Solver has converged in 12 iterations, reaching a relative residual of 0.5"
    );
    status.converged = false;
    assert_eq!(
        status.to_string(),
        "Solver has NOT converged in 12 iterations, reaching a relative residual of 0.5"
    );
}
