//! Unit tests of the AMG solver configuration.

use super::*;

#[test]
fn config_rejects_negative_parameters() {
    for builder in [
        *AMGSolverConfigBuilder::default().tol(-1.0),
        *AMGSolverConfigBuilder::default().strong_threshold(-0.25),
        *AMGSolverConfigBuilder::default().max_row_sum(-1.0),
        *AMGSolverConfigBuilder::default().trunc_factor(-0.1),
    ] {
        assert!(matches!(
            builder.build(),
            Err(HypreError::InvalidParameterPositive)
        ));
    }
    // A negative relaxation weight is valid: hypre then estimates it.
    assert!(AMGSolverConfigBuilder::default()
        .relax_wt(-1.0)
        .build()
        .is_ok());
}
