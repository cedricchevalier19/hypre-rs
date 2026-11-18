//! Unit tests of the preconditioner types and configurations.

use super::*;
use crate::error::HypreError;
use hypre_sys::HYPRE_Int;

#[test]
fn amg_configs_reject_negative_parameters() {
    for builder in [
        *SymmetricAMGPreconditionerConfigBuilder::default().strong_threshold(-0.25),
        *SymmetricAMGPreconditionerConfigBuilder::default().max_row_sum(-1.0),
        *SymmetricAMGPreconditionerConfigBuilder::default().trunc_factor(-0.1),
    ] {
        assert!(matches!(
            builder.build(),
            Err(HypreError::InvalidParameterPositive)
        ));
    }
    for builder in [
        *AMGPreconditionerConfigBuilder::default().strong_threshold(-0.25),
        *AMGPreconditionerConfigBuilder::default().max_row_sum(-1.0),
        *AMGPreconditionerConfigBuilder::default().trunc_factor(-0.1),
    ] {
        assert!(matches!(
            builder.build(),
            Err(HypreError::InvalidParameterPositive)
        ));
    }
}

#[test]
fn symmetric_smoothers_are_hypre_relax_types() {
    let types: Vec<HYPRE_Int> = [
        SymmetricSmoother::Jacobi,
        SymmetricSmoother::HybridSymmetricGaussSeidel,
        SymmetricSmoother::L1SymmetricGaussSeidel,
        SymmetricSmoother::Chebyshev,
        SymmetricSmoother::L1Jacobi,
    ]
    .into_iter()
    .map(HYPRE_Int::from)
    .collect();
    assert_eq!(types, [0, 6, 8, 16, 18]);
}

#[test]
fn jacobi_converts_to_both_families() {
    let symmetric = SymmetricPreconditioner::from(JacobiPreconditioner);
    assert!(matches!(symmetric, SymmetricPreconditioner::Jacobi(_)));
    assert!(matches!(
        Preconditioner::from(symmetric),
        Preconditioner::Symmetric(SymmetricPreconditioner::Jacobi(_))
    ));
    let raw = Preconditioner::from(JacobiPreconditioner).raw();
    assert!(raw.solve.is_some() && raw.setup.is_some() && raw.solver.is_null());
}
