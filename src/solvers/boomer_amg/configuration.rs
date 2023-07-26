use crate::{HypreError, HypreResult};
use derive_builder::Builder;

use hypre_sys::{HYPRE_BoomerAMGSetTol, HYPRE_Int};
use crate::solvers::BoomerAMG;

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BoomerAMGRelToleranceType {
    RHS,
    R0,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BoomerAMGMeasureType {
    LOCAL,
    GLOBAL,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BoomerAMGNodalType {
    UnkownCoarsening,
    Frobenius,
    SumAbsByBlock,
    LargestByBlock,
    RowSum,
    SumByBlock,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BoomerAMGNodalDiagType {
    NoTreatment,
    NegativeSum,
    Opposite,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BoomerAMGInterpolationType {
    Classical,
    LeastSquare,
    ClassicalHyperbolicPDE,
    Direct,
    MultiPass,
    MultiPassWeightsSeparation,
    ExtendedI,
    ExtendedInoC,
    Standard,
    StandardWeightsSeparation,
    ClassicalBlock,
    ClassicalBlockDiag,
    FF,
    FF1,
    Extended,
    AdaptiveWeights,
    ExtendedMatrixMatrix,
    ExtendedIMatrixMatrix,
    ExtendedEMatrixMatrix,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BoomerAMGAggInterpolationType {
    TwoStageExtendedI,
    TwoStageStandard,
    TwoStageExtended,
    MultiPass,
    TwoStageExtendedMatrixMatrix,
    TwoStageExtendedIMatrixMatrix,
    TwoStageExtendedEMatrixMatrix,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BoomerAMGCycleType {
    VCycle,
    WCycle,
    FCycle,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BoomerAMGRelaxOrder {
    Lexico,
    CFRelax,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum SymmetricBoomerAMGRelaxType {
    Jacobi,
    HybridGaussSeidelSymm,
    L1ScaledHybridGaussSeidelSymm,
    FCFJacobi,
    L1ScaledJacobi,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BoomerAMGCoarsenType {
    /// Luby coarsening, parallel
    CLJP,
    /// Classical Ruge-Stueben on each processor, no boundary treatment
    RugeStuebenNB,
    /// Classical Ruge-Stueben on each processor, then a third pass for boundary
    RugeStueben,
    /// RugeStuebenNB followed by CLJP
    Falgout,
    /// CLJP with fixed random vector, debug only
    CLJPFixed,
    /// Parallel independant sets
    PMIS,
    /// PMIS with fixed random vector, debug only
    PMISFixed,
    /// HMIS-coarsening: one-pass Ruge-Stueben then PMIS
    HMIS,
    /// one-pass Ruge-Stueben on each processor, no boundary treatment
    RugeStuebenOnePass,
    /// CGC coarsening
    CGC,
    /// CGC-E
    CGCE,
}

impl Default for BoomerAMGCoarsenType {
    fn default() -> Self {
        BoomerAMGCoarsenType::HMIS
    }
}

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BoomerAMGRelaxType {
    /// Jacobi smoother
    Jacobi,
    /// Gauss-Seidel smoother, sequential
    GaussSeidelSeq,
    /// Gauss-Seidel smoother
    GaussSeidelInterior,
    /// SOR Forward
    HybridGaussSeidelForward,
    /// SOR Backward
    HybridGaussSeidelBackward,
    /// Hybrid Gauss-Seidel, not advised
    HybridGaussSeidelChaos,
    /// SSOR
    HybridGaussSeidelSymm,
    /// $L_1$ scaled SSOR
    L1ScaledHybridGaussSeidelSymm,
    /// Gaussian Elimination on the coarsest level
    GaussianElimination,
    /// Two-stage Gauss-Seidel, forward
    TwoStageGaussSeidelForward,
    /// Two-stage Gauss-Seidel, backward
    TwoStageGaussSeidelBackward,
    /// $L_1$ scaled Gauss-Seidel, forward
    L1ScaledGaussSeidelForward,
    /// $L_1$ scaled Gauss-Seidel, backward
    L1ScaledGaussSeidelBackward,
    /// CG, not a fixed smoother, may required Flexible GMRES
    CG,
    /// Chebychev polynomial
    Chebychev,
    ///
    FCFJacobi,
    /// $L_1$ scaled Jacobi
    L1ScaledJacobi,
}

/// BoomerAMG Configuration
///
#[derive(Default, Debug, Clone, Builder)]
#[builder(setter(into, strip_option), default)]
#[builder(build_fn(validate = "Self::validate", error = "HypreError"))]
pub struct BoomerAMGConfig {
    /// convergence tolerance
    pub tol: Option<f64>,
    /// RHS or $r_0$ scaling tolerance
    pub tol_type: Option<BoomerAMGRelToleranceType>,
    /// Maximum number of iterations
    pub max_iters: Option<usize>,
    /// Maximum number of iterations
    pub min_iters: Option<usize>,
    /// AMG Cycle type
    pub cycle_type: Option<BoomerAMGCycleType>,
    /// Interpolation options
    pub interpolation: Option<BoomerAMGInterpolationConfig>,
    /// Relaxation options
    pub relaxation: Option<BoomerAMGRelaxationConfig>,
    /// Coarsening options
    pub coarsening: Option<BoomerAMGCoarseningConfig>,
}

impl BoomerAMGConfigBuilder {
    /// Validates valid parameters for [BoomerAMGConfig]
    fn validate(&self) -> HypreResult<()> {
        check_positive_parameter![self, tol];
        Ok(())
    }
}

impl BoomerAMGConfig {
    fn apply(&self, amg: &mut BoomerAMG) -> HypreResult<()> {
        set_parameter![HYPRE_BoomerAMGSetTol, amg.internal_solver, self.tol];
        Ok(())
    }
}

/// BoomerAMG Interpolation Configuration
///
#[derive(Default, Debug, Clone, Builder)]
#[builder(setter(into, strip_option), default)]
#[builder(build_fn(validate = "Self::validate", error = "HypreError"))]
pub struct BoomerAMGInterpolationConfig {
    /// Interpolation operator
    pub interpolation_type: Option<BoomerAMGInterpolationType>,
    /// Interpolation truncation factor
    pub trunc_factor: Option<f64>,
    /// Interpolation max number of elements per row
    pub max_elements: Option<usize>,
    /// Interpolation separation of weigths,
    pub weights_separation: Option<bool>,
    /// Aggressive Coarsening Interpolation
    pub agg_type: Option<BoomerAMGAggInterpolationType>,
    /// Aggressive Coarsening Interpolation truncation factor
    pub agg_trunc_factor: Option<f64>,
    /// Two stage interpolation truncation factor
    pub p12_trunc_factor: Option<f64>,
    /// Aggressive coarsening interpolation max number of elements by row
    pub agg_max_elements: Option<usize>,
    /// Max number of elements by row for P1 and P2 during 2-stage interpolation
    pub p12_max_elements: Option<usize>,
}

impl BoomerAMGInterpolationConfigBuilder {
    /// Validates valid parameters for [BoomerAMGInterpolationConfig]
    fn validate(&self) -> HypreResult<()> {
        Ok(())
    }
}

/// BoomerAMG Interpolation Configuration
///
#[derive(Default, Debug, Clone, Builder)]
#[builder(setter(into, strip_option), default)]
#[builder(build_fn(validate = "Self::validate", error = "HypreError"))]
pub struct BoomerAMGRelaxationConfig {
    /// Number of sweeps
    pub num_sweeps: Option<usize>,
    /// Smoother to use
    pub relax_type: Option<BoomerAMGRelaxType>,
    /// Order in which the points are relaxed
    pub relax_order: Option<BoomerAMGRelaxOrder>,
}

impl BoomerAMGRelaxationConfigBuilder {
    /// Validates valid parameters for [BoomerAMGRelaxationConfig]
    fn validate(&self) -> HypreResult<()> {
        Ok(())
    }
}

/// BoomerAMG Coarsening Configuration
///
#[derive(Default, Debug, Clone, Builder)]
#[builder(setter(into, strip_option), default)]
#[builder(build_fn(validate = "Self::validate", error = "HypreError"))]
pub struct BoomerAMGCoarseningConfig {
    /// Maximum coarse size
    pub max_coarse_size: Option<usize>,
    /// Maximum coarse size
    pub min_coarse_size: Option<usize>,
    /// Maximum number of levels
    pub max_levels: Option<usize>,
    /// Coarsen cut factor
    pub coarsen_cut_factor: Option<usize>,
    /// Strong Threshold for AMG
    pub strong_threshold: Option<f64>,
    /// Strong Threshold for Restriction
    pub strong_threshold_r: Option<f64>,
    /// Filter threshold for Restriction
    pub filter_threshold_r: Option<f64>,
    pub max_row_sum: Option<f64>,
    /// Coarsen type
    pub coarsen_type: Option<BoomerAMGCoarsenType>,
    /// Non-Galerkin tolerance
    pub non_galerkin_tol: Option<f64>,
    /// Set global or local measure
    pub measure_type: Option<BoomerAMGMeasureType>,
    /// Set number of aggressive coarsening levels
    pub num_agg_levels: Option<usize>,
    /// Define the degree of aggressive coarsening
    pub num_agg_paths: Option<usize>,
    /// Number of paths for CGC-coarsening
    pub num_cgc_iters: Option<usize>,
    /// Nodal systems coarsening
    pub nodal_type: Option<BoomerAMGNodalType>,
    /// Special diagonal treatment
    pub nodal_diag_type: Option<BoomerAMGNodalDiagType>,
}

impl BoomerAMGCoarseningConfigBuilder {
    /// Validates valid parameters for [BoomerAMGCoarseningConfig]
    fn validate(&self) -> HypreResult<()> {
        Ok(())
    }
}

impl TryFrom<BoomerAMGRelaxType> for SymmetricBoomerAMGRelaxType {
    type Error = HypreError;

    fn try_from(value: BoomerAMGRelaxType) -> Result<Self, Self::Error> {
        match value {
            BoomerAMGRelaxType::Jacobi => Ok(SymmetricBoomerAMGRelaxType::Jacobi),
            BoomerAMGRelaxType::HybridGaussSeidelSymm => {
                Ok(SymmetricBoomerAMGRelaxType::HybridGaussSeidelSymm)
            }
            BoomerAMGRelaxType::L1ScaledHybridGaussSeidelSymm => {
                Ok(SymmetricBoomerAMGRelaxType::L1ScaledHybridGaussSeidelSymm)
            }
            BoomerAMGRelaxType::FCFJacobi => Ok(SymmetricBoomerAMGRelaxType::FCFJacobi),
            BoomerAMGRelaxType::L1ScaledJacobi => Ok(SymmetricBoomerAMGRelaxType::L1ScaledJacobi),
            _ => Err(HypreError::InvalidParameterSymmetry),
        }
    }
}

impl TryFrom<usize> for BoomerAMGRelaxType {
    type Error = HypreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BoomerAMGRelaxType::Jacobi),
            1 => Ok(BoomerAMGRelaxType::GaussSeidelSeq),
            2 => Ok(BoomerAMGRelaxType::GaussSeidelInterior),
            3 => Ok(BoomerAMGRelaxType::HybridGaussSeidelForward),
            4 => Ok(BoomerAMGRelaxType::HybridGaussSeidelBackward),
            5 => Ok(BoomerAMGRelaxType::HybridGaussSeidelChaos),
            6 => Ok(BoomerAMGRelaxType::HybridGaussSeidelSymm),
            8 => Ok(BoomerAMGRelaxType::L1ScaledHybridGaussSeidelSymm),
            9 => Ok(BoomerAMGRelaxType::GaussianElimination),
            11 => Ok(BoomerAMGRelaxType::TwoStageGaussSeidelForward),
            12 => Ok(BoomerAMGRelaxType::TwoStageGaussSeidelBackward),
            13 => Ok(BoomerAMGRelaxType::L1ScaledGaussSeidelForward),
            14 => Ok(BoomerAMGRelaxType::L1ScaledGaussSeidelBackward),
            15 => Ok(BoomerAMGRelaxType::CG),
            16 => Ok(BoomerAMGRelaxType::Chebychev),
            17 => Ok(BoomerAMGRelaxType::FCFJacobi),
            18 => Ok(BoomerAMGRelaxType::L1ScaledJacobi),
            other => Err(HypreError::InvalidParameter(other)),
        }
    }
}

impl Into<HYPRE_Int> for BoomerAMGRelaxType {
    fn into(self) -> HYPRE_Int {
        match self {
            BoomerAMGRelaxType::Jacobi => 0,
            BoomerAMGRelaxType::GaussSeidelSeq => 1,
            BoomerAMGRelaxType::GaussSeidelInterior => 2,
            BoomerAMGRelaxType::HybridGaussSeidelForward => 3,
            BoomerAMGRelaxType::HybridGaussSeidelBackward => 4,
            BoomerAMGRelaxType::HybridGaussSeidelChaos => 5,
            BoomerAMGRelaxType::HybridGaussSeidelSymm => 6,
            BoomerAMGRelaxType::L1ScaledHybridGaussSeidelSymm => 8,
            BoomerAMGRelaxType::GaussianElimination => 9,
            BoomerAMGRelaxType::TwoStageGaussSeidelForward => 11,
            BoomerAMGRelaxType::TwoStageGaussSeidelBackward => 12,
            BoomerAMGRelaxType::L1ScaledGaussSeidelForward => 13,
            BoomerAMGRelaxType::L1ScaledGaussSeidelBackward => 14,
            BoomerAMGRelaxType::CG => 15,
            BoomerAMGRelaxType::Chebychev => 16,
            BoomerAMGRelaxType::FCFJacobi => 17,
            BoomerAMGRelaxType::L1ScaledJacobi => 18,
        }
    }
}

impl From<SymmetricBoomerAMGRelaxType> for BoomerAMGRelaxType {
    fn from(value: SymmetricBoomerAMGRelaxType) -> Self {
        match value {
            SymmetricBoomerAMGRelaxType::Jacobi => BoomerAMGRelaxType::Jacobi,
            SymmetricBoomerAMGRelaxType::HybridGaussSeidelSymm => {
                BoomerAMGRelaxType::HybridGaussSeidelSymm
            }
            SymmetricBoomerAMGRelaxType::L1ScaledHybridGaussSeidelSymm => {
                BoomerAMGRelaxType::L1ScaledHybridGaussSeidelSymm
            }
            SymmetricBoomerAMGRelaxType::FCFJacobi => BoomerAMGRelaxType::FCFJacobi,
            SymmetricBoomerAMGRelaxType::L1ScaledJacobi => BoomerAMGRelaxType::L1ScaledJacobi,
        }
    }
}

impl TryFrom<usize> for BoomerAMGNodalType {
    type Error = HypreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BoomerAMGNodalType::UnkownCoarsening),
            1 => Ok(BoomerAMGNodalType::Frobenius),
            2 => Ok(BoomerAMGNodalType::SumAbsByBlock),
            3 => Ok(BoomerAMGNodalType::LargestByBlock),
            4 => Ok(BoomerAMGNodalType::RowSum),
            6 => Ok(BoomerAMGNodalType::SumByBlock),
            other => Err(HypreError::InvalidParameter(other)),
        }
    }
}

impl Into<HYPRE_Int> for BoomerAMGNodalType {
    fn into(self) -> HYPRE_Int {
        match self {
            BoomerAMGNodalType::UnkownCoarsening => 0,
            BoomerAMGNodalType::Frobenius => 1,
            BoomerAMGNodalType::SumAbsByBlock => 2,
            BoomerAMGNodalType::LargestByBlock => 3,
            BoomerAMGNodalType::RowSum => 4,
            BoomerAMGNodalType::SumByBlock => 6,
        }
    }
}

impl TryFrom<usize> for BoomerAMGNodalDiagType {
    type Error = HypreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BoomerAMGNodalDiagType::NoTreatment),
            1 => Ok(BoomerAMGNodalDiagType::NegativeSum),
            2 => Ok(BoomerAMGNodalDiagType::Opposite),
            other => Err(HypreError::InvalidParameter(other)),
        }
    }
}

impl Into<HYPRE_Int> for BoomerAMGNodalDiagType {
    fn into(self) -> HYPRE_Int {
        match self {
            BoomerAMGNodalDiagType::NoTreatment => 0,
            BoomerAMGNodalDiagType::NegativeSum => 1,
            BoomerAMGNodalDiagType::Opposite => 2,
        }
    }
}

impl TryFrom<usize> for BoomerAMGInterpolationType {
    type Error = HypreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BoomerAMGInterpolationType::Classical),
            1 => Ok(BoomerAMGInterpolationType::LeastSquare),
            2 => Ok(BoomerAMGInterpolationType::ClassicalHyperbolicPDE),
            3 => Ok(BoomerAMGInterpolationType::Direct),
            4 => Ok(BoomerAMGInterpolationType::MultiPass),
            5 => Ok(BoomerAMGInterpolationType::MultiPassWeightsSeparation),
            6 => Ok(BoomerAMGInterpolationType::ExtendedI),
            7 => Ok(BoomerAMGInterpolationType::ExtendedInoC),
            8 => Ok(BoomerAMGInterpolationType::Standard),
            9 => Ok(BoomerAMGInterpolationType::StandardWeightsSeparation),
            10 => Ok(BoomerAMGInterpolationType::ClassicalBlock),
            11 => Ok(BoomerAMGInterpolationType::ClassicalBlockDiag),
            12 => Ok(BoomerAMGInterpolationType::FF),
            13 => Ok(BoomerAMGInterpolationType::FF1),
            14 => Ok(BoomerAMGInterpolationType::Extended),
            15 => Ok(BoomerAMGInterpolationType::AdaptiveWeights),
            16 => Ok(BoomerAMGInterpolationType::ExtendedMatrixMatrix),
            17 => Ok(BoomerAMGInterpolationType::ExtendedIMatrixMatrix),
            18 => Ok(BoomerAMGInterpolationType::ExtendedEMatrixMatrix),
            other => Err(HypreError::InvalidParameter(other)),
        }
    }
}

impl Into<HYPRE_Int> for BoomerAMGInterpolationType {
    fn into(self) -> HYPRE_Int {
        match self {
            BoomerAMGInterpolationType::Classical => 0,
            BoomerAMGInterpolationType::LeastSquare => 1,
            BoomerAMGInterpolationType::ClassicalHyperbolicPDE => 2,
            BoomerAMGInterpolationType::Direct => 3,
            BoomerAMGInterpolationType::MultiPass => 4,
            BoomerAMGInterpolationType::MultiPassWeightsSeparation => 5,
            BoomerAMGInterpolationType::ExtendedI => 6,
            BoomerAMGInterpolationType::ExtendedInoC => 7,
            BoomerAMGInterpolationType::Standard => 8,
            BoomerAMGInterpolationType::StandardWeightsSeparation => 9,
            BoomerAMGInterpolationType::ClassicalBlock => 10,
            BoomerAMGInterpolationType::ClassicalBlockDiag => 11,
            BoomerAMGInterpolationType::FF => 12,
            BoomerAMGInterpolationType::FF1 => 13,
            BoomerAMGInterpolationType::Extended => 14,
            BoomerAMGInterpolationType::AdaptiveWeights => 15,
            BoomerAMGInterpolationType::ExtendedMatrixMatrix => 16,
            BoomerAMGInterpolationType::ExtendedIMatrixMatrix => 17,
            BoomerAMGInterpolationType::ExtendedEMatrixMatrix => 18,
        }
    }
}

impl TryFrom<usize> for BoomerAMGAggInterpolationType {
    type Error = HypreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(BoomerAMGAggInterpolationType::TwoStageExtendedI),
            2 => Ok(BoomerAMGAggInterpolationType::TwoStageStandard),
            3 => Ok(BoomerAMGAggInterpolationType::TwoStageExtended),
            4 => Ok(BoomerAMGAggInterpolationType::MultiPass),
            5 => Ok(BoomerAMGAggInterpolationType::TwoStageExtendedMatrixMatrix),
            6 => Ok(BoomerAMGAggInterpolationType::TwoStageExtendedIMatrixMatrix),
            7 => Ok(BoomerAMGAggInterpolationType::TwoStageExtendedEMatrixMatrix),
            other => Err(HypreError::InvalidParameter(other)),
        }
    }
}

impl Into<HYPRE_Int> for BoomerAMGAggInterpolationType {
    fn into(self) -> HYPRE_Int {
        match self {
            BoomerAMGAggInterpolationType::TwoStageExtendedI => 1,
            BoomerAMGAggInterpolationType::TwoStageStandard => 2,
            BoomerAMGAggInterpolationType::TwoStageExtended => 3,
            BoomerAMGAggInterpolationType::MultiPass => 4,
            BoomerAMGAggInterpolationType::TwoStageExtendedMatrixMatrix => 5,
            BoomerAMGAggInterpolationType::TwoStageExtendedIMatrixMatrix => 6,
            BoomerAMGAggInterpolationType::TwoStageExtendedEMatrixMatrix => 7,
        }
    }
}

impl TryFrom<usize> for BoomerAMGCycleType {
    type Error = HypreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(BoomerAMGCycleType::VCycle),
            2 => Ok(BoomerAMGCycleType::WCycle),
            other => Err(HypreError::InvalidParameter(other)),
        }
    }
}

impl TryFrom<usize> for BoomerAMGRelaxOrder {
    type Error = HypreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BoomerAMGRelaxOrder::Lexico),
            1 => Ok(BoomerAMGRelaxOrder::CFRelax),
            other => Err(HypreError::InvalidParameter(other)),
        }
    }
}

impl Into<HYPRE_Int> for BoomerAMGRelaxOrder {
    fn into(self) -> HYPRE_Int {
        match self {
            BoomerAMGRelaxOrder::Lexico => 0,
            BoomerAMGRelaxOrder::CFRelax => 1,
        }
    }
}

impl TryFrom<usize> for BoomerAMGCoarsenType {
    type Error = HypreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BoomerAMGCoarsenType::CLJP),
            1 => Ok(BoomerAMGCoarsenType::RugeStuebenNB),
            3 => Ok(BoomerAMGCoarsenType::RugeStueben),
            6 => Ok(BoomerAMGCoarsenType::Falgout),
            7 => Ok(BoomerAMGCoarsenType::CLJPFixed),
            8 => Ok(BoomerAMGCoarsenType::PMIS),
            9 => Ok(BoomerAMGCoarsenType::PMISFixed),
            10 => Ok(BoomerAMGCoarsenType::HMIS),
            11 => Ok(BoomerAMGCoarsenType::RugeStuebenOnePass),
            21 => Ok(BoomerAMGCoarsenType::CGC),
            22 => Ok(BoomerAMGCoarsenType::CGCE),
            other => Err(HypreError::InvalidParameter(other)),
        }
    }
}

impl Into<HYPRE_Int> for BoomerAMGCoarsenType {
    fn into(self) -> HYPRE_Int {
        match self {
            BoomerAMGCoarsenType::CLJP => 0,
            BoomerAMGCoarsenType::RugeStuebenNB => 1,
            BoomerAMGCoarsenType::RugeStueben => 3,
            BoomerAMGCoarsenType::Falgout => 6,
            BoomerAMGCoarsenType::CLJPFixed => 7,
            BoomerAMGCoarsenType::PMIS => 8,
            BoomerAMGCoarsenType::PMISFixed => 9,
            BoomerAMGCoarsenType::HMIS => 10,
            BoomerAMGCoarsenType::RugeStuebenOnePass => 11,
            BoomerAMGCoarsenType::CGC => 21,
            BoomerAMGCoarsenType::CGCE => 22,
        }
    }
}
