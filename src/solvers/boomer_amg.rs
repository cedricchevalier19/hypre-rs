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

#[non_exhaustive]
enum SymmetricBoomerAMGRelaxType {
    Jacobi,
    HybridGaussSeidelSymm,
    L1ScaledHybridGaussSeidelSymm,
    FCFJacobi,
    L1ScaledJacobi,
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

#[non_exhaustive]
enum BoomerAMGRelaxType {
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

#[non_exhaustive]
enum BoomerAMGCoarsenType {
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
    /// BAMG-Direct (GPU ?)
    BAMGDirect,
    /// Extended (GPU ?)
    Extended,
    /// Extended+i (GPU ?)
    ExtendedI,
    /// Extended+e (GPU ?)
    ExtendedE,
}

impl Default for BoomerAMGCoarsenType {
    fn default() -> Self {
        BoomerAMGCoarsenType::HMIS
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
            15 => Ok(BoomerAMGCoarsenType::BAMGDirect),
            14 => Ok(BoomerAMGCoarsenType::Extended),
            16 => Ok(BoomerAMGCoarsenType::ExtendedI),
            18 => Ok(BoomerAMGCoarsenType::ExtendedE),
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
            BoomerAMGCoarsenType::BAMGDirect => 15,
            BoomerAMGCoarsenType::Extended => 14,
            BoomerAMGCoarsenType::ExtendedI => 16,
            BoomerAMGCoarsenType::ExtendedE => 18,
        }
    }
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
