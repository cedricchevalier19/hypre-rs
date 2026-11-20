//! Named values for the solver options that hypre encodes as integers.
//!
//! Each enum converts into hypre's integer (`HYPRE_Int::from`), and
//! `current_config` converts hypre's integer back, returning
//! [HypreError::UnknownOptionValue] for a value the enum does not list.
//! The enums are `#[non_exhaustive]`: new hypre options can be added
//! without breaking code that matches on them.

use crate::error::HypreError;
use crate::solvers::solver_macro::FromHypre;
use crate::HypreResult;
use hypre_sys::HYPRE_Int;

/// Defines an option enum with explicit hypre values, and its conversions
/// to and from `HYPRE_Int`.
macro_rules! hypre_option {
    (
        $(#[$doc:meta])*
        $Enum:ident {
            $( $(#[$vdoc:meta])* $Variant:ident = $value:literal, )+
        }
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum $Enum {
            $( $(#[$vdoc])* $Variant, )+
        }

        impl From<$Enum> for HYPRE_Int {
            fn from(option: $Enum) -> Self {
                match option {
                    $( $Enum::$Variant => $value, )+
                }
            }
        }

        impl FromHypre<HYPRE_Int> for $Enum {
            fn from_hypre(raw: HYPRE_Int) -> HypreResult<Self> {
                match raw {
                    $( $value => Ok($Enum::$Variant), )+
                    other => Err(HypreError::UnknownOptionValue(other)),
                }
            }
        }
    };
}

hypre_option! {
    /// Parallel coarsening algorithm of algebraic multigrid
    /// (`HYPRE_BoomerAMGSetCoarsenType`). hypre's default is
    /// [CoarsenType::Hmis].
    CoarsenType {
        /// CLJP: a parallel coarsening using independent sets.
        Cljp = 0,
        /// Classical Ruge-Stueben on each process, without boundary
        /// treatment (not recommended).
        RugeStueben = 1,
        /// Classical Ruge-Stueben on each process, followed by a third pass
        /// that adds coarse points on the process boundaries.
        RugeStuebenThirdPass = 3,
        /// Falgout: Ruge-Stueben on each process, then CLJP starting from
        /// its interior coarse points.
        Falgout = 6,
        /// CLJP with a fixed random vector, for debugging.
        CljpFixedRandom = 7,
        /// PMIS: parallel independent sets, with lower complexities than
        /// CLJP but possibly slower convergence.
        Pmis = 8,
        /// PMIS with a fixed random vector, for debugging.
        PmisFixedRandom = 9,
        /// HMIS: one pass of Ruge-Stueben on each process, then PMIS
        /// starting from its interior coarse points.
        Hmis = 10,
        /// One-pass Ruge-Stueben on each process, without boundary
        /// treatment (not recommended).
        RugeStuebenOnePass = 11,
        /// CGC coarsening (Griebel, Metsch and Schweitzer).
        Cgc = 21,
        /// CGC-E coarsening (Griebel, Metsch and Schweitzer).
        CgcE = 22,
    }
}

hypre_option! {
    /// Interpolation operator of algebraic multigrid
    /// (`HYPRE_BoomerAMGSetInterpType`). hypre's default is
    /// [InterpType::ExtendedPlusI].
    InterpType {
        /// Classical modified interpolation.
        ClassicalModified = 0,
        /// LS interpolation, for use with GSMG.
        LeastSquares = 1,
        /// Classical modified interpolation for hyperbolic PDEs.
        ClassicalHyperbolic = 2,
        /// Direct interpolation, with separation of weights.
        Direct = 3,
        /// Multipass interpolation.
        Multipass = 4,
        /// Multipass interpolation, with separation of weights.
        MultipassSeparated = 5,
        /// Extended+i interpolation.
        ExtendedPlusI = 6,
        /// Extended+i interpolation, only where there is no common coarse
        /// neighbor.
        ExtendedPlusIIfNoCommonC = 7,
        /// Standard interpolation.
        Standard = 8,
        /// Standard interpolation, with separation of weights.
        StandardSeparated = 9,
        /// Classical block interpolation (nodal systems only).
        ClassicalBlock = 10,
        /// Classical block interpolation with diagonalized diagonal blocks
        /// (nodal systems only).
        ClassicalBlockDiagonal = 11,
        /// FF interpolation.
        Ff = 12,
        /// FF1 interpolation.
        Ff1 = 13,
        /// Extended interpolation.
        Extended = 14,
        /// Interpolation with adaptive weights (GPU only).
        AdaptiveWeights = 15,
        /// Extended interpolation, in matrix-matrix form.
        ExtendedMatrix = 16,
        /// Extended+i interpolation, in matrix-matrix form.
        ExtendedPlusIMatrix = 17,
        /// Extended+e interpolation, in matrix-matrix form.
        ExtendedPlusEMatrix = 18,
    }
}

hypre_option! {
    /// Smoother of algebraic multigrid on every level but the coarsest
    /// (`HYPRE_BoomerAMGSetRelaxType`; the coarsest level then uses Gaussian
    /// elimination). hypre's default, ℓ1 Gauss-Seidel forward on the way
    /// down and backward on the way up, has no single value here.
    ///
    /// Only the smoothers that hypre 2.26 and later document are listed.
    /// For the conjugate gradient solver, see
    /// [SymmetricSmoother](crate::preconditioners::SymmetricSmoother).
    Smoother {
        /// Jacobi; see `relax_wt`.
        Jacobi = 0,
        /// Sequential Gauss-Seidel (very slow).
        GaussSeidelSequential = 1,
        /// Gauss-Seidel, interior points in parallel and boundary points
        /// sequentially (slow).
        GaussSeidelInteriorParallel = 2,
        /// Hybrid Gauss-Seidel or SOR, forward.
        HybridGaussSeidelForward = 3,
        /// Hybrid Gauss-Seidel or SOR, backward.
        HybridGaussSeidelBackward = 4,
        /// Hybrid chaotic Gauss-Seidel (with OpenMP only).
        HybridChaoticGaussSeidel = 5,
        /// Hybrid symmetric Gauss-Seidel, or SSOR.
        HybridSymmetricGaussSeidel = 6,
        /// ℓ1-scaled hybrid symmetric Gauss-Seidel.
        L1SymmetricGaussSeidel = 8,
        /// ℓ1 Gauss-Seidel, forward.
        L1GaussSeidelForward = 13,
        /// ℓ1 Gauss-Seidel, backward.
        L1GaussSeidelBackward = 14,
        /// Conjugate gradient: not a fixed operator, so it may need a
        /// flexible outer solver (FlexGMRES).
        Cg = 15,
        /// Chebyshev polynomial.
        Chebyshev = 16,
        /// FCF-Jacobi.
        FcfJacobi = 17,
        /// ℓ1-scaled Jacobi.
        L1Jacobi = 18,
    }
}

hypre_option! {
    /// Multigrid cycle (`HYPRE_BoomerAMGSetCycleType`). hypre's default is
    /// [CycleType::V].
    CycleType {
        /// V-cycle.
        V = 1,
        /// W-cycle.
        W = 2,
    }
}

hypre_option! {
    /// Convergence criterion of the algebraic multigrid solver
    /// (`HYPRE_BoomerAMGSetConvergeType`). hypre's default is
    /// [ConvergeType::RelativeToRhs].
    ConvergeType {
        /// `‖r‖ / ‖b‖`, or `‖r‖` when `b = 0`.
        RelativeToRhs = 0,
        /// `‖r‖ / ‖r₀‖`, relative to the initial residual.
        RelativeToInitialResidual = 1,
    }
}

hypre_option! {
    /// Number of classical Gram-Schmidt passes per orthogonalization in
    /// COGMRES (`HYPRE_COGMRESSetCGS`).
    GramSchmidtPasses {
        /// One pass.
        One = 1,
        /// Two passes (reorthogonalization), more robust.
        Two = 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_round_trip_through_hypre_values() -> HypreResult<()> {
        for option in [CoarsenType::Cljp, CoarsenType::Hmis, CoarsenType::CgcE] {
            assert_eq!(CoarsenType::from_hypre(HYPRE_Int::from(option))?, option);
        }
        assert_eq!(HYPRE_Int::from(InterpType::ExtendedPlusI), 6);
        assert_eq!(HYPRE_Int::from(Smoother::L1Jacobi), 18);
        assert_eq!(HYPRE_Int::from(CycleType::W), 2);
        assert_eq!(
            ConvergeType::from_hypre(1)?,
            ConvergeType::RelativeToInitialResidual
        );
        assert_eq!(GramSchmidtPasses::from_hypre(2)?, GramSchmidtPasses::Two);
        Ok(())
    }

    #[test]
    fn unknown_hypre_values_are_errors() {
        assert!(matches!(
            CoarsenType::from_hypre(2),
            Err(HypreError::UnknownOptionValue(2))
        ));
        assert!(matches!(
            CycleType::from_hypre(0),
            Err(HypreError::UnknownOptionValue(0))
        ));
    }
}
