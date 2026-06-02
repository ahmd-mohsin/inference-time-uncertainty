# src/dad/__init__.py
from src.dad.claim_extractor import (
    MathClaim, SolutionProfile, profile_solution,
    extract_boxed_answer, canonicalize_value, extract_numeric_value,
)
from src.dad.disagreement_analyzer import (
    ClaimCluster, DisagreementMap, build_disagreement_map,
    build_claim_dag, compute_leverage, format_workspace, compute_entropy,
)
from src.dad.allocation import (
    AllocationConfig, allocate_round, update_rho, offline_water_fill,
)
from src.dad.dad_generator import DADGenerator, DADResult

__all__ = [
    "MathClaim", "SolutionProfile", "profile_solution",
    "extract_boxed_answer", "canonicalize_value", "extract_numeric_value",
    "ClaimCluster", "DisagreementMap", "build_disagreement_map",
    "build_claim_dag", "compute_leverage", "format_workspace", "compute_entropy",
    "AllocationConfig", "allocate_round", "update_rho", "offline_water_fill",
    "DADGenerator", "DADResult",
]