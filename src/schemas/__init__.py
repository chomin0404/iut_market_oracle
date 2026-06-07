"""Typed schemas shared across all research modules (T200–T1700).

This package replaces the monolithic schemas.py.
All public names are re-exported here so that existing
``from schemas import X`` statements continue to work unchanged.
"""

from .bayesian import (
    ClaimTag,
    Evidence,
    EvidenceKind,
    PosteriorSummary,
    PriorSpec,
)
from .entropy import (
    AlertType,
    EntropyAlert,
    EntropyReport,
)
from .exit import (
    ExitOption,
    ExitType,
    ExitValueSummary,
    TimingDistribution,
)
from .experiments import ExperimentMeta
from .gnss import (
    EpochReport,
    FaultClass,
    MCSimReport,
    MSRunResult,
    MSRunTrace,
    MSSimReport,
    ObservationEpoch,
    RecommendedAction,
    ResilienceTwinReport,
    RunResult,
    RunTrace,
    TwinRunReport,
)
from .graph import (
    EdgeMeta,
    GraphInput,
    NodeMeta,
    PortfolioMetrics,
)
from .market import (
    MarketEvolutionResult,
    MatroidLogConcavityResult,
    RegimeSwitchResult,
)
from .modelforge import (
    ForgeReport,
    TraceNode,
    TraceNodeType,
    VerificationCheck,
    VerificationReport,
    VerificationStatus,
)
from .models import (
    IdeaInput,
    ModelRecommendation,
    ModelRegistryEntry,
    ModelSpec,
    ParsedIdeaResponse,
    ProblemStructure,
)
from .strategy import (
    BLResult,
    BLView,
    BusinessUnit,
    CausalEdge,
    CausalEffect,
    MacroEnvironment,
    MoatDimension,
    MoatScore,
    SOTPSegment,
    StrategyTwinReport,
    ViabilityCondition,
)
from .twin import (
    DigitalTwinState,
    SimulationResult,
)
from .valuation import (
    AssumptionSet,
    ScenarioResult,
)
from .yield_twin import (
    DOERecommendation,
    ExperimentPoint,
    FactorSpec,
    YieldTwinReport,
)

__all__ = [
    # bayesian
    "ClaimTag",
    "Evidence",
    "EvidenceKind",
    "PosteriorSummary",
    "PriorSpec",
    # entropy
    "AlertType",
    "EntropyAlert",
    "EntropyReport",
    # experiments
    "ExperimentMeta",
    # exit
    "ExitOption",
    "ExitType",
    "ExitValueSummary",
    "TimingDistribution",
    # gnss
    "EpochReport",
    "FaultClass",
    "MCSimReport",
    "MSRunResult",
    "MSRunTrace",
    "MSSimReport",
    "ObservationEpoch",
    "RecommendedAction",
    "ResilienceTwinReport",
    "RunResult",
    "RunTrace",
    "TwinRunReport",
    # graph
    "EdgeMeta",
    "GraphInput",
    "NodeMeta",
    "PortfolioMetrics",
    # market
    "MarketEvolutionResult",
    "MatroidLogConcavityResult",
    "RegimeSwitchResult",
    # modelforge
    "ForgeReport",
    "TraceNode",
    "TraceNodeType",
    "VerificationCheck",
    "VerificationReport",
    "VerificationStatus",
    # models
    "IdeaInput",
    "ModelRecommendation",
    "ModelRegistryEntry",
    "ModelSpec",
    "ParsedIdeaResponse",
    "ProblemStructure",
    # strategy
    "BLResult",
    "BLView",
    "BusinessUnit",
    "CausalEdge",
    "CausalEffect",
    "MacroEnvironment",
    "MoatDimension",
    "MoatScore",
    "SOTPSegment",
    "StrategyTwinReport",
    "ViabilityCondition",
    # twin
    "DigitalTwinState",
    "SimulationResult",
    # valuation
    "AssumptionSet",
    "ScenarioResult",
    # yield_twin
    "DOERecommendation",
    "ExperimentPoint",
    "FactorSpec",
    "YieldTwinReport",
]
