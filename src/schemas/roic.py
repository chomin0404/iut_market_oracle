"""ROIC evaluation schemas.

Discrete state vocabulary
-------------------------
すべてのノードが共有する 3 状態ラベル:
    "high"   — 高水準（良好）
    "medium" — 中水準（普通）
    "low"    — 低水準（不良）

macro_environment の意味:
    "high"   = 外部環境が良好（低金利・高成長・強い産業需要）
    "medium" = 中立的な外部環境
    "low"    = 逆風環境（高金利・景気後退・規制強化など）

ROIC 期待スコア計算式
--------------------
E[ROIC score] = Σ_s  STATE_SCORE[s] × P(roic_level = s | evidence)

STATE_SCORE: {"high": 1.0, "medium": 0.5, "low": 0.0}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ROICState = Literal["high", "medium", "low"]

# ---------------------------------------------------------------------------
# Constants (used by roic_net.py to build and query the network)
# ---------------------------------------------------------------------------

# State → numeric value for expected-score computation
STATE_SCORE: dict[str, float] = {"high": 1.0, "medium": 0.5, "low": 0.0}

# Weights for roic_level CPT score formula (must sum to 1.0)
ROIC_WEIGHT_PRICING_POWER: float = 0.40
ROIC_WEIGHT_CAPITAL_ALLOCATION: float = 0.40
ROIC_WEIGHT_MACRO_ENVIRONMENT: float = 0.20


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class ROICEvidence(BaseModel):
    """Evidence injection for a single ROIC network query.

    Fields set to ``None`` are treated as unobserved and marginalized out
    during Variable Elimination inference.

    Example
    -------
    >>> ev = ROICEvidence(industry_competitiveness="high", macro_environment="high")
    >>> ev.as_dict()
    {'industry_competitiveness': 'high', 'macro_environment': 'high'}
    """

    industry_competitiveness: ROICState | None = None
    management_efficiency: ROICState | None = None
    macro_environment: ROICState | None = None
    pricing_power: ROICState | None = None
    capital_allocation: ROICState | None = None

    def as_dict(self) -> dict[str, str]:
        """Return only observed (non-None) fields as a plain string dict."""
        raw: dict[str, Any] = self.model_dump(exclude_none=True)
        return {k: v for k, v in raw.items() if isinstance(v, str)}


# ---------------------------------------------------------------------------
# Observation (for Dirichlet CPT update)
# ---------------------------------------------------------------------------


class ROICObservation(BaseModel):
    """A single training observation for Dirichlet online CPT updates.

    All fields are optional.  Missing fields are silently skipped during
    count accumulation — they contribute no information to that node's CPT.

    Example
    -------
    >>> obs = ROICObservation(industry_competitiveness="high", roic_level="high")
    >>> obs.as_dict()
    {'industry_competitiveness': 'high', 'roic_level': 'high'}
    """

    industry_competitiveness: ROICState | None = None
    management_efficiency: ROICState | None = None
    macro_environment: ROICState | None = None
    pricing_power: ROICState | None = None
    capital_allocation: ROICState | None = None
    roic_level: ROICState | None = None

    def as_dict(self) -> dict[str, str]:
        """Return only observed (non-None) fields as a plain string dict."""
        raw: dict[str, Any] = self.model_dump(exclude_none=True)
        return {k: v for k, v in raw.items() if isinstance(v, str)}


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


class ROICEvaluation(BaseModel):
    """Output of one ROIC Bayesian network evaluation.

    Attributes
    ----------
    evidence:
        Observed node states used for inference (empty dict = prior marginal).
    roic_posterior:
        P(roic_level = s | evidence) for each state s ∈ {high, medium, low}.
        Values sum to 1.0.
    expected_roic_score:
        Posterior expected score in [0, 1]:
            E = Σ_s STATE_SCORE[s] × P(s | evidence)
        Interpretation guide (approximate):
            ≥ 0.65  → high ROIC quality
            0.40–0.65 → medium ROIC quality
            < 0.40  → low ROIC quality
    dominant_state:
        argmax_s P(roic_level = s | evidence).
    n_observations:
        Cumulative training observations accumulated via Dirichlet updates.
    """

    evidence: dict[str, str] = Field(description="Observed node states used for inference.")
    roic_posterior: dict[str, float] = Field(
        description="Posterior P(roic_level=s | evidence) for s in {high, medium, low}."
    )
    expected_roic_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Posterior expected ROIC score in [0, 1].",
    )
    dominant_state: ROICState = Field(description="Most probable ROIC state given evidence.")
    n_observations: int = Field(
        default=0,
        ge=0,
        description="Cumulative Dirichlet training observations.",
    )
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @model_validator(mode="after")
    def _check_posterior_sums_to_one(self) -> ROICEvaluation:
        total = sum(self.roic_posterior.values())
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"roic_posterior must sum to 1.0, got {total:.8f}")
        return self
