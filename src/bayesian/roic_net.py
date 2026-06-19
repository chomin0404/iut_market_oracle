"""ROIC Evaluation Bayesian Network.

Network structure (DAG)
-----------------------

    industry_competitiveness ──→ pricing_power ──┐
                                                  ├──→ roic_level
    management_efficiency    ──→ capital_allocation ──┘
                                                  │
    macro_environment ────────────────────────────┘

All 6 nodes share the same 3-state vocabulary: ["high", "medium", "low"].

ROIC CPT formula
----------------
A weighted composite score drives the conditional probability of roic_level:

    score = w_pp × v(pricing_power)
          + w_ca × v(capital_allocation)
          + w_macro × v(macro_environment)

where v(high) = 1.0, v(medium) = 0.5, v(low) = 0.0, and weights are
defined in schemas.roic (w_pp = 0.40, w_ca = 0.40, w_macro = 0.20).

Score bands → P(roic_level = [high, medium, low]):

    score ≥ 0.70  →  [0.75, 0.20, 0.05]   (strong ROIC)
    score ≥ 0.50  →  [0.45, 0.40, 0.15]   (moderate-high ROIC)
    score ≥ 0.30  →  [0.20, 0.45, 0.35]   (moderate-low ROIC)
    score  < 0.30  →  [0.05, 0.25, 0.70]   (weak ROIC)

Dirichlet online update
-----------------------
Every node carries a Dirichlet prior initialised from its CPT:

    α = CPT_row × equivalent_sample_size

After calling update_from_observations(), the posterior mean is applied:

    θ̂ = (α + n) / Σ_row(α + n)

A larger equivalent_sample_size (ESS) makes the network more resistant to
updating from new data.  The default ESS = 10.0 corresponds to 10 virtual
observations per CPT row.
"""

from __future__ import annotations

from typing import cast

from bayesian.network import BayesianNetwork
from schemas.roic import (
    ROIC_WEIGHT_CAPITAL_ALLOCATION,
    ROIC_WEIGHT_MACRO_ENVIRONMENT,
    ROIC_WEIGHT_PRICING_POWER,
    STATE_SCORE,
    ROICEvaluation,
    ROICEvidence,
    ROICObservation,
    ROICState,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATES: list[str] = ["high", "medium", "low"]

# Root node priors P(node = [high, medium, low])
_PRIOR_INDUSTRY: list[float] = [0.30, 0.45, 0.25]  # 業界競争力
_PRIOR_MGMT: list[float] = [0.30, 0.45, 0.25]  # 経営効率
_PRIOR_MACRO: list[float] = [0.35, 0.40, 0.25]  # マクロ環境

# P(pricing_power | industry_competitiveness)
_CPT_PRICING_POWER: dict[tuple[str, ...], list[float]] = {
    ("high",): [0.70, 0.25, 0.05],
    ("medium",): [0.25, 0.55, 0.20],
    ("low",): [0.05, 0.30, 0.65],
}

# P(capital_allocation | management_efficiency)
_CPT_CAPITAL_ALLOC: dict[tuple[str, ...], list[float]] = {
    ("high",): [0.65, 0.30, 0.05],
    ("medium",): [0.20, 0.60, 0.20],
    ("low",): [0.05, 0.25, 0.70],
}

# Score thresholds → P(roic_level = [high, medium, low])
_SCORE_BANDS: list[tuple[float, list[float]]] = [
    (0.70, [0.75, 0.20, 0.05]),
    (0.50, [0.45, 0.40, 0.15]),
    (0.30, [0.20, 0.45, 0.35]),
    (0.00, [0.05, 0.25, 0.70]),
]


# ---------------------------------------------------------------------------
# CPT builders
# ---------------------------------------------------------------------------


def _roic_cpt_row(pp: str, ca: str, macro: str) -> list[float]:
    """Compute P(roic_level | pp, ca, macro) from the weighted score."""
    score = (
        ROIC_WEIGHT_PRICING_POWER * STATE_SCORE[pp]
        + ROIC_WEIGHT_CAPITAL_ALLOCATION * STATE_SCORE[ca]
        + ROIC_WEIGHT_MACRO_ENVIRONMENT * STATE_SCORE[macro]
    )
    for threshold, probs in _SCORE_BANDS:
        if score >= threshold:
            return probs
    return _SCORE_BANDS[-1][1]  # pragma: no cover — score=0.0 matches last band


def _build_roic_cpt() -> dict[tuple[str, ...], list[float]]:
    """Enumerate all 27 parent-state combinations for the roic_level CPT.

    Parent order (matches add_edge() call order in ROICBayesNet._build()):
        (pricing_power, capital_allocation, macro_environment)
    """
    cpt: dict[tuple[str, ...], list[float]] = {}
    for pp in _STATES:
        for ca in _STATES:
            for macro in _STATES:
                cpt[(pp, ca, macro)] = _roic_cpt_row(pp, ca, macro)
    return cpt


# ---------------------------------------------------------------------------
# ROICBayesNet
# ---------------------------------------------------------------------------


class ROICBayesNet:
    """Bayesian Network for ROIC evaluation with Dirichlet online updates.

    Workflow
    --------
    1. Instantiate::

           net = ROICBayesNet()

    2. Query prior marginal::

           result = net.evaluate()

    3. Query with evidence::

           result = net.evaluate(
               ROICEvidence(
                   industry_competitiveness="high",
                   macro_environment="high",
               )
           )
           # result.dominant_state → "high"
           # result.expected_roic_score → float in [0, 1]

    4. Update from observations::

           net.update_from_observations([
               ROICObservation(
                   industry_competitiveness="high",
                   management_efficiency="high",
                   roic_level="high",
               ),
               ...
           ])

    5. Re-query to obtain posteriors that reflect accumulated data.

    Parameters
    ----------
    equivalent_sample_size:
        Pseudocount total per CPT row for Dirichlet priors.
        Default 10.0 ≈ 10 virtual observations per row.
        Larger values → more inertia against new data.
    """

    def __init__(self, equivalent_sample_size: float = 10.0) -> None:
        if equivalent_sample_size <= 0:
            raise ValueError(
                f"equivalent_sample_size must be positive, got {equivalent_sample_size}"
            )
        self._ess = float(equivalent_sample_size)
        self._n_observations: int = 0
        self._net = BayesianNetwork()
        self._build()

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Construct DAG, assign CPTs, and initialise Dirichlet priors."""
        net = self._net

        # Root nodes
        net.add_node("industry_competitiveness", states=list(_STATES))
        net.add_node("management_efficiency", states=list(_STATES))
        net.add_node("macro_environment", states=list(_STATES))

        # Intermediate nodes
        net.add_node("pricing_power", states=list(_STATES))
        net.add_node("capital_allocation", states=list(_STATES))

        # Target node
        net.add_node("roic_level", states=list(_STATES))

        # Edges — order determines parent tuple order in CPTs
        net.add_edge("industry_competitiveness", "pricing_power")
        net.add_edge("management_efficiency", "capital_allocation")
        net.add_edge("pricing_power", "roic_level")
        net.add_edge("capital_allocation", "roic_level")
        net.add_edge("macro_environment", "roic_level")

        # Priors for root nodes
        net.set_prior("industry_competitiveness", _PRIOR_INDUSTRY)
        net.set_prior("management_efficiency", _PRIOR_MGMT)
        net.set_prior("macro_environment", _PRIOR_MACRO)

        # CPTs for intermediate and target nodes
        net.set_cpt("pricing_power", _CPT_PRICING_POWER)
        net.set_cpt("capital_allocation", _CPT_CAPITAL_ALLOC)
        net.set_cpt("roic_level", _build_roic_cpt())

        # Dirichlet priors for every node (enables online updates)
        for nid in net.topological_order():
            net.init_dirichlet(nid, equivalent_sample_size=self._ess)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def evaluate(self, evidence: ROICEvidence | None = None) -> ROICEvaluation:
        """Compute P(roic_level | evidence) via Variable Elimination.

        Parameters
        ----------
        evidence:
            Observed node states.  ``None`` or empty evidence queries the
            prior marginal P(roic_level).

        Returns
        -------
        ROICEvaluation
            Posterior distribution, expected score, and dominant state.

        Raises
        ------
        ValueError
            If an evidence state is not in the node's state space, or if
            the evidence combination has zero probability.
        """
        ev_dict: dict[str, str] = evidence.as_dict() if evidence is not None else {}
        posterior = self._net.update(ev_dict, ["roic_level"])["roic_level"]

        expected_score = sum(STATE_SCORE[s] * p for s, p in posterior.items())
        dominant_state: ROICState = cast(ROICState, max(posterior, key=lambda s: posterior[s]))

        return ROICEvaluation(
            evidence=ev_dict,
            roic_posterior=posterior,
            expected_roic_score=round(expected_score, 6),
            dominant_state=dominant_state,
            n_observations=self._n_observations,
        )

    # ------------------------------------------------------------------
    # Dirichlet online update
    # ------------------------------------------------------------------

    def update_from_observations(self, observations: list[ROICObservation]) -> None:
        """Accumulate counts and apply Dirichlet posterior mean to all CPTs.

        For each node that has an initialised Dirichlet prior, every
        observation whose record contains both the node's state and its
        parent states contributes a count of 1.  After accumulation the
        CPT is updated in-place via:

            θ̂ = (α₀ + n) / Σ_row(α₀ + n)

        Parameters
        ----------
        observations:
            List of :class:`~schemas.roic.ROICObservation` records.  Fields
            set to ``None`` are silently skipped.
        """
        if not observations:
            return

        raw: list[dict[str, str]] = [obs.as_dict() for obs in observations]
        self._net.observe_batch(raw)

        for nid in self._net.topological_order():
            self._net.apply_dirichlet_posterior(nid)

        self._n_observations += len(observations)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        """Total number of observations accumulated so far."""
        return self._n_observations

    @property
    def equivalent_sample_size(self) -> float:
        """ESS used to initialise Dirichlet priors."""
        return self._ess
