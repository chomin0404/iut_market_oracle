"""ROIC Bayesian Network evaluation endpoint."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from api.schemas.roic import ROICEvaluateRequest, ROICEvaluateResponse
from bayesian.roic_net import ROICBayesNet
from schemas.roic import ROICEvidence

router = APIRouter()
_logger = logging.getLogger(__name__)

_EXAMPLES = {
    "blackstone": {
        "summary": "BX (Blackstone Inc.) — high-quality alt-asset manager",
        "value": {
            "industry_competitiveness": "high",
            "management_efficiency": "high",
            "macro_environment": "medium",
            "pricing_power": "high",
            "capital_allocation": "high",
            "equivalent_sample_size": 10.0,
        },
    },
    "prior_marginal": {
        "summary": "Prior marginal P(roic_level) — no evidence",
        "value": {"equivalent_sample_size": 10.0},
    },
}


@router.post(
    "/evaluate",
    response_model=ROICEvaluateResponse,
    summary="ROIC Bayesian evaluation",
    description=(
        "Compute P(roic_level | evidence) via Variable Elimination on a 6-node DAG. "
        "Omit any node to query its marginal. Returns posterior distribution, "
        "expected ROIC score ∈ [0,1], and an A/B/C grade."
    ),
)
def evaluate_roic(
    body: Annotated[ROICEvaluateRequest, Body(openapi_examples=_EXAMPLES)],  # type: ignore[arg-type]
) -> ROICEvaluateResponse:
    try:
        net = ROICBayesNet(equivalent_sample_size=body.equivalent_sample_size)
        evidence = ROICEvidence(
            industry_competitiveness=body.industry_competitiveness,
            management_efficiency=body.management_efficiency,
            macro_environment=body.macro_environment,
            pricing_power=body.pricing_power,
            capital_allocation=body.capital_allocation,
        )
        result = net.evaluate(evidence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    score = result.expected_roic_score
    if score >= 0.65:
        grade = "A (High ROIC quality)"
    elif score >= 0.40:
        grade = "B (Medium ROIC quality)"
    else:
        grade = "C (Low ROIC quality)"

    _logger.info(
        "roic_eval dominant=%s score=%.4f grade=%s evidence=%s",
        result.dominant_state,
        score,
        grade,
        result.evidence,
    )

    return ROICEvaluateResponse(
        evidence=result.evidence,
        roic_posterior=result.roic_posterior,
        expected_roic_score=score,
        dominant_state=result.dominant_state,
        grade=grade,
    )
