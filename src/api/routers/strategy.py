"""FastAPI router for Strategy Twin (T1700).

Endpoint:
    POST /strategy/run  →  StrategyTwinReport
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.strategy import StrategyRunRequest
from schemas import StrategyTwinReport
from strategy_twin.twin import StrategyTwin, StrategyTwinConfig

router = APIRouter()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=StrategyTwinReport)
def run_strategy_twin(request: StrategyRunRequest) -> StrategyTwinReport:
    """Run the Strategy Twin decision engine.

    Returns a consolidated report including:
    - SOTP valuation (moat-adjusted DCF per business unit)
    - Black-Litterman posterior returns
    - Reverse DCF viability conditions (g*, s*, m*)
    - Causal ATE estimates from macro → FCF
    - Plain-language verdict
    """
    try:
        config = StrategyTwinConfig(
            business_units=request.business_units,
            macro=request.macro,
            views=request.views,
            causal_edges=request.causal_edges,
            target_ev=request.target_ev,
            net_debt=request.net_debt,
            current_market_share=request.current_market_share,
            current_fcf_margin=request.current_fcf_margin,
            causal_causes=request.causal_causes,
            causal_effects=request.causal_effects,
        )
        twin = StrategyTwin(config)
        return twin.run()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
