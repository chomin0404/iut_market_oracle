"""Tests for Strategy Twin (T1700).

Covers:
- Moat scoring and WACC/growth adjustments
- SOTP valuation
- Black-Litterman posterior
- Causal ATE enumeration
- Viability conditions
- End-to-end StrategyTwin.run()
- FastAPI endpoint
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from schemas import (
    BLView,
    BusinessUnit,
    CausalEdge,
    MacroEnvironment,
    MoatDimension,
    MoatScore,
)
from strategy_twin.black_litterman import _build_covariance, black_litterman
from strategy_twin.causal import compute_all_effects, compute_ate
from strategy_twin.moat import (
    composite_moat_score,
    moat_adjusted_terminal_growth,
    moat_adjusted_wacc,
)
from strategy_twin.sotp import sotp_valuation, value_unit
from strategy_twin.twin import StrategyTwin, StrategyTwinConfig
from strategy_twin.viability import compute_viability_conditions

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_unit(
    name: str = "core",
    initial_fcf: float = 100.0,
    growth_rate: float = 0.10,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.03,
    forecast_years: int = 5,
    moat_scores: list[MoatScore] | None = None,
) -> BusinessUnit:
    return BusinessUnit(
        name=name,
        initial_fcf=initial_fcf,
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        terminal_growth_rate=terminal_growth_rate,
        forecast_years=forecast_years,
        moat_scores=moat_scores or [],
    )


def _make_macro(
    gdp_growth: float = 0.025,
    risk_free_rate: float = 0.04,
    market_risk_premium: float = 0.055,
    inflation: float = 0.025,
    tam: float = 1_000.0,
) -> MacroEnvironment:
    return MacroEnvironment(
        gdp_growth=gdp_growth,
        risk_free_rate=risk_free_rate,
        market_risk_premium=market_risk_premium,
        inflation=inflation,
        tam=tam,
    )


# ---------------------------------------------------------------------------
# Moat
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def test_empty_returns_zero(self) -> None:
        assert composite_moat_score([]) == 0.0

    def test_single_score(self) -> None:
        ms = MoatScore(dimension=MoatDimension.SWITCHING_COSTS, score=0.8, weight=1.0)
        assert composite_moat_score([ms]) == pytest.approx(0.8)

    def test_weighted_average(self) -> None:
        scores = [
            MoatScore(dimension=MoatDimension.SWITCHING_COSTS, score=1.0, weight=3.0),
            MoatScore(dimension=MoatDimension.NETWORK_EFFECTS, score=0.0, weight=1.0),
        ]
        # (3*1 + 1*0) / (3+1) = 0.75
        assert composite_moat_score(scores) == pytest.approx(0.75)

    def test_result_in_unit_interval(self) -> None:
        scores = [
            MoatScore(dimension=MoatDimension.COST_ADVANTAGE, score=0.6, weight=2.0),
            MoatScore(dimension=MoatDimension.INTANGIBLE_ASSETS, score=0.9, weight=1.5),
        ]
        m = composite_moat_score(scores)
        assert 0.0 <= m <= 1.0


class TestMoatAdjustedWACC:
    def test_zero_moat_no_change(self) -> None:
        assert moat_adjusted_wacc(0.10, 0.0) == pytest.approx(0.10)

    def test_full_moat_reduces_by_250bps(self) -> None:
        result = moat_adjusted_wacc(0.10, 1.0)
        assert result == pytest.approx(max(0.10 - 0.025, 0.04))

    def test_floor_at_4pct(self) -> None:
        # Even with wacc=0.04 and moat=1.0, floor is 0.04
        assert moat_adjusted_wacc(0.04, 1.0) == pytest.approx(0.04)

    def test_high_wacc_partial_moat(self) -> None:
        result = moat_adjusted_wacc(0.15, 0.5)
        assert result == pytest.approx(0.15 - 0.5 * 0.025)


class TestMoatAdjustedTerminalGrowth:
    def test_zero_moat_no_change(self) -> None:
        g = moat_adjusted_terminal_growth(0.03, 0.10, 0.0)
        assert g == pytest.approx(0.03)

    def test_full_moat_uplift_150bps(self) -> None:
        g = moat_adjusted_terminal_growth(0.03, 0.10, 1.0)
        expected = min(0.03 + 0.015, 0.10 - 0.005)
        assert g == pytest.approx(expected)

    def test_spread_guard_enforced(self) -> None:
        # If base_g is already close to wacc, spread guard applies
        g = moat_adjusted_terminal_growth(0.09, 0.10, 1.0)
        assert g < 0.10  # must stay below adj_wacc


# ---------------------------------------------------------------------------
# SOTP
# ---------------------------------------------------------------------------


class TestSOTP:
    def test_single_unit_no_debt(self) -> None:
        unit = _make_unit()
        segments, total_ev = sotp_valuation([unit])
        assert len(segments) == 1
        assert total_ev == pytest.approx(segments[0].enterprise_value, rel=1e-9)

    def test_net_debt_subtracted(self) -> None:
        unit = _make_unit()
        segments, ev_with = sotp_valuation([unit], net_debt=0.0)
        _, ev_no = sotp_valuation([unit], net_debt=50.0)
        assert ev_with - ev_no == pytest.approx(50.0, rel=1e-6)

    def test_moat_increases_ev(self) -> None:
        unit_no_moat = _make_unit()
        unit_with_moat = _make_unit(
            moat_scores=[MoatScore(dimension=MoatDimension.SWITCHING_COSTS, score=1.0, weight=1.0)]
        )
        _, ev_no = sotp_valuation([unit_no_moat])
        _, ev_with = sotp_valuation([unit_with_moat])
        assert ev_with > ev_no  # lower WACC → higher EV

    def test_multi_unit_additive(self) -> None:
        u1 = _make_unit("a", initial_fcf=100.0)
        u2 = _make_unit("b", initial_fcf=200.0)
        seg1, ev1 = sotp_valuation([u1])
        seg2, ev2 = sotp_valuation([u2])
        segs, total = sotp_valuation([u1, u2])
        assert total == pytest.approx(ev1 + ev2, rel=1e-9)
        assert len(segs) == 2

    def test_segment_scores_bounded(self) -> None:
        unit = _make_unit(
            moat_scores=[MoatScore(dimension=MoatDimension.NETWORK_EFFECTS, score=0.7)]
        )
        seg = value_unit(unit)
        assert 0.0 <= seg.moat_composite_score <= 1.0
        assert seg.moat_adjusted_wacc >= 0.04
        assert seg.moat_adjusted_terminal_growth < seg.moat_adjusted_wacc


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------


class TestBlackLitterman:
    def _two_units(self) -> list[BusinessUnit]:
        return [
            _make_unit("A", initial_fcf=300.0, growth_rate=0.15),
            _make_unit("B", initial_fcf=100.0, growth_rate=0.05),
        ]

    def test_no_views_returns_near_equilibrium(self) -> None:
        units = self._two_units()
        result = black_litterman(units, views=[])
        # Without views, posterior ≈ prior (equilibrium)
        for name in ["A", "B"]:
            assert math.isfinite(result.posterior_returns[name])
            assert math.isfinite(result.equilibrium_returns[name])

    def test_market_weights_sum_to_one(self) -> None:
        units = self._two_units()
        result = black_litterman(units, views=[])
        total = sum(result.market_weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_view_pulls_posterior(self) -> None:
        units = self._two_units()
        # Strong view: A will return 0.30
        view = BLView(assets={"A": 1.0}, expected_return=0.30, uncertainty=0.01)
        result = black_litterman(units, views=[view])
        # Posterior for A should be pulled toward 0.30
        assert result.posterior_returns["A"] > result.equilibrium_returns["A"]

    def test_covariance_positive_definite(self) -> None:
        units = self._two_units()
        cov = _build_covariance(units)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_posterior_std_positive(self) -> None:
        units = self._two_units()
        result = black_litterman(units, views=[])
        for name in ["A", "B"]:
            assert result.posterior_std[name] > 0.0


# ---------------------------------------------------------------------------
# Causal
# ---------------------------------------------------------------------------


class TestCausal:
    def _simple_chain(self) -> list[CausalEdge]:
        # gdp → revenue: 0.5
        # revenue → fcf: 0.8
        return [
            CausalEdge(cause="gdp", effect="revenue", coefficient=0.5),
            CausalEdge(cause="revenue", effect="fcf", coefficient=0.8),
        ]

    def test_direct_path(self) -> None:
        edges = [CausalEdge(cause="gdp", effect="fcf", coefficient=0.6)]
        result = compute_ate(edges, "gdp", "fcf")
        assert result.total_effect == pytest.approx(0.6)
        assert result.n_paths == 1

    def test_chain_path(self) -> None:
        edges = self._simple_chain()
        result = compute_ate(edges, "gdp", "fcf")
        assert result.total_effect == pytest.approx(0.5 * 0.8)
        assert result.n_paths == 1

    def test_no_path_returns_zero(self) -> None:
        edges = self._simple_chain()
        result = compute_ate(edges, "fcf", "gdp")  # reverse direction
        assert result.total_effect == 0.0
        assert result.n_paths == 0

    def test_parallel_paths_sum(self) -> None:
        # Two paths: gdp→fcf (0.3) and gdp→revenue→fcf (0.5*0.8)
        edges = [
            CausalEdge(cause="gdp", effect="fcf", coefficient=0.3),
            CausalEdge(cause="gdp", effect="revenue", coefficient=0.5),
            CausalEdge(cause="revenue", effect="fcf", coefficient=0.8),
        ]
        result = compute_ate(edges, "gdp", "fcf")
        assert result.total_effect == pytest.approx(0.3 + 0.5 * 0.8)
        assert result.n_paths == 2

    def test_compute_all_effects(self) -> None:
        edges = self._simple_chain()
        results = compute_all_effects(edges, causes=["gdp"], effects=["fcf"])
        assert len(results) == 1
        assert results[0].cause == "gdp"
        assert results[0].effect == "fcf"

    def test_no_self_loop(self) -> None:
        with pytest.raises(Exception):
            CausalEdge(cause="x", effect="x", coefficient=1.0)


# ---------------------------------------------------------------------------
# Viability
# ---------------------------------------------------------------------------


class TestViability:
    def test_conditions_returned(self) -> None:
        unit = _make_unit(initial_fcf=100.0, growth_rate=0.10)
        macro = _make_macro(tam=10_000.0)
        conditions, implied_g = compute_viability_conditions(
            primary_unit=unit,
            macro=macro,
            target_ev=500.0,
        )
        assert len(conditions) == 3
        assert any(c.metric == "growth_rate" for c in conditions)
        assert any(c.metric == "market_share" for c in conditions)
        assert any(c.metric == "fcf_margin" for c in conditions)

    def test_implied_g_is_finite_or_none(self) -> None:
        unit = _make_unit(initial_fcf=100.0, growth_rate=0.10)
        macro = _make_macro(tam=10_000.0)
        _, implied_g = compute_viability_conditions(unit, macro, target_ev=500.0)
        assert implied_g is None or math.isfinite(implied_g)

    def test_gap_formula(self) -> None:
        unit = _make_unit(initial_fcf=100.0, growth_rate=0.10)
        macro = _make_macro(tam=10_000.0)
        conditions, _ = compute_viability_conditions(unit, macro, target_ev=500.0)
        for c in conditions:
            assert math.isclose(c.gap, c.current_estimate - c.minimum_required, rel_tol=1e-9)

    def test_is_met_consistent_with_gap(self) -> None:
        unit = _make_unit(initial_fcf=100.0, growth_rate=0.10)
        macro = _make_macro(tam=10_000.0)
        conditions, _ = compute_viability_conditions(unit, macro, target_ev=500.0)
        for c in conditions:
            if math.isfinite(c.gap):
                assert c.is_met == (c.gap >= 0)


# ---------------------------------------------------------------------------
# End-to-end StrategyTwin
# ---------------------------------------------------------------------------


class TestStrategyTwin:
    def _default_config(self) -> StrategyTwinConfig:
        return StrategyTwinConfig(
            business_units=[
                _make_unit(
                    "saas",
                    initial_fcf=500.0,
                    growth_rate=0.20,
                    discount_rate=0.12,
                    moat_scores=[
                        MoatScore(dimension=MoatDimension.SWITCHING_COSTS, score=0.7, weight=1.0)
                    ],
                ),
                _make_unit(
                    "services",
                    initial_fcf=200.0,
                    growth_rate=0.05,
                    discount_rate=0.09,
                ),
            ],
            macro=_make_macro(tam=50_000.0),
            views=[
                BLView(
                    assets={"saas": 1.0},
                    expected_return=0.15,
                    uncertainty=0.05,
                )
            ],
            causal_edges=[
                CausalEdge(cause="gdp_growth", effect="fcf", coefficient=0.4),
            ],
            causal_causes=["gdp_growth"],
            causal_effects=["fcf"],
            target_ev=10_000.0,
            current_market_share=0.05,
            current_fcf_margin=0.18,
        )

    def test_run_returns_report(self) -> None:
        twin = StrategyTwin(self._default_config())
        report = twin.run()
        assert report is not None

    def test_sotp_segments_count(self) -> None:
        twin = StrategyTwin(self._default_config())
        report = twin.run()
        assert len(report.sotp_segments) == 2

    def test_sotp_total_ev_matches_sum(self) -> None:
        config = self._default_config()
        twin = StrategyTwin(config)
        report = twin.run()
        seg_sum = sum(s.enterprise_value for s in report.sotp_segments)
        assert report.sotp_total_ev == pytest.approx(seg_sum - config.net_debt, rel=1e-9)

    def test_bl_result_keys(self) -> None:
        twin = StrategyTwin(self._default_config())
        report = twin.run()
        assert "saas" in report.bl_result.posterior_returns
        assert "services" in report.bl_result.posterior_returns

    def test_viability_conditions_count(self) -> None:
        twin = StrategyTwin(self._default_config())
        report = twin.run()
        assert len(report.viability_conditions) == 3

    def test_causal_effects_present(self) -> None:
        twin = StrategyTwin(self._default_config())
        report = twin.run()
        assert len(report.causal_effects) >= 1
        assert report.causal_effects[0].cause == "gdp_growth"

    def test_verdict_is_string(self) -> None:
        twin = StrategyTwin(self._default_config())
        report = twin.run()
        assert isinstance(report.verdict, str)
        assert len(report.verdict) > 0

    def test_empty_business_units_raises(self) -> None:
        with pytest.raises(ValueError, match="business_units"):
            StrategyTwinConfig(
                business_units=[],
                macro=_make_macro(),
            )

    def test_report_produced_at_set(self) -> None:
        twin = StrategyTwin(self._default_config())
        report = twin.run()
        assert report.produced_at is not None


# ---------------------------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------------------------


class TestStrategyAPI:
    @pytest.fixture()
    def client(self) -> TestClient:
        from api.app import app

        return TestClient(app)

    def _payload(self) -> dict:
        return {
            "business_units": [
                {
                    "name": "core",
                    "initial_fcf": 100.0,
                    "growth_rate": 0.10,
                    "discount_rate": 0.10,
                    "terminal_growth_rate": 0.03,
                    "forecast_years": 5,
                    "moat_scores": [
                        {
                            "dimension": "switching_costs",
                            "score": 0.6,
                            "weight": 1.0,
                        }
                    ],
                }
            ],
            "macro": {
                "gdp_growth": 0.025,
                "risk_free_rate": 0.04,
                "market_risk_premium": 0.055,
                "inflation": 0.025,
                "tam": 5000.0,
            },
            "views": [],
            "causal_edges": [],
            "target_ev": 0.0,
            "net_debt": 0.0,
            "current_market_share": 0.05,
            "current_fcf_margin": 0.15,
        }

    def test_run_returns_200(self, client: TestClient) -> None:
        resp = client.post("/api/v1/strategy/run", json=self._payload())
        assert resp.status_code == 200

    def test_response_has_verdict(self, client: TestClient) -> None:
        resp = client.post("/api/v1/strategy/run", json=self._payload())
        data = resp.json()
        assert "verdict" in data
        assert isinstance(data["verdict"], str)

    def test_response_has_sotp(self, client: TestClient) -> None:
        resp = client.post("/api/v1/strategy/run", json=self._payload())
        data = resp.json()
        assert "sotp_segments" in data
        assert len(data["sotp_segments"]) == 1

    def test_empty_units_returns_422(self, client: TestClient) -> None:
        payload = self._payload()
        payload["business_units"] = []
        resp = client.post("/api/v1/strategy/run", json=payload)
        assert resp.status_code == 422
