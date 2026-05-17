"""Tests for src/bayesian/water_demand_net.py — 福岡市水道需要予測ネット。"""

from __future__ import annotations

import pytest

from bayesian.water_demand_net import (
    _CPT_DEMAND_LEVEL,
    _CPT_TEMPERATURE,
    _CPT_USAGE_TIER,
    build_fukuoka_water_demand_net,
)

# ---------------------------------------------------------------------------
# ネット構築
# ---------------------------------------------------------------------------


class TestBuild:
    def test_build_returns_network(self):
        net = build_fukuoka_water_demand_net()
        assert net is not None

    def test_node_states(self):
        net = build_fukuoka_water_demand_net()
        assert net.node_states("season") == ["spring", "summer", "autumn", "winter"]
        assert net.node_states("day_type") == ["weekday", "holiday"]
        assert net.node_states("temperature") == ["low", "normal", "high"]
        assert net.node_states("usage_tier") == ["low", "mid", "high"]
        assert net.node_states("demand_level") == ["low", "normal", "high"]

    def test_topological_order(self):
        net = build_fukuoka_water_demand_net()
        order = net.topological_order()
        assert order.index("season") < order.index("temperature")
        assert order.index("day_type") < order.index("usage_tier")
        assert order.index("temperature") < order.index("demand_level")
        assert order.index("usage_tier") < order.index("demand_level")


# ---------------------------------------------------------------------------
# CPT の確率値検証（事前条件）
# ---------------------------------------------------------------------------


class TestCPTSanity:
    @pytest.mark.parametrize("season", ["spring", "summer", "autumn", "winter"])
    def test_temperature_cpt_rows_sum_to_one(self, season: str):
        row = _CPT_TEMPERATURE[(season,)]
        assert abs(sum(row) - 1.0) < 1e-9, f"season={season}: {row}"

    @pytest.mark.parametrize("day", ["weekday", "holiday"])
    def test_usage_tier_cpt_rows_sum_to_one(self, day: str):
        row = _CPT_USAGE_TIER[(day,)]
        assert abs(sum(row) - 1.0) < 1e-9, f"day_type={day}: {row}"

    @pytest.mark.parametrize(
        "key",
        [
            ("low", "low"),
            ("low", "mid"),
            ("low", "high"),
            ("normal", "low"),
            ("normal", "mid"),
            ("normal", "high"),
            ("high", "low"),
            ("high", "mid"),
            ("high", "high"),
        ],
    )
    def test_demand_level_cpt_rows_sum_to_one(self, key: tuple[str, str]):
        row = _CPT_DEMAND_LEVEL[key]
        assert abs(sum(row) - 1.0) < 1e-9, f"key={key}: {row}"

    def test_demand_level_cpt_has_all_nine_rows(self):
        assert len(_CPT_DEMAND_LEVEL) == 9

    def test_temperature_cpt_has_four_rows(self):
        assert len(_CPT_TEMPERATURE) == 4

    def test_usage_tier_cpt_has_two_rows(self):
        assert len(_CPT_USAGE_TIER) == 2


# ---------------------------------------------------------------------------
# 周辺分布（エビデンスなし）
# ---------------------------------------------------------------------------


class TestMarginals:
    def test_demand_level_sums_to_one(self):
        net = build_fukuoka_water_demand_net()
        post = net.posterior("demand_level")
        assert sum(post.values()) == pytest.approx(1.0, abs=1e-9)

    def test_temperature_sums_to_one(self):
        net = build_fukuoka_water_demand_net()
        post = net.posterior("temperature")
        assert sum(post.values()) == pytest.approx(1.0, abs=1e-9)

    def test_usage_tier_sums_to_one(self):
        net = build_fukuoka_water_demand_net()
        post = net.posterior("usage_tier")
        assert sum(post.values()) == pytest.approx(1.0, abs=1e-9)

    def test_season_marginal_equals_prior(self):
        net = build_fukuoka_water_demand_net()
        post = net.posterior("season")
        for state in ["spring", "summer", "autumn", "winter"]:
            assert post[state] == pytest.approx(0.25, abs=1e-9)

    def test_weekday_marginal_equals_prior(self):
        """P(day_type=weekday) = 5/7."""
        net = build_fukuoka_water_demand_net()
        post = net.posterior("day_type")
        assert post["weekday"] == pytest.approx(5 / 7, abs=1e-9)


# ---------------------------------------------------------------------------
# 夏季エビデンス: 高需要が最大になること
# ---------------------------------------------------------------------------


class TestSummerDemand:
    def test_summer_holiday_high_demand_dominates(self):
        """夏季+休日 → high需要が最大確率になること。"""
        net = build_fukuoka_water_demand_net()
        net.observe("season", "summer")
        net.observe("day_type", "holiday")
        post = net.posterior("demand_level")
        assert post["high"] > post["normal"]
        assert post["high"] > post["low"]

    def test_summer_high_temperature_shifts_demand_up(self):
        """夏季+高温 → high需要 ≥ low需要。"""
        net = build_fukuoka_water_demand_net()
        net.observe("season", "summer")
        net.observe("temperature", "high")
        post = net.posterior("demand_level")
        assert post["high"] > post["low"]

    def test_summer_posterior_sums_to_one(self):
        net = build_fukuoka_water_demand_net()
        net.observe("season", "summer")
        post = net.posterior("demand_level")
        assert sum(post.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 冬季エビデンス: 低需要が最大になること
# ---------------------------------------------------------------------------


class TestWinterDemand:
    def test_winter_weekday_low_demand_dominates(self):
        """冬季+平日 → low需要が最大確率になること。"""
        net = build_fukuoka_water_demand_net()
        net.observe("season", "winter")
        net.observe("day_type", "weekday")
        post = net.posterior("demand_level")
        assert post["low"] > post["high"]

    def test_winter_low_temperature_lowers_demand(self):
        """冬季+低気温 → low需要 > high需要。"""
        net = build_fukuoka_water_demand_net()
        net.observe("season", "winter")
        net.observe("temperature", "low")
        post = net.posterior("demand_level")
        assert post["low"] > post["high"]


# ---------------------------------------------------------------------------
# 休日効果: 同じ季節・気温でも休日は需要が上昇すること
# ---------------------------------------------------------------------------


class TestHolidayEffect:
    def test_holiday_raises_high_demand_vs_weekday(self):
        """同一条件で holiday > weekday の high需要確率。"""
        net = build_fukuoka_water_demand_net()

        net.observe("season", "spring")
        net.observe("temperature", "normal")
        net.observe("day_type", "holiday")
        post_holiday = net.posterior("demand_level")

        net.reset_evidence()
        net.observe("season", "spring")
        net.observe("temperature", "normal")
        net.observe("day_type", "weekday")
        post_weekday = net.posterior("demand_level")

        assert post_holiday["high"] > post_weekday["high"]

    def test_holiday_lowers_low_demand_vs_weekday(self):
        net = build_fukuoka_water_demand_net()

        net.observe("day_type", "holiday")
        post_holiday = net.posterior("demand_level")

        net.reset_evidence()
        net.observe("day_type", "weekday")
        post_weekday = net.posterior("demand_level")

        assert post_holiday["low"] < post_weekday["low"]


# ---------------------------------------------------------------------------
# update() API
# ---------------------------------------------------------------------------


class TestUpdateAPI:
    def test_update_returns_demand_level(self):
        net = build_fukuoka_water_demand_net()
        result = net.update(
            evidence={"season": "summer", "day_type": "holiday"},
            queries=["demand_level"],
        )
        assert "demand_level" in result
        assert sum(result["demand_level"].values()) == pytest.approx(1.0, abs=1e-9)

    def test_update_does_not_mutate_evidence(self):
        net = build_fukuoka_water_demand_net()
        baseline = net.posterior("demand_level")
        net.update({"season": "summer"}, ["demand_level"])
        after = net.posterior("demand_level")
        for state in ["low", "normal", "high"]:
            assert after[state] == pytest.approx(baseline[state], abs=1e-12)

    def test_update_multi_query(self):
        net = build_fukuoka_water_demand_net()
        result = net.update(
            evidence={"season": "winter"},
            queries=["temperature", "demand_level"],
        )
        assert set(result.keys()) == {"temperature", "demand_level"}
