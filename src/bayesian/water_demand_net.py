"""福岡市水道需要予測ベイジアンネット。

DAG
---
    season ──→ temperature ──→ demand_level
    day_type ──→ usage_tier ──→ demand_level

ノード定義
----------
season      : 季節 (spring / summer / autumn / winter)
day_type    : 曜日種別 (weekday / holiday)
temperature : 気温水準 (low / normal / high)   — season に条件付き
usage_tier  : 使用量帯 (low / mid / high)       — day_type に条件付き
demand_level: 需要水準 (low / normal / high)    — 予測対象

CPT の根拠
----------
すべての確率値はドメイン知識に基づく事前分布である。
実測データが得られた場合は各 CPT を上書きすることで更新できる。

水道需要への主な寄与:
  - 夏季・高温 → 冷房・散水・入浴頻度増加 → 需要増
  - 休日       → 在宅時間増加              → 使用量帯が上方シフト
  - 冬季・低温 → 節水傾向                  → 需要減

出典(料金データ): data/raw/fukuoka_water_basic_rates.csv
                  data/raw/fukuoka_water_volume_rates.csv
"""

from __future__ import annotations

import numpy as np

from bayesian.network import BayesianNetwork

# ---------------------------------------------------------------------------
# 事前確率 (root ノード)
# ---------------------------------------------------------------------------

# 季節: 等確率（福岡の場合、季節長はほぼ均等と仮定）
_P_SEASON: list[float] = [0.25, 0.25, 0.25, 0.25]

# 曜日種別: 週5日 / 2日の比率
_P_DAY_TYPE: list[float] = [5 / 7, 2 / 7]

# ---------------------------------------------------------------------------
# 条件付き確率表 (CPT)
# ---------------------------------------------------------------------------

# P(temperature | season)
# rows: spring / summer / autumn / winter
# cols: low / normal / high
_CPT_TEMPERATURE: dict[tuple[str, ...], list[float]] = {
    ("spring",): [0.20, 0.65, 0.15],
    ("summer",): [0.05, 0.25, 0.70],
    ("autumn",): [0.30, 0.60, 0.10],
    ("winter",): [0.70, 0.28, 0.02],
}

# P(usage_tier | day_type)
# rows: weekday / holiday
# cols: low / mid / high
_CPT_USAGE_TIER: dict[tuple[str, ...], list[float]] = {
    ("weekday",): [0.20, 0.60, 0.20],
    ("holiday",): [0.10, 0.40, 0.50],
}

# P(demand_level | temperature, usage_tier)
# rows: (temperature, usage_tier) の全組み合わせ 3×3=9
# cols: low / normal / high
_CPT_DEMAND_LEVEL: dict[tuple[str, ...], list[float]] = {
    ("low", "low"): [0.70, 0.25, 0.05],
    ("low", "mid"): [0.50, 0.40, 0.10],
    ("low", "high"): [0.20, 0.55, 0.25],
    ("normal", "low"): [0.30, 0.55, 0.15],
    ("normal", "mid"): [0.15, 0.60, 0.25],
    ("normal", "high"): [0.05, 0.45, 0.50],
    ("high", "low"): [0.10, 0.40, 0.50],
    ("high", "mid"): [0.05, 0.30, 0.65],
    ("high", "high"): [0.02, 0.18, 0.80],
}

# ---------------------------------------------------------------------------
# ファクトリ関数
# ---------------------------------------------------------------------------


def build_fukuoka_water_demand_net() -> BayesianNetwork:
    """福岡市水道需要予測ネットを構築して返す。

    Returns
    -------
    BayesianNetwork
        CPT 設定済みのネット。``posterior("demand_level")`` で需要水準の
        事後分布を取得できる。

    Examples
    --------
    >>> net = build_fukuoka_water_demand_net()
    >>> net.observe("season", "summer")
    >>> net.observe("day_type", "holiday")
    >>> net.posterior("demand_level")
    {'low': ..., 'normal': ..., 'high': ...}
    """
    net = BayesianNetwork()

    # ノード登録
    net.add_node("season", states=["spring", "summer", "autumn", "winter"])
    net.add_node("day_type", states=["weekday", "holiday"])
    net.add_node("temperature", states=["low", "normal", "high"])
    net.add_node("usage_tier", states=["low", "mid", "high"])
    net.add_node("demand_level", states=["low", "normal", "high"])

    # エッジ定義
    net.add_edge("season", "temperature")
    net.add_edge("day_type", "usage_tier")
    net.add_edge("temperature", "demand_level")
    net.add_edge("usage_tier", "demand_level")

    # 事前確率
    net.set_prior("season", _P_SEASON)
    net.set_prior("day_type", _P_DAY_TYPE)

    # CPT
    net.set_cpt("temperature", _CPT_TEMPERATURE)
    net.set_cpt("usage_tier", _CPT_USAGE_TIER)
    net.set_cpt("demand_level", _CPT_DEMAND_LEVEL)

    return net


def calibrate_usage_tier_cpt(
    net: BayesianNetwork,
    *,
    weekday_counts: list[float] | None = None,
    holiday_counts: list[float] | None = None,
    equivalent_sample_size: float = 20.0,
) -> BayesianNetwork:
    """Re-calibrate the usage_tier CPT with observed day-type usage data.

    Uses Dirichlet-Multinomial conjugate update:

        α  = current CPT row × equivalent_sample_size   (prior strength)
        posterior mean = (α + counts) / Σ(α + counts)

    The default weekday counts shift the symmetric prior [0.20, 0.60, 0.20]
    toward the empirically observed asymmetric distribution, improving the
    discriminative power of ``day_type=weekday``.

    Parameters
    ----------
    net:
        Network built by :func:`build_fukuoka_water_demand_net`.
    weekday_counts:
        Observed counts ``[low, mid, high]`` for weekdays.
        Default: ``[28, 58, 14]`` (synthetic estimate; weekday low-usage
        is higher and high-usage is lower than the symmetric prior implies).
    holiday_counts:
        Observed counts ``[low, mid, high]`` for holidays.
        Default: ``[10, 40, 50]`` (consistent with prior — no update needed).
    equivalent_sample_size:
        Prior strength in pseudocount units per CPT row. Default 20.0.

    Returns
    -------
    BayesianNetwork
        The same network with ``usage_tier`` CPT updated in-place.
    """
    if weekday_counts is None:
        # Shift the symmetric prior [0.20, 0.60, 0.20]:
        #   weekday high-usage is lower (commuters away from home)
        #   weekday low-usage is higher (fewer people home midday)
        weekday_counts = [28.0, 58.0, 14.0]
    if holiday_counts is None:
        holiday_counts = [10.0, 40.0, 50.0]

    net.init_dirichlet("usage_tier", equivalent_sample_size=equivalent_sample_size)

    # CPT shape: (2, 3)  — rows: weekday/holiday, cols: low/mid/high
    counts = np.array([weekday_counts, holiday_counts], dtype=float)
    net.accumulate_counts("usage_tier", counts)
    net.apply_dirichlet_posterior("usage_tier")

    return net
