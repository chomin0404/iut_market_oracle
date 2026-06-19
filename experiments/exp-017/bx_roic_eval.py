"""BX (Blackstone Inc.) ROIC Bayesian Network Evaluation.

評価根拠
--------
BX は世界最大のオルタナティブ資産運用会社（AUM ~$1.1 兆、2024年末）。
軽資産ビジネスモデル（管理報酬+成功報酬）のため、伝統的 ROIC の
分子（NOPAT）は極めて大きく分母（投下資本）は小さい。
ただし LP への受託責任と GP コミットメント（通常 2-5% 自己出資）を
考慮した実質的な資本効率を評価する。

各ノードの評価根拠
------------------
industry_competitiveness = high
  - 参入障壁: 35年超の実績、グローバル LP ネットワーク、$1 兆 AUM のスケール
  - Preqin 調査でブラックストーンは PE/RE/クレジット全分野で Top-3 常連
  - 新規参入者が同等ブランドを構築するには 15-20 年かかると業界で言われる

management_efficiency = high
  - FRE マージン: ~55% (2023-2024 平均)
  - Fee-Related Earnings の CAGR: 過去 5 年間 ~20%
  - 人件費/AUM 比率が業界最低水準のひとつ
  - CEO グレイを中心とした後継計画も確立済み

macro_environment = medium
  - 2022-2024 の急速な利上げで PE ディール・商業不動産の Exit が減速
  - BREIT（不動産 REIT）の出金制限（2022-2023）がブランドへの一時的逆風
  - 2024H2 以降の利下げ転換で新規買収ファンド調達が回復傾向
  - 総合的に「中立やや下」= medium

pricing_power = high (直接観測可能)
  - 旗艦 PE ファンドの管理報酬 1.5% + 成功報酬 20% を全期間維持
  - 競合他社（Carlyle, KKR）との比較でも値下げ圧力を受けていない
  - BX の LP 引き留め率 ~95%（業界平均 ~80%）

capital_allocation = high (直接観測可能)
  - 過去最大の逆張り実績: 2009 年金融危機直後に Hilton を $26 億で取得 → $14 億超回収
  - COVID 後の物流 REIT (BioMed/BREIT 物流) 大量取得 → 評価額 3-4 倍
  - ヴィンテージ別 IRR: 最高実現ファンドは 40%超、全体加重平均 ~22%
"""

from __future__ import annotations

from bayesian.roic_net import ROICBayesNet
from schemas.roic import ROICEvidence, ROICObservation

# ---------------------------------------------------------------------------
# 1. 各ノードの証拠設定
# ---------------------------------------------------------------------------

EVIDENCE = ROICEvidence(
    industry_competitiveness="high",  # AUM スケール・参入障壁
    management_efficiency="high",  # FRE マージン 55%・CAGR 20%
    macro_environment="medium",  # 利上げ→利下げ転換期、不動産逆風残存
    pricing_power="high",  # 管理報酬 1.5%+20% 維持、LP 引留率 95%
    capital_allocation="high",  # 歴史的逆張り投資の実績
)

# ---------------------------------------------------------------------------
# 2. 過去実績からの Dirichlet 更新（BX ファンドの典型的サイクル観測）
# ---------------------------------------------------------------------------
# BX の過去ファンド（2005-2023 ヴィンテージ）に基づく観測: 概ね "high ROIC"
# 但し GFC 期（2007-2008）や BREIT 問題（2022-2023）は "medium" ケースも含む

HISTORICAL_OBSERVATIONS: list[ROICObservation] = [
    # 良好期（2010-2019 黄金期）: 低金利・M&A ブーム・IPO 市場活況
    *[
        ROICObservation(
            industry_competitiveness="high",
            management_efficiency="high",
            macro_environment="high",
            pricing_power="high",
            capital_allocation="high",
            roic_level="high",
        )
    ]
    * 12,
    # 中立期（2020-2021 初期 COVID・2022 利上げ転換直前）
    *[
        ROICObservation(
            industry_competitiveness="high",
            management_efficiency="high",
            macro_environment="medium",
            pricing_power="high",
            capital_allocation="high",
            roic_level="high",
        )
    ]
    * 6,
    # 逆風期（2022-2023 急激な利上げ・BREIT 出金制限）
    *[
        ROICObservation(
            industry_competitiveness="high",
            management_efficiency="high",
            macro_environment="low",
            pricing_power="high",
            capital_allocation="medium",  # Exit が遅延しキャッシュ効率が低下
            roic_level="medium",
        )
    ]
    * 4,
    # GFC 期（2007-2009）：業界全体の信用収縮
    *[
        ROICObservation(
            industry_competitiveness="medium",
            management_efficiency="high",
            macro_environment="low",
            pricing_power="medium",
            capital_allocation="high",  # 危機中の逆張り取得は高評価
            roic_level="medium",
        )
    ]
    * 2,
]

# ---------------------------------------------------------------------------
# 3. 実行
# ---------------------------------------------------------------------------


def main() -> None:
    net = ROICBayesNet(equivalent_sample_size=20.0)

    # ── ベースライン（事前分布のみ） ──────────────────────────────────────────
    prior_result = net.evaluate()
    print("=" * 65)
    print("BX (Blackstone Inc.) ROIC Bayesian Evaluation")
    print("=" * 65)

    print("\n[1] Prior marginal P(roic_level)  ← 証拠なし")
    _print_posterior(prior_result.roic_posterior)
    print(f"    Expected ROIC score : {prior_result.expected_roic_score:.4f}")
    print(f"    Dominant state      : {prior_result.dominant_state}")

    # ── 証拠注入後の推論 ─────────────────────────────────────────────────────
    evidence_result = net.evaluate(EVIDENCE)
    print("\n[2] Posterior P(roic_level | BX evidence)  ← 全ノードに証拠")
    _print_evidence(EVIDENCE)
    _print_posterior(evidence_result.roic_posterior)
    print(f"    Expected ROIC score : {evidence_result.expected_roic_score:.4f}")
    print(f"    Dominant state      : {evidence_result.dominant_state}")

    # ── 過去観測で Dirichlet 更新後 ───────────────────────────────────────────
    net.update_from_observations(HISTORICAL_OBSERVATIONS)
    updated_result = net.evaluate(EVIDENCE)
    print(f"\n[3] Posterior after {net.n_observations} historical observations")
    _print_posterior(updated_result.roic_posterior)
    print(f"    Expected ROIC score : {updated_result.expected_roic_score:.4f}")
    print(f"    Dominant state      : {updated_result.dominant_state}")

    # ── 感度分析（マクロ環境シナリオ） ──────────────────────────────────────
    print("\n[4] Sensitivity -- macro_environment scenario comparison")
    cols = "    {:<10s}  {:>8s}  {:>10s}  {:>8s}  {:>9s}".format(
        "macro", "P(high)", "P(medium)", "P(low)", "E[score]"
    )
    print(cols)
    print("    " + "-" * 54)
    for macro in ("high", "medium", "low"):
        ev = ROICEvidence(
            industry_competitiveness="high",
            management_efficiency="high",
            macro_environment=macro,  # type: ignore[arg-type]
            pricing_power="high",
            capital_allocation="high",
        )
        r = net.evaluate(ev)
        p = r.roic_posterior
        print(
            f"    {macro:10s}  {p['high']:>8.4f}  {p['medium']:>10.4f}  "
            f"{p['low']:>8.4f}  {r.expected_roic_score:>9.4f}"
        )

    # ── 総合判定 ─────────────────────────────────────────────────────────────
    score = updated_result.expected_roic_score
    if score >= 0.65:
        grade = "A  (High ROIC quality)"
    elif score >= 0.40:
        grade = "B  (Medium ROIC quality)"
    else:
        grade = "C  (Low ROIC quality)"

    print("\n" + "=" * 65)
    print(f"  総合 ROIC スコア : {score:.4f}  →  {grade}")
    print("=" * 65)


def _print_posterior(post: dict[str, float]) -> None:
    bar_width = 30
    for state in ("high", "medium", "low"):
        p = post[state]
        bar = "#" * int(p * bar_width)
        print(f"    {state:8s} {p:6.4f}  {bar}")


def _print_evidence(ev: ROICEvidence) -> None:
    d = ev.as_dict()
    for k, v in d.items():
        print(f"    {k:30s} = {v}")


if __name__ == "__main__":
    main()
