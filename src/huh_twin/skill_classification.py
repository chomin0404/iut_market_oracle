from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class SkillBasis(StrEnum):
    PROBABILISTIC_INFERENCE = "probabilistic_inference"
    VALUATION_AND_CAPITAL_ALLOCATION = "valuation_and_capital_allocation"
    INVERSE_PROBLEMS_AND_SIMULATION = "inverse_problems_and_simulation"
    OPTIMAL_EXPERIMENT_DESIGN = "optimal_experiment_design"
    STRATEGY_AND_IMPLEMENTATION = "strategy_and_implementation"


@dataclass(slots=True, frozen=True)
class ClassifiedSkill:
    name: str
    basis: SkillBasis
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ClassifiedSkill.name must be non-empty.")


EXPECTED_PROMPT_ITEMS: Final[tuple[str, ...]] = (
    "フェルミ推定",
    "ベイズ統計",
    "ベイジアンネット（ベイジアン最適化）",
    "確率分布（パレート分布）",
    "ガンマ・ポアソン分布",
    "ラプラス分布",
    "NBDモデル",
    "コーシー分布",
    "ディリクレ分布",
    "ブライト・ウィグナー分布",
    "レイリー・ジーンズ分布",
    "具体的な出口戦略（EXIT）の数理モデル",
    "DCF法",
    "マクロ経済指標",
    "市場規模",
    "競争優位性",
    "Moat（経済的な堀）",
    "Sensitivity Analysis（感応度分析）",
    "MCMCマルコフ連鎖モンテカルロ・シミュレーション",
    "エントロピー理論",
    "ビジネスモデル",
    "成長率10%",
    "割引率10%",
    "量子化学計算",
    "デザイン思考",
    "哲学",
    "確率思考の戦略論（森岡毅氏）",
    "デジタルツイン",
    "仮想空間シミュレーション",
    "Claudecodeの活用",
    "シグモイド曲線",
    "アナロジー",
    "ゲーム理論（ナッシュ均衡）",
    "SOTP（Sum-of-the-Parts：事業別評価合算）分析",
    "ブラック・リッターマン・モデル",
    "配当成長モデル（ゴードン・モデル）",
    "リバースDCF（Reverse Discounted Cash Flow）",
    "因果推論（Causal Inference）",
    "虚数時間",
    "フィールズ賞（歴代フィールズ賞受賞者の論文）",
    "Maryna Viazovska",
    "カバード・コール（Covered Call）",
    "フォークト関数（Voigt profile）",
    "Lax-Phillips基盤",
    "ラドン変換（Radon Transform）",
    "Lippmann-Schwinger方程式",
    "Adjoint State Method（随伴変数法）",
    "フィッシャー情報行列式を最大化（D-Optimality）",
    "感応度行列（Jacobian）",
    "特異摂動解析",
    "メトロポリス・ヘイスティングス法（MH法）",
    "ブライト・ウィグナー分布",
    "シャノン・エントロピー監視",
    "Hilbert空間",
    "量子共鳴解析",
    "成長断片化方程式の確率論的アプローチ",
    "レジーム転換モデル",
    "もつれ状態のコヒーレンス崩壊",
    "実験計画法（Optimal Design of Experiments）",
    "ディオファントス近似",
    "2DMAT",
    "極値関数をどう設計するか",
    "Lippmann-Schwinger",
)


SKILL_CLASSIFICATION: Final[tuple[ClassifiedSkill, ...]] = (
    ClassifiedSkill("フェルミ推定", SkillBasis.PROBABILISTIC_INFERENCE, "粗い事前分布と数量オーダーの初期化"),
    ClassifiedSkill("ベイズ統計", SkillBasis.PROBABILISTIC_INFERENCE, "事前・事後更新の中核"),
    ClassifiedSkill("ベイジアンネット（ベイジアン最適化）", SkillBasis.PROBABILISTIC_INFERENCE, "依存構造と逐次最適化"),
    ClassifiedSkill("確率分布（パレート分布）", SkillBasis.PROBABILISTIC_INFERENCE, "裾の厚い事象のモデリング"),
    ClassifiedSkill("ガンマ・ポアソン分布", SkillBasis.PROBABILISTIC_INFERENCE, "カウント過程と事後更新"),
    ClassifiedSkill("ラプラス分布", SkillBasis.PROBABILISTIC_INFERENCE, "ロバスト誤差と尖度"),
    ClassifiedSkill("NBDモデル", SkillBasis.PROBABILISTIC_INFERENCE, "反復回数と顧客行動の確率モデル"),
    ClassifiedSkill("コーシー分布", SkillBasis.PROBABILISTIC_INFERENCE, "極端値とロバスト推定"),
    ClassifiedSkill("ディリクレ分布", SkillBasis.PROBABILISTIC_INFERENCE, "混合比率とカテゴリ事前分布"),
    ClassifiedSkill("具体的な出口戦略（EXIT）の数理モデル", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "保有・売却・段階的EXITの定量比較"),
    ClassifiedSkill("DCF法", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "企業価値評価の中核"),
    ClassifiedSkill("マクロ経済指標", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "割引率・需要・資本コストの外生変数"),
    ClassifiedSkill("市場規模", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "売上上限と成長余地"),
    ClassifiedSkill("競争優位性", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "超過収益持続性の源泉"),
    ClassifiedSkill("Moat（経済的な堀）", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "持続的優位性の評価軸"),
    ClassifiedSkill("Sensitivity Analysis（感応度分析）", SkillBasis.STRATEGY_AND_IMPLEMENTATION, "価値ドライバーの識別"),
    ClassifiedSkill("MCMCマルコフ連鎖モンテカルロ・シミュレーション", SkillBasis.PROBABILISTIC_INFERENCE, "事後分布近似"),
    ClassifiedSkill("エントロピー理論", SkillBasis.OPTIMAL_EXPERIMENT_DESIGN, "情報量・不確実性の測定"),
    ClassifiedSkill("ビジネスモデル", SkillBasis.STRATEGY_AND_IMPLEMENTATION, "収益機構と実装対象の定義"),
    ClassifiedSkill("成長率10%", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "ベースケースの成長仮定"),
    ClassifiedSkill("割引率10%", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "ベースケースの資本コスト仮定"),
    ClassifiedSkill("量子化学計算", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "物性・反応のシミュレーション"),
    ClassifiedSkill("デザイン思考", SkillBasis.STRATEGY_AND_IMPLEMENTATION, "問題設定と仮説形成"),
    ClassifiedSkill("哲学", SkillBasis.STRATEGY_AND_IMPLEMENTATION, "概念設計と前提の吟味"),
    ClassifiedSkill("確率思考の戦略論（森岡毅氏）", SkillBasis.STRATEGY_AND_IMPLEMENTATION, "戦略意思決定への確率思考の実装"),
    ClassifiedSkill("デジタルツイン", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "状態空間の複製と更新"),
    ClassifiedSkill("仮想空間シミュレーション", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "介入前評価と反実仮想"),
    ClassifiedSkill("Claudecodeの活用", SkillBasis.STRATEGY_AND_IMPLEMENTATION, "研究OSと実装自動化"),
    ClassifiedSkill("シグモイド曲線", SkillBasis.STRATEGY_AND_IMPLEMENTATION, "普及・成長段階の認識"),
    ClassifiedSkill("アナロジー", SkillBasis.STRATEGY_AND_IMPLEMENTATION, "異分野写像による仮説生成"),
    ClassifiedSkill("ゲーム理論（ナッシュ均衡）", SkillBasis.STRATEGY_AND_IMPLEMENTATION, "競争環境の均衡分析"),
    ClassifiedSkill("SOTP（Sum-of-the-Parts：事業別評価合算）分析", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "事業別価値の合算"),
    ClassifiedSkill("ブラック・リッターマン・モデル", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "市場均衡と主観ビューの統合"),
    ClassifiedSkill("配当成長モデル（ゴードン・モデル）", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "永続成長の終価評価"),
    ClassifiedSkill("リバースDCF（Reverse Discounted Cash Flow）", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "価格に織り込まれた期待の逆算"),
    ClassifiedSkill("因果推論（Causal Inference）", SkillBasis.PROBABILISTIC_INFERENCE, "介入効果の識別"),
    ClassifiedSkill("虚数時間", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "解析接続と安定化の道具"),
    ClassifiedSkill("フィールズ賞（歴代フィールズ賞受賞者の論文）", SkillBasis.STRATEGY_AND_IMPLEMENTATION, "抽象理論から再利用可能な原理を抽出"),
    ClassifiedSkill("Maryna Viazovska", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "高次元配置と極値設計の参照軸"),
    ClassifiedSkill("カバード・コール（Covered Call）", SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION, "保有資産の収益補強と下方緩和"),
    ClassifiedSkill("フォークト関数（Voigt profile）", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "線形状の混合広がり"),
    ClassifiedSkill("Lax-Phillips基盤", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "散乱理論と共鳴解析"),
    ClassifiedSkill("ラドン変換（Radon Transform）", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "逆問題と断層再構成"),
    ClassifiedSkill("Lippmann-Schwinger方程式", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "散乱逆問題の基本方程式"),
    ClassifiedSkill("Adjoint State Method（随伴変数法）", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "勾配計算と逆最適化"),
    ClassifiedSkill("フィッシャー情報行列式を最大化（D-Optimality）", SkillBasis.OPTIMAL_EXPERIMENT_DESIGN, "識別力最大化の実験設計"),
    ClassifiedSkill("感応度行列（Jacobian）", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "局所感度と線形化"),
    ClassifiedSkill("特異摂動解析", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "多重スケール系の分離"),
    ClassifiedSkill("メトロポリス・ヘイスティングス法（MH法）", SkillBasis.PROBABILISTIC_INFERENCE, "MCMCの基本更新"),
    ClassifiedSkill("ブライト・ウィグナー分布", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "共鳴幅と散乱ピーク"),
    ClassifiedSkill("ブライト・ウィグナー分布", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "確率密度としての共鳴分布（重複登録）"),
    ClassifiedSkill("レイリー・ジーンズ分布", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "スペクトル分布の基礎モデル"),
    ClassifiedSkill("シャノン・エントロピー監視", SkillBasis.OPTIMAL_EXPERIMENT_DESIGN, "分布変化と異常兆候の監視"),
    ClassifiedSkill("Hilbert空間", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "作用素・射影・観測表現の基盤"),
    ClassifiedSkill("量子共鳴解析", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "準束縛状態とスペクトル構造"),
    ClassifiedSkill("成長断片化方程式の確率論的アプローチ", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "分裂成長系の動的モデル"),
    ClassifiedSkill("レジーム転換モデル", SkillBasis.PROBABILISTIC_INFERENCE, "状態遷移と構造変化の推定"),
    ClassifiedSkill("もつれ状態のコヒーレンス崩壊", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "開放系ダイナミクス"),
    ClassifiedSkill("実験計画法（Optimal Design of Experiments）", SkillBasis.OPTIMAL_EXPERIMENT_DESIGN, "観測設計と識別効率化"),
    ClassifiedSkill("ディオファントス近似", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "離散近似と共鳴条件の解析補助"),
    ClassifiedSkill("2DMAT", SkillBasis.OPTIMAL_EXPERIMENT_DESIGN, "材料探索・設計空間探索の実装系"),
    ClassifiedSkill("極値関数をどう設計するか", SkillBasis.OPTIMAL_EXPERIMENT_DESIGN, "目的関数と探索戦略の設計"),
    ClassifiedSkill("Lippmann-Schwinger", SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION, "散乱逆問題の基本方程式（短縮表記）"),
)


def _build_name_index() -> dict[str, tuple[ClassifiedSkill, ...]]:
    bucket: dict[str, list[ClassifiedSkill]] = defaultdict(list)
    for item in SKILL_CLASSIFICATION:
        bucket[item.name].append(item)
    return {name: tuple(items) for name, items in bucket.items()}


def _build_basis_index() -> dict[SkillBasis, tuple[ClassifiedSkill, ...]]:
    bucket: dict[SkillBasis, list[ClassifiedSkill]] = defaultdict(list)
    for item in SKILL_CLASSIFICATION:
        bucket[item.basis].append(item)
    return {basis: tuple(bucket.get(basis, [])) for basis in SkillBasis}


CLASSIFICATION_BY_NAME: Final[dict[str, tuple[ClassifiedSkill, ...]]] = _build_name_index()
GROUPED_BY_BASIS: Final[dict[SkillBasis, tuple[ClassifiedSkill, ...]]] = _build_basis_index()
BASIS_COUNTS: Final[dict[SkillBasis, int]] = {
    basis: len(items) for basis, items in GROUPED_BY_BASIS.items()
}
PROMPT_DUPLICATES: Final[tuple[str, ...]] = tuple(
    name for name, count in Counter(EXPECTED_PROMPT_ITEMS).items() if count > 1
)
CLASSIFICATION_DUPLICATES: Final[tuple[str, ...]] = tuple(
    name for name, count in Counter(item.name for item in SKILL_CLASSIFICATION).items() if count > 1
)
MISSING_FROM_CLASSIFICATION: Final[tuple[str, ...]] = tuple(
    name for name in EXPECTED_PROMPT_ITEMS if name not in CLASSIFICATION_BY_NAME
)
EXTRA_IN_CLASSIFICATION: Final[tuple[str, ...]] = tuple(
    name for name in CLASSIFICATION_BY_NAME if name not in EXPECTED_PROMPT_ITEMS
)


def get_by_name(name: str) -> tuple[ClassifiedSkill, ...]:
    return CLASSIFICATION_BY_NAME.get(name, ())


def get_unique(name: str) -> ClassifiedSkill:
    matches = get_by_name(name)
    if not matches:
        raise KeyError(f"Unknown skill: {name}")
    if len(matches) != 1:
        raise ValueError(f"Expected one skill for {name!r}, found {len(matches)}")
    return matches[0]


def grouped_by_basis() -> dict[SkillBasis, tuple[ClassifiedSkill, ...]]:
    return GROUPED_BY_BASIS


def basis_counts() -> dict[SkillBasis, int]:
    return BASIS_COUNTS


def classification_dict() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, items in CLASSIFICATION_BY_NAME.items():
        first = items[0]
        suffix = ""
        if len(items) > 1:
            suffix = f" ({len(items)} entries)"
        result[name] = {
            "basis": first.basis.value,
            "note": first.note + suffix,
        }
    return result


def validate() -> dict[str, tuple[str, ...]]:
    return {
        "prompt_duplicates": PROMPT_DUPLICATES,
        "classification_duplicates": CLASSIFICATION_DUPLICATES,
        "missing_from_classification": MISSING_FROM_CLASSIFICATION,
        "extra_in_classification": EXTRA_IN_CLASSIFICATION,
    }


if __name__ == "__main__":
    print("basis_counts=", {basis.value: count for basis, count in BASIS_COUNTS.items()})
    print("validate=", validate())
