# IUT Market Oracle

> **数式を資産に変える** — 定量研究から意思決定まで、すべてをひとつのシステムで。

---

## 秘密の3要素

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-green?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![uv](https://img.shields.io/badge/managed%20by-uv-blueviolet)](https://github.com/astral-sh/uv)

**デモ** — API を起動後、ブラウザで対話型ドキュメントを開く:

```
http://127.0.0.1:8000/docs
```

**コントリビューション** — バグ報告・改善提案はいつでも歓迎:

1. このリポジトリを Fork する
2. `git checkout -b feat/your-feature` でブランチを切る
3. `make lint && make test-cov` がグリーンであることを確認する
4. Pull Request を送る（必ず実験 ID と変更の意図を明記）

---

## 特徴

- **再現性を保証する実験追跡**
  すべての実験結果は `experiment_id → config → code → report` の連鎖でトレースできる。
  `experiments/registry.md` が唯一の正典。乱数シードから生成チャートまで、後から完全再現できる。

- **数学的に厳密なモデル群が即使える**
  DCF 評価・ベイズ推論・グラフ理論（マトロイド対数凹性 / Fiedler 固有値 / RMT）・デジタルツイン・エントロピー監視・GNSS スプーフィング検出・製造歩留まり最適化まで、研究グレードのアルゴリズムを型安全な Pydantic スキーマで提供する。

- **探索から本番まで一本の API**
  FastAPI が 12 のルーターを束ね、Swagger UI から全エンドポイントを試せる。
  CLI・Jupyter・REST クライアントのどこからでも同じロジックを呼び出せる。

- **インタラクティブ React ダッシュボード**
  `frontend/` に React (Vite + TypeScript) 製の SPA を同梱。
  GNSS Twin・DCF 評価・ベイズ推論・エントロピー監視を UI から直接実行できる。

---

## 使い方

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. テストの実行

```bash
uv run pytest -q
```

### 3. レポートの生成（DCF + グラフ + チャート）

```bash
uv run python -m src.report
```

生成物は `reports/exp-<id>/` に書き出される。

### 4. API サーバーの起動

```bash
uv run uvicorn src.api.app:app --reload
```

ブラウザで `http://127.0.0.1:8000/docs` を開くと、全エンドポイントを対話的に試せる。

### 5. React ダッシュボードの起動（開発モード）

```bash
cd frontend
npm install
npm run dev
```

ブラウザで `http://localhost:5173` を開く（Vite dev server がバックエンドに自動プロキシする）。

**本番ビルド:**

```bash
cd frontend
npm run build   # dist/ に出力
```

#### ダッシュボード構成

| ページ | パス | 概要 |
|---|---|---|
| Overview | `/` | 全モジュールへのナビゲーション |
| GNSS Twin | `/gnss` | T1500 — スプーフィング検出・Monte Carlo ベンチマーク |
| Valuation | `/valuation` | T400 — DCF 内在価値・逆算 DCF 含意成長率 |
| Bayesian | `/bayesian` | T200 — Beta / Normal 共役更新・事後分布ビジュアライザ |
| Entropy Monitor | `/entropy` | T1000 — Shannon エントロピー・KL 発散・レジーム変化点検出 |

---

## モジュール構成

| モジュール | タスク | 概要 |
|---|---|---|
| `valuation/` | T400 | DCF・ゴードン成長モデル・逆算 DCF |
| `bayesian/` | T200 | ベイズ更新・事後分布サマリー・ネットワーク |
| `graph/` | T300 | マトロイド対数凹性・Fiedler 固有値・RMT スペクトル |
| `twin/` | T800 / T1100 | Markov レジーム切替・Gamma-Poisson 市場進化 |
| `exit/` | T900 | 出口戦略オプション価格・タイミング分布 |
| `entropy/` | T1000 | Shannon エントロピー・KL 発散・ローリング変化率アラート |
| `gnss/spoof_sim` | T1300 | Fisher 統合スコアによる GNSS スプーフィング MC 検出 |
| `gnss/multi_sensor_sim` | T1350 | 浸透グラフ + Lorentz AoA 多センサー融合 |
| `gnss/resilience_twin` | T1500 | GM-RAIM + IMM-KF + スペクトル + エントロピー 4 クラス判別 |
| `yield_twin/` | T1600 | GP 代理モデル (ARD RBF) + D 最適実験計画 + EI 融合 |
| `models/` | T1400 | モデルレジストリ・LLM モデル推薦・アイデア形式化 |
| `huh_twin/` | — | DCF ラッパー・感度分析・スキルグラフ・Bayes エンジン |

---

## ディレクトリ構成

```
iut_market_oracle/
├── src/                  # 実装（再利用可能）
├── frontend/             # React ダッシュボード (Vite + TypeScript)
│   ├── src/pages/        # ページコンポーネント
│   ├── src/components/   # 共有 UI コンポーネント
│   ├── src/styles/       # デザイントークン
│   └── src/api.ts        # バックエンド API クライアント
├── tests/                # ユニットテスト・プロパティテスト
├── configs/              # 事前分布・シナリオ・モデルレジストリ YAML
├── experiments/          # 実行スコープ出力 + メタデータ
├── reports/              # 生成チャート・テーブル・サマリー
├── notes/                # 数学ノート・証明スケッチ
├── data/raw/             # 不変ソースデータ（変更禁止）
└── data/processed/       # 派生データセット
```

---

## 設計原則

- **Explore → Plan → Edit** の順で作業する
- 生成した数値には必ず実験 ID が付く（出所不明の結果を出さない）
- テストが通らないコードはデプロイしない
- マジックナンバーは名前付き定数にする
- 乱数シードを設定して再現性を保証する

---

*Every reportable result must have a discoverable origin: experiment ID → config → code.*
