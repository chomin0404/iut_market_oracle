# Problem Formulation Template

このドキュメントは、問題を定式化し、実装計画を切り、検証可能なタスクへ落とすためのテンプレートである。
目的は、曖昧なアイディアを「実装できる最小単位」に変換することにある。

---

## 1. Problem

- Task name:
- One-sentence problem:
- Who is affected:
- Why this matters now:

記入原則:
- 1文で書く。
- 実装対象を含める。
- 「分析したい」ではなく「何を入力し、何を返すか」で書く。

例:
- GNSS observation logs から spoofing risk score と alarm を返す。

---

## 2. Objective

- Primary objective:
- Secondary objective:
- Final output:
- Success condition:
- Failure condition:

記入原則:
- 成功条件は観測可能にする。
- 抽象目標ではなく、判定可能な条件にする。

例:
- spoofing シナリオで alarm が出る。
- genuine シナリオで false alarm rate を集計できる。
- score_components と reasons が返る。

---

## 3. Inputs

- Main observations:
- Auxiliary signals:
- Input format:
- Sampling assumptions:
- Missing data assumptions:
- Noise assumptions:

記入原則:
- データの単位、時系列性、欠損を先に書く。
- 後で実装時に迷う曖昧語を残さない。

例:
- pseudorange, Doppler, C/N0, AoA, INS residual
- JSONL or CSV
- satellite dropout may occur

---

## 4. States and Structure

- Hidden states:
- Observable states:
- Change points:
- Normal hypothesis:
- Abnormal hypothesis:
- Structural assumptions:

記入原則:
- 「何が見えていて、何が見えていないか」を分ける。
- 異常の定義を先に置く。

例:
- hidden state: genuine / spoofed / jammed
- abnormal hypothesis: satellite channels become unnaturally coherent

---

## 5. Constraints

- Physical constraints:
- Computational constraints:
- Operational constraints:
- Explainability constraints:
- Deployment constraints:

記入原則:
- 数理的な美しさより、運用上の制約を優先して書く。
- 今回のMVPで守るべき制約に絞る。

例:
- run must finish within practical CLI time
- output must be explainable to non-research users

---

## 6. Candidate Models

- Candidate A:
- Candidate B:
- Candidate C:
- Chosen model:
- Why chosen:
- Why not the others:

記入原則:
- 候補は最低2つ書く。
- 「なぜ選ぶか」と「なぜ捨てるか」を両方書く。

例:
- z-score aggregation
- Bayesian state-space update
- graph coherence + subset selection

---

## 7. Metrics

- Detection metric:
- False alarm metric:
- Delay metric:
- Robustness metric:
- Reproducibility metric:
- Explainability metric:

記入原則:
- accuracy だけで終わらせない。
- 誤警報と再現性を必ず入れる。

例:
- detection rate
- false alarm rate
- epochs to alarm
- same config and seed reproduce same result

---

## 8. MVP Scope

- In scope:
- Out of scope:
- Minimum acceptable output:
- Files likely to change:
- Deliverables:

記入原則:
- 今回やらないことを書く。
- 1日で終わらない巨大タスクにしない。

例:
- in scope: log ingest, score, alarm, reasons
- out of scope: full OSNMA integration
- deliverables: CLI output, tests, sample report

---

## 9. Verification Plan

- Unit tests to add:
- Integration checks:
- Baseline to compare:
- Output artifacts:
- Logging / traceability:
- Failure modes to inspect:

記入原則:
- コードを書く前に検証を決める。
- 成果物を output/ に残す前提で書く。

例:
- normal sample gives low score
- spoofing sample gives alarm
- save metrics CSV and JSON report

---

## 10. Implementation Plan

- Smallest diff:
- Target files:
- Non-target files:
- Dependencies:
- Acceptance criteria:
- Stop condition for today:

記入原則:
- 1差分1目的にする。
- 「今日終える条件」を必ず書く。

例:
- target files: analyzer.py, schemas.py, test_analyzer.py
- acceptance: sample input returns valid JSON with score and reason

---

## 11. Review

- What worked:
- What failed:
- What remains uncertain:
- Next action:

記入原則:
- 学習記録ではなく、次の実装判断に使える形で残す。
- 不確実性を消さずに書く。
