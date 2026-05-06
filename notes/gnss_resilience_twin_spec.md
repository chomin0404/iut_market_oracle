# GNSS Resilience Twin — Product Specification (T1500)
<!-- sections 11–20 -->
<!-- Fix log:
     [F1] run_id 所有権: サーバ生成 UUID v7 に変更 (§12, §13)
     [F2] batch モード非同期化: 202 Accepted + task_id / GET polling (§12)
     [F3] imu_residual 型明確化: IMUResidual 型を新設、単位・正規化方法を明記 (§11.1, §13)
     [F4] Section 11.1 補完: 全型定義を追記 (§11.1)
     [F5] State Posterior 合計制約明示: §13.5 StatePosterior 不変量を追加
     [F6] PVTReestimate 出力定義: §14.2 PVTReestimateResult を新設
     [F7] 既存実装との対応表修正: §14 を Layer 1–10 の一貫番号体系に統一、EpochDiagnosis フィールドパスを追加
     [F8] OSNMA 検証モデル深化: §21 TESLA 5チェック階層・per-satellite AuthState・Bayesian仰角加重・攻撃パターン対応表を追記
     [F9] calibrate スキーマ定義: §22 CalibrateRequest/CalibrateResult/POST /v1/calibrate を追記
     [F10] pagination 設計: §23 cursor-based pagination (エポック列挙・レポート一覧) を追記
-->

---

## 11. Input / Output Schema

### 11.1 Input Types

<!-- [F4] 補完: 前バージョンではこのセクションが途中で終わっていた。全フィールドを定義する。 -->
<!-- [F3] imu_residual: IMUResidual 型を定義 -->

#### 11.1.1 IMUResidual

INS とGNSS 速度解の不整合を表す複合型。

| フィールド | 型 | 単位 | 説明 |
|---|---|---|---|
| `delta_v` | `float[3]` | m/s | 体座標系 (body frame) 3軸速度偏差 `v_ins − v_gnss` |
| `mahalanobis_dist` | `float` | 無次元 | `sqrt(Δv^T · Σ^{-1} · Δv)` ここで `Σ = diag(σ_ins²)·I₃` |
| `norm_ms` | `float` | m/s | `‖Δv‖₂` — 正規化前の絶対偏差ノルム |
| `alert` | `bool` | — | `mahalanobis_dist > χ²_{0.95}(3)^{0.5} ≈ 2.795` のとき `true` |

**正規化規則**:
- `σ_ins` は `TwinRunRequest.ins_noise_std` ([m/s], デフォルト `0.05`) に等しい。
- `mahalanobis_dist = norm_ms / σ_ins`（等方性仮定のため対角化が成立）。
- 単位行列スケーリングにより、異なる `ins_noise_std` 設定間で比較可能な無次元距離となる。

#### 11.1.2 ChannelFeature

1 衛星・1 エポック分の受信機出力を格納する型。全フィールドはオプショナル（未計測センサを `null` で表現）。

| フィールド | 型 | 単位 | 説明 |
|---|---|---|---|
| `svid` | `int` | — | 衛星番号 (1–36 for Galileo; 1–32 for GPS) |
| `doppler_residual` | `float` | Hz | 予測値との差分 `Δf = f_meas − f_pred` |
| `pseudorange_residual` | `float \| null` | m | 搬送波平滑化後の擬似距離残差 |
| `cn0_dbhz` | `float \| null` | dB-Hz | 搬送波対雑音密度比 (典型範囲: 20–60) |
| `elevation_deg` | `float \| null` | deg | 受信機から見た仰角 (0–90) |
| `sqm` | `float \| null` | 無次元 | 信号品質指標 ∈ [0, 1]。0 = 正常, 1 = 劣化 |
| `osnma_auth` | `bool \| null` | — | Galileo OSNMA 認証結果 (`true` = 認証成功) |

#### 11.1.3 AuthenticationEvent

TESLA/OSNMA 認証層から発行されるイベント型。

| フィールド | 型 | 説明 |
|---|---|---|
| `epoch` | `int` | サブフレームエポック番号 |
| `svid` | `int` | 対象衛星番号 |
| `key_valid` | `bool` | TESLA ハッシュチェーン検証結果 |
| `mac_valid` | `bool` | MAC タグ検証結果 |
| `receipt_safe` | `bool` | 受信タイミング安全性検証結果 |
| `spoofing_detected` | `bool` | `not (key_valid and mac_valid and receipt_safe)` |
| `auth_fraction` | `float` | このエポックまでの累積認証成功率 ∈ [0, 1] |

#### 11.1.4 SimulationScenario

MC シミュレーション 1 試行の注入条件。`run_resilience_simulation()` 内部で生成され、リプレイ用にシリアライズ可能。

| フィールド | 型 | 説明 |
|---|---|---|
| `fault_class` | `FaultClass` | 注入した障害クラス (`nominal` / `multipath` / `hardware_fault` / `spoofing`) |
| `attack_start` | `int \| null` | 攻撃開始エポック (spoofing / hw_fault のみ) |
| `attack_end` | `int \| null` | 攻撃終了エポック |
| `faulty_sat_idx` | `int \| null` | 障害衛星インデックス (hardware_fault のみ) |
| `spoof_bias_hz` | `float \| null` | 注入したメアコニングバイアス [Hz] |
| `random_seed` | `int` | 試行に使用した乱数シード |

#### 11.1.5 AnalysisRequest

`POST /v1/analyze` エンドポイントへのリクエスト本体。

<!-- [F1] run_id はクライアントから送付しない。サーバ側で生成 (§12 参照) -->

| フィールド | 型 | 必須 | デフォルト | 説明 |
|---|---|---|---|---|
| `observations` | `ObservationEpoch[]` | Y | — | 観測エポックのシーケンス (2–5000 件) |
| `n_sats` | `int` | N | `6` | 可視衛星数 (5–20)。全エポックの `doppler_residuals` 長と一致すること |
| `los_vectors` | `float[n_sats][3] \| null` | N | `null` | 単位 LOS ベクトル行列。省略時は Fibonacci 格子配置を自動生成 |
| `doppler_noise_std` | `float` | N | `0.30` | 公称ドップラー雑音 1σ [Hz] |
| `graph_sigma` | `float` | N | `1.50` | スペクトルグラフ RBF カーネル帯域幅 σ [Hz] |
| `ins_noise_std` | `float` | N | `0.05` | INS 速度雑音 1σ [m/s]。`IMUResidual.mahalanobis_dist` の正規化定数として使用 |
| `mode` | `"realtime" \| "batch"` | N | `"realtime"` | `realtime`: 同期応答 (200 OK) / `batch`: 非同期応答 (202 Accepted) — §12 参照 |

> **run_id はリクエストに含めない。** サーバが UUID v7 を生成してレスポンスで返す。クライアントがIDを指定すると 422 を返す。

#### 11.1.6 AnalysisResult

分析結果の最上位コンテナ。`realtime` モードでは 200 レスポンス本文に直接含まれる。`batch` モードでは `GET /v1/reports/{run_id}` でポーリングして取得する。

| フィールド | 型 | 説明 |
|---|---|---|
| `run_id` | `string` | サーバ生成 UUID v7。結果の一意識別子 |
| `created_at` | `datetime` | 生成タイムスタンプ (UTC ISO 8601) |
| `status` | `"completed" \| "running" \| "failed"` | 処理状態 |
| `n_epochs` | `int` | 処理したエポック数 |
| `n_sats` | `int` | 衛星数 |
| `epoch_reports` | `EpochReport[]` | 各エポックの診断結果 (`status == "completed"` のときのみ存在) |
| `dominant_diagnosis` | `FaultClass.value` | 全エポックで最頻の MAP 診断 |
| `mean_authenticity_genuine` | `float` | 平均 P(genuine) |
| `mean_integrity_nominal` | `float` | 平均 P(nominal) |
| `alert_epochs` | `int[]` | エントロピーアラートが発火したエポック番号リスト |
| `spoofing_window` | `[int, int] \| null` | P(spoofing) > 0.50 の最初/最後エポック |
| `worst_action` | `RecommendedAction` | セッション内で最悪の推奨アクション |
| `error` | `string \| null` | `status == "failed"` のときのエラーメッセージ |

---

## 12. API Draft

### 12.1 エンドポイント一覧

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/v1/analyze` | 観測シーケンスを送信して分析を開始 |
| `GET` | `/v1/reports` | レポート一覧 (cursor-based pagination) — §23.2 |
| `GET` | `/v1/reports/{run_id}` | 分析結果の取得 (realtime / batch 共通) |
| `GET` | `/v1/reports/{run_id}/status` | 処理状態のみを取得 (batch ポーリング用) |
| `GET` | `/v1/reports/{run_id}/epochs` | エポックレポートの分割取得 — §23.3 |
| `DELETE` | `/v1/reports/{run_id}` | 結果の削除 |
| `POST` | `/v1/calibrate` | 正常観測窓から雑音パラメータを推定 — §22 |
| `POST` | `/v1/simulate` | MC ベンチマーク実行 |
| `GET` | `/v1/health` | ヘルスチェック |

### 12.2 POST /v1/analyze

<!-- [F1] run_id はサーバが生成する。クライアントはリクエストボディに含めない。 -->
<!-- [F2] batch モードは 202 Accepted を返す。realtime モードは 200 OK を返す。 -->

#### realtime モード (`mode: "realtime"`, デフォルト)

```
POST /v1/analyze
Content-Type: application/json

{
  "observations": [...],
  "n_sats": 6,
  "mode": "realtime"
}
```

**レスポンス: 200 OK**

```json
{
  "run_id": "018f4b2e-7c3a-7d45-b891-0a1234567890",
  "created_at": "2026-05-06T12:00:00.000Z",
  "status": "completed",
  "n_epochs": 80,
  "dominant_diagnosis": "nominal",
  ...
}
```

- `run_id` はサーバが UUID v7 (時刻ソート可能な単調増加 UUID) で生成する。
- レスポンスは `AnalysisResult` スキーマに準拠し、`epoch_reports` を含む。
- タイムアウト: 30 秒。超過時は 504 Gateway Timeout。

#### batch モード (`mode: "batch"`)

<!-- [F2] 非同期化: 202 Accepted + task_id を即時返却。完了後に GET /v1/reports/{run_id} でポーリング。 -->

```
POST /v1/analyze
Content-Type: application/json

{
  "observations": [...],
  "n_sats": 6,
  "mode": "batch"
}
```

**レスポンス: 202 Accepted**

```json
{
  "run_id": "018f4b2e-7c3a-7d45-b891-0a1234567890",
  "status": "running",
  "poll_url": "/v1/reports/018f4b2e-7c3a-7d45-b891-0a1234567890"
}
```

- `run_id` はサーバが生成。クライアントはリクエストに `run_id` を含めない。
- バックグラウンドで処理が開始され、`poll_url` (= `GET /v1/reports/{run_id}`) でステータスを確認できる。
- ポーリング間隔は最小 1 秒を推奨。
- `status: "running"` のレスポンスは `epoch_reports` フィールドを含まない。
- 処理完了後は `status: "completed"` に遷移し、`epoch_reports` が付与される。
- エラー時は `status: "failed"` と `error` フィールドが付与される。

#### エラーレスポンス

| HTTP コード | 条件 |
|---|---|
| `400 Bad Request` | バリデーションエラー (不正な衛星数、重複観測等) |
| `422 Unprocessable Entity` | リクエストボディに `run_id` フィールドが含まれている |
| `504 Gateway Timeout` | realtime モードで 30 秒超過 |

### 12.3 GET /v1/reports/{run_id}

```
GET /v1/reports/018f4b2e-7c3a-7d45-b891-0a1234567890
```

**レスポンス: 200 OK**
`AnalysisResult` スキーマに準拠。`status` が `completed` のとき `epoch_reports` を含む。

**レスポンス: 404 Not Found**
指定 `run_id` が存在しない、または保持期間 (デフォルト: 24 時間) を超過した場合。

### 12.4 GET /v1/reports/{run_id}/status

完了前のポーリングコストを低減するための軽量エンドポイント。

**レスポンス: 200 OK**

```json
{
  "run_id": "018f4b2e-7c3a-7d45-b891-0a1234567890",
  "status": "running",
  "created_at": "2026-05-06T12:00:00.000Z",
  "progress_epochs": 42
}
```

`progress_epochs`: 処理済みエポック数 (推定値、`n_epochs` に達したら `completed` に遷移)。

---

## 13. Data Model

### 13.1 FaultClass

4 クラスの GNSS 障害分類。インデックス順は `fault_posterior` 配列と一致する。

| インデックス | 値 | 説明 |
|---|---|---|
| 0 | `nominal` | 正常動作 |
| 1 | `multipath` | マルチパス (仰角相関雑音) |
| 2 | `hardware_fault` | 単一衛星ハードウェア障害 |
| 3 | `spoofing` | メアコニング (共通バイアス攻撃) |

### 13.2 IMUResidual (詳細)

<!-- [F3] 型明確化: 前バージョンでは imu_residual が float として曖昧に定義されていた -->

`IMUResidual` は §11.1.1 で定義した複合型。実装上の対応:

```python
# src/gnss/resilience_twin.py — INS coupling layer (Layer 5)
@dataclass(frozen=True)
class INSCouplingResult:
    chi2_vel: float      # = IMUResidual.mahalanobis_dist ** 2
    alert: bool          # = IMUResidual.alert
    ins_velocity: np.ndarray | None  # = IMUResidual.delta_v (3-vector)
```

**旧定義 (廃止)**:

```
imu_residual: float   # 曖昧: 何のノルム? 単位は? 閾値は?
```

**新定義**:

```
imu_residual: IMUResidual {
    delta_v:          float[3]  # [m/s]  — body-frame velocity difference
    mahalanobis_dist: float     # [無次元] — delta_v / ins_noise_std (等方性)
    norm_ms:          float     # [m/s]  — L2 norm of delta_v
    alert:            bool      # mahalanobis_dist > 2.795 (chi2(3, p=0.95)^0.5)
}
```

API レスポンスでは `EpochReport.ins_chi2_vel = mahalanobis_dist²` として露出する。
`ins_noise_std` が未指定の場合は `0.05 m/s` (デフォルト) を使用する。

### 13.3 run_id の所有権

<!-- [F1] 所有権規則を明文化 -->

| 属性 | 値 |
|---|---|
| **生成者** | サーバ (クライアントは指定不可) |
| **形式** | UUID バージョン 7 (RFC 9562) — 48 ビット Unix 時刻 + 74 ビット乱数 |
| **単調性** | 生成時刻で語彙的ソート可能 |
| **返却タイミング** | POST /v1/analyze レスポンス (`200 OK` または `202 Accepted`) |
| **有効期間** | デフォルト 24 時間 (サーバ設定で変更可能) |
| **冪等性** | 同一リクエストボディを再送しても毎回新しい `run_id` が生成される |

> クライアントが `run_id` をリクエストボディに含めた場合、サーバは **422 Unprocessable Entity** を返す。

### 13.5 StatePosterior — 合計制約と不変量

<!-- [F5] State Posterior の合計制約を明示する -->

`fault_posterior` は 4 次元確率単体 (probability simplex) S⁴ の要素である:

```
Invariant:  Σᵢ fault_posterior[i] = 1.0   (i ∈ {0,1,2,3})
            fault_posterior[i] ≥ 0.0       for all i
```

**実装上の保証**:

| 保証場所 | 方法 |
|---|---|
| `ResilienceTwin.step()` | `softmax(score)` を適用 → 数値的に和 = 1 |
| `TwinDiagnosis.epoch_diag.fault_posterior` | `GMMRaim` + 融合スコアの出力をそのまま格納 |
| `EpochReport.fault_posterior` | API シリアライズ時に丸めは行わない (float 精度のまま) |
| テスト | `test_fault_posterior_sums_to_one`: `abs(sum(fp) − 1.0) < 1e-9` |

**診断フィールドとの対応**:

```
fault_posterior = (P_nominal, P_multipath, P_hardware_fault, P_spoofing)
                   index 0       index 1       index 2           index 3

authenticity["genuine"]  = P_nominal + P_multipath + P_hardware_fault
authenticity["spoofed"]  = P_spoofing
integrity["nominal"]     = P_nominal
integrity["degraded"]    = P_multipath + P_hardware_fault + P_spoofing
```

これらの派生フィールドも常に [0, 1] に収まる（単体制約から自動保証）。

### 13.4 ObservationEpoch (既存型との対応)

```python
# src/schemas.py (現行)
class ObservationEpoch(BaseModel):
    doppler_residuals: list[float]          # (n_sats,) [Hz]
    elevations_deg:    list[float] | None   # (n_sats,) [deg], optional
    ins_velocity_ms:   list[float] | None   # (3,) [m/s], optional — Δv raw value
    osnma_auth_per_sat: list[bool] | None   # (n_sats,), optional
```

`ins_velocity_ms` は `IMUResidual.delta_v` の入力元。サーバ側で `ins_noise_std` を使って
`mahalanobis_dist` に変換する。

---

## 14. Detection Layers (参照)

<!-- [F7] Layer 1–10 の一貫番号体系に統一、EpochDiagnosis フィールドパスを追加 -->

4 本柱 10 レイヤの実装マッピング。`EpochDiagnosis` フィールドパスは `ResilienceTwin.step()` の戻り値へのアクセス経路。

| Layer | 柱 | 実装クラス | ソースファイル | 主要出力フィールド | EpochDiagnosis フィールドパス |
|---|---|---|---|---|---|
| 1 | 認証 | `OSNMAReceiver` + `AuthMonitor` | `gnss/core.py`, `resilience_twin.py` | `auth_fraction`, `p_spoofed`, `alert` | `diag.auth.osnma.auth_fraction`, `diag.auth.p_spoofed`, `diag.auth.alert` |
| 2 | 完全性 | `GMMRaim` | `resilience_twin.py` | `gamma[n_sats]`, `n_fault`, `sign_corr` | `diag.integrity.gmm.gamma`, `diag.integrity.gmm.n_fault`, `diag.integrity.gmm.sign_corr` |
| 3 | 完全性 | `IMMKalman` | `resilience_twin.py` | `mode_weights[3]`, `x_fused[4]`, `innovation_norms[3]` | `diag.integrity.imm.mode_weights`, `diag.integrity.imm.x_fused` |
| 4 | 完全性 | `INSCoupling` | `resilience_twin.py` | `chi2_vel`, `alert` | `diag.integrity.ins.chi2_vel`, `diag.integrity.ins.alert` |
| 5 | 完全性 | `CoopRAIM` | `resilience_twin.py` | `parity_chi2`, `parity_alert` | `diag.integrity.coop_raim.parity_chi2`, `diag.integrity.coop_raim.parity_alert` |
| 6 | 構造 | `SpectralMonitor` | `resilience_twin.py` | `fiedler_ratio`, `rmt_anomaly`, `spectral_entropy` | `diag.structure.spectral.fiedler_ratio`, `diag.structure.spectral.rmt_anomaly` |
| 7 | 構造 | `StructuralDependencyMonitor` | `resilience_twin.py` | `fiedler_streak`, `alert` | `diag.structure.structural.fiedler_streak`, `diag.structure.structural.alert` |
| 8 | 介入 | `FaultEntropyMonitor` | `resilience_twin.py` | `entropy`, `kl`, `alert` | `diag.entropy.entropy`, `diag.entropy.kl`, `diag.entropy.alert` |
| 9 | 完全性 | `HuhSubsetSelector` | `resilience_twin.py` | `selected_subset`, `n_selected`, `det_ratio`, `log_concavity_ratio` | `diag.integrity.huh.selected_subset`, `diag.integrity.huh.det_ratio`, `diag.integrity.huh.log_concavity_ratio` |
| 10 | 構造 | `DuminilCopinPhaseMonitor` | `resilience_twin.py` | `susceptibility_peak`, `percolation_threshold`, `lcc_at_null`, `phase_alert` | `diag.structure.phase.susceptibility_peak`, `diag.structure.phase.percolation_threshold`, `diag.structure.phase.phase_alert` |

### 14.1 スコア融合

```
P_fault_raw[k] ∝ exp(score[k])   where k ∈ {nominal, multipath, hw_fault, spoofing}

score[nominal]       = −(gmm_mean_gamma) − imm_spoof − spectral_anomaly
score[multipath]     = 0.40·elev_corr + 0.20·fiedler_rise + 0.40·entropy_grad
score[hw_fault]      = 0.70·max(gamma) − 0.30·sign_corr + huh_det_ratio
score[spoofing]      = 0.50·(1−fiedler_ratio) + 0.30·rmt_anomaly + imm_spoof_weight
                       + 0.20·(1−auth_fraction) + 10·phase_alert

fault_posterior = softmax(score)   # sum = 1
diagnosis       = argmax(fault_posterior)
confidence      = max(fault_posterior)
```

### 14.2 PVTReestimateResult

<!-- [F6] PVT 再推定の出力を定義する -->

Layer 9 (`HuhSubsetSelector`) が選択した健全衛星サブセットを用いて PVT を再推定した結果。
`HuhSubsetResult.selected_subset` を入力として位置・速度・時計バイアスを再計算する。

#### 出力フィールド

| フィールド | 型 | 単位 | 説明 |
|---|---|---|---|
| `position_enu` | `float[3]` | m | ENU 座標系での位置推定値 `[East, North, Up]` (参照点からの偏差) |
| `velocity_enu` | `float[3]` | m/s | ENU 座標系での速度推定値 |
| `clock_bias_m` | `float` | m | 受信機時計バイアス (光速換算) |
| `clock_drift_ms` | `float` | m/s | 受信機時計ドリフト (光速換算) |
| `gdop` | `float` | 無次元 | Geometric DOP — `sqrt(trace(H^T H)^{-1})` |
| `pdop` | `float` | 無次元 | Position DOP — 3D 位置成分のみの DOP |
| `n_sats_used` | `int` | — | 使用衛星数 (= `HuhSubsetResult.n_selected`) |
| `excluded_indices` | `int[]` | — | 除外衛星インデックス (= `HuhSubsetResult.n_excluded` 分) |
| `postfit_residuals` | `float[n_selected]` | m | 推定後の擬似距離残差 |
| `integrity_ok` | `bool` | — | PDOP < 6.0 かつ `n_sats_used` ≥ 4 のとき `true` |

#### 算出規則

```
H = geometry_matrix(los[selected_subset])   # (n_selected × 4) — [e_x, e_y, e_z, 1]
x̂ = (H^T W H)^{-1} H^T W r                 # 重み付き最小二乗 PVT
W = diag(sin²(el_i))                        # 仰角加重

postfit_residuals = r[selected_subset] − H · x̂
gdop = sqrt(trace((H^T H)^{-1}))
pdop = sqrt(trace((H^T H)^{-1})[:3, :3]))
integrity_ok = (pdop < 6.0) and (n_sats_used >= 4)
```

#### 実装との対応

現行実装では `HuhSubsetSelector` は衛星インデックスの選択のみを行い、
PVT 再計算は `gnss/spoof_sim.py` の `_geometry_matrix()` と `_propagate_state()` を利用して
呼び出し元 (`ResilienceTwin`) が実行する想定。`PVTReestimateResult` は次期バージョンで
`EpochDiagnosis.integrity.pvt_reestimate` フィールドとして追加予定。

#### API での露出

`EpochReport` への追加フィールド（予定）:

| フィールド | 型 | 説明 |
|---|---|---|
| `pvt_gdop` | `float \| null` | 再推定後 GDOP |
| `pvt_pdop` | `float \| null` | 再推定後 PDOP |
| `pvt_integrity_ok` | `bool \| null` | PVT 完全性フラグ |
| `pvt_n_sats_used` | `int \| null` | 再推定使用衛星数 (= `huh_n_selected` と一致すること) |

---

## 15. Failsafe State Machine

4 段階のフェイルセーフ。状態は `FailsafeState` として `ControlAction` に添付される。

| レベル | 遷移条件 (降下) | INS 重み範囲 |
|---|---|---|
| `nominal` | 初期状態 | [0.0, 1.0] |
| `degraded` | P(spoof) > 0.50 または active_sats < 5 | [0.45, 0.70] |
| `ins_only` | P(spoof) > 0.80 または active_sats < 4 | [0.90, 0.90] |
| `dead_reckoning` | active_sats < min_sats (= 4) かつ GNSS ロスト | [1.00, 1.00] |

- **即時降下**: 遷移条件成立後 1 エポックで下位状態へ移行。
- **回復遷移**: `_FAILSAFE_RECOVERY_EPOCHS` (= 3) 連続エポック条件解消後に 1 段階回復。

---

## 16. INS Blending Weight Formula

```
w_raw = sum(P(class_k) * w_k  for k in {nominal, multipath, hw_fault, spoofing})

where:
    w_nominal       = 0.05
    w_multipath     = 0.25
    w_hardware_fault = 0.60
    w_spoofing      = 0.90

# EMA smoothing (cold-start: w_ema = w_raw on first call)
w_ema ← 0.30 * w_raw + 0.70 * w_ema

# Confidence gate (bypass EMA when confidence is low)
if confidence >= 0.70:
    w_ins = w_ema
else:
    blend = confidence / 0.70
    w_ins = blend * w_ema + (1 − blend) * w_raw

# Failsafe clamping
w_ins = clip(w_ins, failsafe.ins_weight_floor, failsafe.ins_weight_ceil)
```

---

## 17. Alert Hierarchisation

`AlertEvent` は各エポックで `AlertBuilder` が生成する。

| レベル | 条件 |
|---|---|
| `info` | アラートソースなし |
| `caution` | アラートソース 1 件 |
| `warning` | アラートソース ≥ 2 件、または P(spoof) > 0.50 |
| `critical` | P(spoof) > 0.80、または failsafe ≥ `ins_only` |

アラートソース: `entropy_alert`, `osnma_alert`, `phase_alert`, `structural_alert`, `ins_alert`, `coop_parity_alert`

---

## 18. MVP Pipeline Architecture

```
RawEpochData
     │
     ▼
 ReceiverAgent         (Module 1) — C/N0 gate + SQM gate + IMU conversion
     │ ReceiverObservation
     ▼
 TwinCore              (Module 2) — ResilienceTwin (10 layers) + MC replay
     │ TwinDiagnosis
     ▼
 ActionPlanner         (Module 3) — Satellite scoring + INS weight + Failsafe + Alert
     │ ControlAction
     ▼
 MVPPipeline.history   (Module 4) — Multi-epoch state + dominant_diagnosis + mean_ins_weight
```

### 18.1 衛星スコアリング (SatelliteScorer)

```
s_i = 0.60·γ_i + 0.25·1[SQM_i > 0.70] + 0.15·(1 − auth_i)

s_i ≥ 0.75  → hard_exclude  (weight = 0)
0.40 ≤ s_i  → downweight    (weight = 1 − s_i)
s_i < 0.40  → accept        (weight = 1.0)
```

フォールバック規則: 除外後の残存衛星が `min_sats` (= 4) を下回る場合、
RX pre-excluded のみ除外 → それでも下回る場合は全衛星保持。

---

## 19. MC Simulation Entry Point

```python
from gnss.resilience_twin import ResilienceTwinConfig, run_resilience_simulation

cfg = ResilienceTwinConfig(
    n_mc=400,
    n_epochs=80,
    n_sats=6,
    doppler_noise_std=0.30,
    spoof_bias_std=2.50,
    spoof_diff_std=0.80,
    graph_sigma=1.50,
    dirichlet_alpha=2.0,
    random_seed=42,
)
report = run_resilience_simulation(cfg)
# report: ResilienceTwinReport
#   .p_detection, .p_false_alarm, .auc
#   .confusion_matrix[4][4]
#   .per_class_accuracy {FaultClass.value: float}
#   .mean_confidence
#   .n_mc, .n_mc_per_class
```

試行は NOMINAL / MULTIPATH / HARDWARE_FAULT / SPOOFING をラウンドロビンで循環する。
各クラスの試行数は `n_mc_per_class` (= `n_mc // 4` ± 1) に格納される。

---

## 20. Validation Gates

| チェック | コマンド | 合格基準 |
|---|---|---|
| Lint | `make lint` | ruff exit 0 |
| Unit tests | `make test-cov` | 全件 PASS + カバレッジ ≥ 80% |
| MC AUC | `pytest tests/test_gnss_resilience_twin.py -q` | AUC ∈ [0, 1] |
| API smoke | `uv run uvicorn src.api.app:app` → `POST /gnss/resilience-sim` | HTTP 200 |
| Report | `make report` | `reports/exp-*/` に生成物あり |

### 20.1 変更時の確認事項

1. **fault_posterior の和が常に 1.0** — `softmax` 正規化を変更した場合は `test_fault_posterior_sums_to_one` を再確認。
2. **Failsafe 降下の即時性** — 回復条件変更時は `test_failsafe_descent_immediate` を追加。
3. **IMUResidual の単位一貫性** — `ins_noise_std` を変更した場合、`mahalanobis_dist` の閾値 (2.795) が p=0.95 水準を維持することを確認。
4. **run_id の冪等非保証** — 同一リクエストの再送で異なる `run_id` が生成されることは仕様。キャッシュが必要な場合はクライアント側でリクエストハッシュを管理する。

---

## 21. OSNMA 検証モデル深化

<!-- [F8] TESLA 5チェック階層・per-satellite AuthState・Bayesian 仰角加重・攻撃パターン対応表 -->

### 21.1 TESLA プロトコル検証フロー

TESLA (Timed Efficient Stream Loss-tolerant Authentication) は Galileo OSNMA の中核暗号プロトコル。
受信機は鍵開示遅延 δ = `DISCLOSURE_DELAY` (= 2 サブフレーム) 後に各メッセージを後方検証する。

#### 21.1.1 ハッシュチェーン構造

```
K_0 ←H K_1 ←H K_2 ←H ... ←H K_{n-1}   (右端 = root)

K_i = SHA-256( K_{i+1} || LE32(i) ) [:KEY_BYTES]
KEY_BYTES = 16 (128-bit, OSNMA SIS ICD v1.1)
```

検証規則: 既認証アンカー `K_{i_a}` から候補 `K_i` を確認するには
`hash^{i_a − i}(K_{i_a}) == K_i` を計算する。

#### 21.1.2 5 チェック階層

| # | チェック名 | 判定条件 | 対応する実装フィールド |
|---|---|---|---|
| 1 | `key_chain` | `hash^δ(K_{i+δ}) == K_i` | `VerificationResult.key_valid` |
| 2 | `receipt_safe` | `t_receive < t_disclose = i + δ` | `VerificationResult.receipt_safe` |
| 3 | `mac_valid` | `HMAC-SHA256(K_i, payload)[:MAC_BYTES] == mac_tag` | `VerificationResult.mac_valid` |
| 4 | `quantum_fidelity` | `F(eph_received, eph_expected) > τ_q = 0.85` | `VerificationResult.quantum_anomaly` (反転) |
| 5 | `chain_continuity` | `∃ anchor_epoch > i` (verified keys dict は空でない) | `OSNMAReceiver._verified_keys` が非空 |

**総合判定**:

```
detected = NOT(key_chain AND receipt_safe AND mac_valid) OR quantum_anomaly
```

チェック 1–3 = Galileo OSNMA SIS ICD v1.1 標準検証。
チェック 4 = 量子耐性拡張 (`QuantumFidelityDetector`)。`key_compromise` 攻撃への対策。
チェック 5 = 実装固有。`_verified_keys` が空の場合 `key_valid = False` を強制。

#### 21.1.3 攻撃パターンと検出チェーン

| 攻撃種別 | key_chain | receipt_safe | mac_valid | quantum_fidelity | 検出率 |
|---|---|---|---|---|---|
| `naive_replay` | **✗** | — | — | — | 100 % |
| `modified_replay` | ✓ | ✓ | **✗** | — | 100 % |
| `key_disclosure` | ✓ | **✗** | ✓ | — | 100 % |
| `late_injection` | ✓ | **✗** | ✓ | — | 100 % |
| `key_compromise` | ✓ | ✓ | ✓ | **✗** | ≈ attack_prob (量子層のみ) |

`key_compromise` はすべての TESLA チェックを通過し、量子忠実度レイヤのみで検出される。
`quantum_detections` フィールド (`SimReport`) で独立に計上する。

### 21.2 per-satellite 認証状態型 SatelliteAuthState

1 衛星・1 エポックの完全な認証状態を表す型。`VerificationResult`（`gnss/core.py`）に 1 対 1 で対応する。

| フィールド | 型 | 説明 |
|---|---|---|
| `svid` | `int` | 衛星番号 (Galileo: 1–36, GPS: 1–32) |
| `epoch` | `int` | バッファエポック番号 (メッセージが受信されたエポック) |
| `key_valid` | `bool` | チェック 1: TESLA チェーン検証結果 |
| `mac_valid` | `bool` | チェック 3: HMAC-SHA256 検証結果 |
| `receipt_safe` | `bool` | チェック 2: 受信タイミング安全性 |
| `quantum_anomaly` | `bool` | チェック 4: エフェメリス量子忠実度 < τ_q = 0.85 |
| `detected` | `bool` | `NOT(key ∧ receipt ∧ mac) OR quantum_anomaly` |
| `chain_gap` | `bool` | チェック 5: `_verified_keys` にアンカーが存在しない場合 True |

`SatelliteAuthState` は将来 `OSNMALayerResult` の `per_sat: list[SatelliteAuthState]` として露出予定。
現行 API では `EpochReport.osnma_auth_fraction` に集約されている。

### 21.3 Bayesian 認証ポスタリア (仰角加重集計)

現行実装は単純算術平均 `auth_fraction = n_auth / n_total`。
深化版では衛星間独立性仮定のもとで仰角加重積を使う:

```
w_i = sin²(el_i)                              # 仰角加重 (高仰角 → 信頼度大)

P_auth_i = 1.0   if key_valid ∧ receipt_safe ∧ mac_valid ∧ NOT quantum_anomaly
         = 0.0   otherwise

P_genuiness = (Σ_i w_i · P_auth_i) / (Σ_i w_i)   # 加重平均

p_spoof_bayes = 1 − P_genuiness                    # 融合スコアへの寄与
```

**現行実装との差分**:

| 属性 | 現行 | 深化版 |
|---|---|---|
| 集計方式 | 算術平均 `n_auth/n_total` | 仰角加重平均 |
| 入力 | `list[bool]` | `list[SatelliteAuthState]` + `elevations` |
| 実装箇所 | `OSNMALayer.assess()` | 同メソッドに `elevations: np.ndarray \| None` 引数追加 |
| 融合スコアへの反映 | `score[spoofing] += 0.20 * (1 − auth_fraction)` | `score[spoofing] += 0.20 * p_spoof_bayes` |

仰角データがない場合は等重み (`w_i = 1`) にフォールバックし、現行と等価になる。

### 21.4 鍵チェーン継続性の追跡

`_verified_keys: dict[int, bytes]` のギャップ（検証済み鍵がないエポック区間）はチェーン中断を示す。

| 状態 | anchor の有無 | key_valid | 推奨対応 |
|---|---|---|---|
| 正常 | 近傍に存在 | True / False (メッセージ依存) | 通常フロー |
| チェーン中断 | None | False（強制） | `chain_gap_alert = True` を付与 |
| 遅延開示 | 後続エポックで到着 | True | `flush_expired()` で事後検証 |

`chain_gap_alert` フィールドを `OSNMALayerResult` に追加予定:

```python
# 追加予定フィールド (現行実装では未実装)
chain_gap_alert: bool   # True if _verified_keys が空で key_chain チェックが実行不可能
consecutive_gap: int    # chain_gap が続いた連続エポック数
```

### 21.5 ResilienceTwin との統合

4 本柱融合スコアへの OSNMA の寄与（Layer 1）:

```
# 現行 (gnss/resilience_twin.py: _score_fusion())
score[spoofing] += 0.20 * (1 − auth_fraction)   # = OSNMALayerResult.p_spoof_contribution

# 深化版（仰角加重 Bayesian 集計に置き換え）
score[spoofing] += 0.20 * p_spoof_bayes          # = 仰角加重 P(spoofed)
```

フル TESLA 検証（`VerificationResult` を直接入力とする経路）を組み込むには、
`run_twin_on_observations()` の `osnma_sequence` 引数の型を
`list[list[bool] | None]` から `list[list[SatelliteAuthState] | None]` に変更する必要がある。
後方互換性のため `list[bool]` 形式を受けた場合は全チェックを `True` として扱う移行オプションを提供する。

---

## 22. Calibrate エンドポイント

<!-- [F9] CalibrateRequest / CalibrateResult / POST /v1/calibrate スキーマ定義 -->

### 22.1 目的

受信機の公称雑音パラメータを**既知-正常な観測窓**から統計的に推定し、
後続の `POST /v1/analyze` / `POST /v1/twin/run` の初期化パラメータとして使用する。

キャリブレーション後のパラメータを適用することで、デフォルト値
(`doppler_noise_std=0.30 Hz`, `ins_noise_std=0.05 m/s`) からの乖離による偽陽性アラートを低減できる。

> **前提**: `observations` に含まれるエポックが**全て正常動作中**であることをユーザが保証する。
> 攻撃が混入した窓でキャリブレーションを行うと過大な雑音推定値が得られる。

### 22.2 CalibrateRequest

| フィールド | 型 | 必須 | デフォルト | 説明 |
|---|---|---|---|---|
| `observations` | `ObservationEpoch[]` | Y | — | 正常観測窓 (5–500 件) |
| `n_sats` | `int` | N | `6` | 衛星数 (5–20) |
| `los_vectors` | `float[n_sats][3] \| null` | N | `null` | 既知の LOS ベクトル。省略時は Fibonacci 格子を自動生成 |
| `confidence_level` | `float` | N | `0.95` | χ² 適合検定の有意水準 ∈ (0, 1) |
| `fit_ins_noise` | `bool` | N | `true` | INS 雑音を推定するか。INS データがない場合は自動スキップ |
| `fit_graph_sigma` | `bool` | N | `true` | グラフカーネル帯域幅 σ を Fiedler null model から推定するか |

> 観測数が少ないほど推定精度が低下する。**最低 30 エポック以上**を推奨。

### 22.3 CalibrateResult

| フィールド | 型 | 単位 | 説明 |
|---|---|---|---|
| `doppler_noise_std` | `float` | Hz | 推定ドップラー雑音 1σ |
| `ins_noise_std` | `float \| null` | m/s | 推定 INS 速度雑音 1σ。INS データなしの場合 `null` |
| `graph_sigma` | `float` | Hz | 推定グラフカーネル帯域幅 σ |
| `gmm_fault_prior` | `float` | — | 推定 GM-RAIM 障害事前確率 ∈ [0, 1] |
| `n_epochs_used` | `int` | — | 実際に推定に使用したエポック数 |
| `doppler_chi2_stat` | `float` | — | ドップラーデータの χ² 適合検定統計量 |
| `doppler_chi2_pvalue` | `float` | — | 対応する p 値 (< `confidence_level` → 正規分布仮定棄却) |
| `ins_chi2_pvalue` | `float \| null` | — | INS データの χ² 検定 p 値 (`null` if INS なし) |
| `fit_quality` | `"good" \| "marginal" \| "poor"` | — | 推定品質サマリー |
| `warnings` | `string[]` | — | 注意事項 (例: 観測数不足、非正規性検出) |

#### fit_quality 判定基準

| 判定 | 条件 |
|---|---|
| `good` | `doppler_chi2_pvalue ≥ confidence_level` かつ `n_epochs_used ≥ 30` |
| `marginal` | `doppler_chi2_pvalue ≥ 0.05` かつ `n_epochs_used ≥ 10` |
| `poor` | 上記以外（キャリブレーション結果の信頼性が低い） |

### 22.4 推定アルゴリズム

```
# Step 1: Doppler 雑音推定 (MLE under Gaussian assumption)
all_residuals = flatten(obs.doppler_residuals for obs in observations)
μ̂_D = mean(all_residuals)
σ̂_D = std(all_residuals − μ̂_D)          # unbiased std (ddof=1)

# Gaussianity test: 10-bin χ² goodness-of-fit
chi2_stat, p_value = chi2_gof(all_residuals, dist=N(μ̂_D, σ̂_D), n_bins=10)

# Step 2: INS 雑音推定 (if fit_ins_noise and data available)
ins_norms = [‖obs.ins_velocity_ms‖₂ for obs in observations if obs.ins_velocity_ms is not None]
σ̂_ins = std(ins_norms) / sqrt(3)          # 各軸独立仮定、L2 ノルムから逆算
         if len(ins_norms) >= 5 else None

# Step 3: グラフ帯域幅推定 (if fit_graph_sigma)
# σ* = argmin |mean(ρ_F(σ)) − 1.0|  via bisection on σ ∈ [0.1, 10.0]
def mean_fiedler(σ):
    return mean(SpectralMonitor(σ).analyze(obs).fiedler_ratio for obs in observations)
σ̂_graph = bisect(lambda σ: mean_fiedler(σ) − 1.0, lo=0.1, hi=10.0, tol=1e-3)

# Step 4: GMM 障害事前確率推定
# GMMRaim を推定済み σ̂_D で実行し、gmm_n_fault > 0 となるエポック割合
gmm_fault_prior = count(epochs with gmm_n_fault > 0) / n_epochs_used
```

**定数**:
- 等方性仮定: `Σ = σ̂_D² · I` (ドップラー共分散)
- χ² 適合検定の bin 数: 10（外れ値の少ない範囲 `μ̂ ± 3σ̂` を等区間分割）

### 22.5 エンドポイント定義

```
POST /v1/calibrate
Content-Type: application/json
```

**レスポンス: 200 OK** — `CalibrateResult`

**エラーレスポンス**:

| HTTP コード | 条件 |
|---|---|
| `400 Bad Request` | 衛星数不一致、観測件数 < 5 |
| `422 Unprocessable Entity` | `confidence_level` 範囲外 (≤ 0 or ≥ 1) |

**典型的な使用パターン**:

```python
# Step 1: キャリブレーション (正常窓)
cal = POST("/v1/calibrate", {
    "observations": nominal_window,   # 30+ エポックの正常観測
    "n_sats": 8,
    "fit_graph_sigma": True,
})
# → CalibrateResult { doppler_noise_std: 0.28, graph_sigma: 1.42,
#                     ins_noise_std: 0.047, fit_quality: "good" }

# Step 2: キャリブレーション結果を使って分析
POST("/v1/analyze", {
    "observations": monitoring_window,
    "n_sats": 8,
    "doppler_noise_std": cal["doppler_noise_std"],
    "graph_sigma": cal["graph_sigma"],
    "ins_noise_std": cal["ins_noise_std"],
})
```

---

## 23. Pagination 設計

<!-- [F10] cursor-based pagination: エポックレポート分割取得・レポート一覧 -->

### 23.1 設計方針

`epoch_reports` は最大 5000 エポック（未圧縮 JSON で数 MB）に達する可能性がある。
`GET /v1/reports` 一覧もデプロイ規模に応じて増大する。

**カーソルベースを採用する理由**:

| 比較軸 | オフセットベース (skip/limit) | カーソルベース (after_X) |
|---|---|---|
| 中間挿入 | ページ境界が「ずれる」 | 影響なし |
| 一貫性 | 弱い | 強い |
| 実装コスト | 低 | 低 (epoch は整数、run_id は UUID v7) |
| 任意ジャンプ | 可能 | 不可 (先頭・特定 epoch への直接アクセスは別途) |

`epoch` は単調増加整数のため `after_epoch` カーソルをそのまま整数値として使える。
`run_id` は UUID v7（時刻ソート可能）のため `after_run_id` カーソルも不透明化不要。

### 23.2 GET /v1/reports — レポート一覧

```
GET /v1/reports?limit=20&after_run_id=018f4b2e-7c3a-7d45-b891-0a1234567890&status=completed
```

#### クエリパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `limit` | `int` | `20` | 返却件数 (1–100) |
| `after_run_id` | `string \| null` | `null` | このUUID v7 の直後から返す（初回は省略） |
| `status` | `"completed" \| "running" \| "failed" \| null` | `null` | 状態フィルタ。省略時は全状態 |
| `since` | `datetime \| null` | `null` | 指定 UTC 日時以降の `run_id` のみ (ISO 8601) |

#### レスポンス: 200 OK — ReportListPage

| フィールド | 型 | 説明 |
|---|---|---|
| `items` | `ReportSummary[]` | このページのサマリー一覧 (run_id 昇順) |
| `next_cursor` | `string \| null` | 次ページ先頭 `run_id`。`null` = 最終ページ |
| `total_count` | `int \| null` | フィルタ条件に一致する総件数。コスト高の場合 `null` 許可 |

#### ReportSummary (軽量型)

`epoch_reports` を含まない軽量型。全件取得は `/reports/{run_id}` または `/epochs` を使用。

| フィールド | 型 | 説明 |
|---|---|---|
| `run_id` | `string` | UUID v7 |
| `created_at` | `datetime` | 生成タイムスタンプ (UTC) |
| `status` | `"completed" \| "running" \| "failed"` | 処理状態 |
| `n_epochs` | `int \| null` | 処理エポック数 (未完了時は `null`) |
| `dominant_diagnosis` | `string \| null` | MAP 診断 (未完了時は `null`) |
| `worst_action` | `RecommendedAction \| null` | 最悪アクション (未完了時は `null`) |

### 23.3 GET /v1/reports/{run_id}/epochs — エポックレポートの分割取得

```
GET /v1/reports/018f4b2e-7c3a-7d45-b891-0a1234567890/epochs?limit=100&after_epoch=199
```

#### クエリパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `limit` | `int` | `100` | 返却エポック数 (1–500) |
| `after_epoch` | `int \| null` | `null` | このエポック番号の直後から返す（初回は省略） |

#### レスポンス: 200 OK — EpochReportPage

| フィールド | 型 | 説明 |
|---|---|---|
| `run_id` | `string` | 対象 run_id |
| `items` | `EpochReport[]` | `epoch` 昇順の EpochReport リスト |
| `next_cursor` | `int \| null` | 次ページ先頭 epoch 番号。`null` = 最終ページ |
| `total_epochs` | `int` | このレポートの全エポック数 |
| `page_first_epoch` | `int` | このページ最初の epoch 番号 |
| `page_last_epoch` | `int` | このページ最後の epoch 番号 |

**エラーレスポンス**:

| HTTP コード | 条件 |
|---|---|
| `404 Not Found` | `run_id` が存在しない |
| `400 Bad Request` | `limit` が範囲外 (< 1 または > 500) |
| `202 Accepted` | 処理中 (`status == "running"`) — `items` は空、`total_epochs` は暫定値 |

### 23.4 AnalysisResult への埋め込みページネーション

`GET /v1/reports/{run_id}` は `status == "completed"` のとき、デフォルトで
**先頭 100 エポックのみ** `epoch_reports` に含める。
全件取得には `/epochs` エンドポイントを使用する。

`AnalysisResult` への追加フィールド:

| フィールド | 型 | 説明 |
|---|---|---|
| `epoch_reports_truncated` | `bool` | `true` のとき `epoch_reports` は全エポックの部分集合 |
| `epoch_reports_next_cursor` | `int \| null` | `truncated == true` のとき続きの `after_epoch` 値 |

```json
{
  "run_id": "018f4b2e-...",
  "status": "completed",
  "n_epochs": 5000,
  "epoch_reports": [ /* epoch 0–99 のみ */ ],
  "epoch_reports_truncated": true,
  "epoch_reports_next_cursor": 100
}
```

### 23.5 カーソルエンコード仕様

| エンドポイント | カーソルフィールド | 型 | 不透明化 | 例 |
|---|---|---|---|---|
| `GET /v1/reports` | `after_run_id` | UUID v7 文字列 | 不要 (時刻ソート可能) | `018f4b2e-7c3a-7d45-b891-0a1234567890` |
| `GET /v1/reports/{id}/epochs` | `after_epoch` | 整数 | 不要 (単調増加) | `199` |

カーソルが指す項目自体は返却しない（exclusive lower bound）。
カーソルを偽造しても取得できるのは自身のデータのみであり、認可制御は `run_id` スコープで行う。

### 23.6 クライアント実装例

```python
# 全エポックを逐次取得するジェネレータ
def iter_epoch_reports(run_id: str, page_size: int = 200):
    after_epoch = None
    while True:
        params: dict = {"limit": page_size}
        if after_epoch is not None:
            params["after_epoch"] = after_epoch
        page = GET(f"/v1/reports/{run_id}/epochs", params=params)
        yield from page["items"]
        if page["next_cursor"] is None:
            break
        after_epoch = page["next_cursor"]

# レポート一覧を全件取得
def iter_all_reports(status: str | None = None):
    after_run_id = None
    while True:
        params: dict = {"limit": 100}
        if after_run_id is not None:
            params["after_run_id"] = after_run_id
        if status is not None:
            params["status"] = status
        page = GET("/v1/reports", params=params)
        yield from page["items"]
        if page["next_cursor"] is None:
            break
        after_run_id = page["next_cursor"]
```
