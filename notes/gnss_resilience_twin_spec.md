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
| `GET` | `/v1/reports/{run_id}` | 分析結果の取得 (realtime / batch 共通) |
| `GET` | `/v1/reports/{run_id}/status` | 処理状態のみを取得 (batch ポーリング用) |
| `DELETE` | `/v1/reports/{run_id}` | 結果の削除 |
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
