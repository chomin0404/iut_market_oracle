import numpy as np
import yfinance as yf

from .entropy import entropy_guarded_filter
from .oracle import IUTValueOracle
from .resonance import ResonanceGuard
from .screener import screen_iut_distortion

LOOKBACK_DAYS = 60
RESONANCE_FREQ_FALLBACK = 0.05


def fetch_returns(ticker: str, lookback_days: int = LOOKBACK_DAYS) -> np.ndarray | None:
    """対数リターン系列を取得する。取得失敗時は None を返す。"""
    try:
        df = yf.download(ticker, period=f"{lookback_days}d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        prices = df["Close"].to_numpy(dtype=float).flatten()
        return np.diff(np.log(prices + 1e-9))
    except Exception as exc:
        print(f"[WARN] {ticker}: {exc}")
        return None


def dominant_frequency(returns: np.ndarray) -> float:
    """パワースペクトルの支配周波数を返す（DC成分を除く）。"""
    n = len(returns)
    freqs = np.fft.rfftfreq(n)
    power = np.abs(np.fft.rfft(returns)) ** 2
    dominant_idx = int(np.argmax(power[1:])) + 1
    return float(freqs[dominant_idx])


def run_analysis_pipeline(tickers: list[str]) -> list[dict]:
    """
    IUT パイプラインを実行し、結果を辞書のリストで返す。

    Returns:
        各ティッカーの分析結果。スクリーニング・エントロピーフィルタを
        通過した銘柄のみ含まれる。
    """
    # Step 1: IUT歪みスクリーニング
    candidates = screen_iut_distortion(tickers)
    if not candidates:
        return []

    candidate_tickers = [c["ticker"] for c in candidates]
    distortion_map = {c["ticker"]: c["distortion"] for c in candidates}

    # Step 2: リターンデータ取得
    window_data: dict[str, np.ndarray] = {}
    for ticker in candidate_tickers:
        r = fetch_returns(ticker)
        if r is not None:
            window_data[ticker] = r

    if not window_data:
        return []

    # Step 3: エントロピーフィルタ
    reliable = entropy_guarded_filter(candidate_tickers, window_data)
    if not reliable:
        return []

    # Step 4: ResonanceGuard + IUTValueOracle
    results: list[dict] = []
    for item in reliable:
        ticker = item["ticker"]
        returns = window_data[ticker]

        freq = dominant_frequency(returns)
        if freq <= 0:
            freq = RESONANCE_FREQ_FALLBACK

        guard = ResonanceGuard(resonance_freq=freq)
        integrity = guard.check_integrity(returns)
        purity = guard.analyze_voigt_profile(returns)

        oracle = IUTValueOracle(data_stream=returns)
        oracle.construct_hodge_theater()
        oracle.apply_theta_link(q_parameter=float(returns.std()))
        reconstructed, recon_entropy = oracle.reconstruct_structure()

        results.append(
            {
                "ticker": ticker,
                "distortion": round(distortion_map[ticker], 6),
                "entropy": round(item["entropy"], 6),
                "resonance_status": integrity,
                "purity": round(purity, 6),
                "reconstructed_mean": round(float(reconstructed.mean()), 6),
                "recon_entropy": round(recon_entropy, 6),
            }
        )

    return results
