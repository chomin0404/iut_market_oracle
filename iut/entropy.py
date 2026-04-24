import numpy as np

ENTROPY_THRESHOLD = 1.5
HISTOGRAM_BINS = 10


def calculate_shannon_entropy(return_series: np.ndarray) -> float:
    """リターン系列のヒストグラムからシャノン・エントロピーを算出する（単位: nats）。"""
    counts, _ = np.histogram(return_series, bins=HISTOGRAM_BINS, density=False)
    probs = counts / (counts.sum() + 1e-9)
    return float(-np.sum(probs * np.log(probs + 1e-9)))


def entropy_guarded_filter(
    candidates: list[str],
    window_data: dict[str, np.ndarray],
) -> list[dict]:
    """エントロピーが閾値以下の「信頼できる歪み」のみを抽出する。"""
    guarded_results = []

    for ticker in candidates:
        if ticker not in window_data:
            print(f"[WARN] {ticker}: window_data に存在しません")
            continue

        entropy = calculate_shannon_entropy(window_data[ticker])

        if entropy < ENTROPY_THRESHOLD:
            guarded_results.append(
                {
                    "ticker": ticker,
                    "entropy": entropy,
                    "status": "Reliable Distortion",
                }
            )

    return guarded_results
