import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

PURITY_THRESHOLD = 1.2


class ResonanceGuard:
    """フォークトプロファイルによるスペクトル純度監視。"""

    def __init__(self, resonance_freq: float) -> None:
        self.target_freq = resonance_freq

    def analyze_voigt_profile(self, market_wave: np.ndarray) -> float:
        """パワースペクトルに Voigt 関数をフィットし SNR（purity）を返す。"""
        wave = (market_wave - market_wave.mean()) / (market_wave.std() + 1e-9)

        n = len(wave)
        freqs = np.fft.rfftfreq(n)
        power = np.abs(np.fft.rfft(wave)) ** 2

        center_idx = int(np.argmin(np.abs(freqs - self.target_freq)))
        half_win = max(5, n // 20)
        win = slice(
            max(0, center_idx - half_win),
            min(len(freqs), center_idx + half_win),
        )
        x_win = freqs[win] - self.target_freq
        y_win = power[win]

        def voigt_model(
            x: np.ndarray,
            amplitude: float,
            sigma: float,
            gamma: float,
            offset: float,
        ) -> np.ndarray:
            return amplitude * voigt_profile(x, sigma, gamma) + offset

        try:
            p0 = [float(y_win.max()), 0.01, 0.01, float(y_win.min())]
            popt, _ = curve_fit(voigt_model, x_win, y_win, p0=p0, maxfev=2000)
            amplitude, _, _, offset = popt
            purity = float(amplitude) / (abs(float(offset)) + 1e-9)
        except RuntimeError:
            purity = float(power[center_idx]) / (float(power.mean()) + 1e-9)

        return purity

    def check_integrity(self, market_wave: np.ndarray) -> str:
        purity = self.analyze_voigt_profile(market_wave)
        if purity < PURITY_THRESHOLD:
            return "EVADE: Invariant Subspace Active"
        return "STABLE: Resonance Maintained"
