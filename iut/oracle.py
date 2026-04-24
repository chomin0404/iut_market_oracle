import numpy as np
from scipy.fft import fft, ifft

RANDOM_SEED = 42


class IUTValueOracle:
    """IUT理論の「テータ・リンク」と「構造復元」を模倣した価値計算エンジン。"""

    def __init__(self, data_stream: np.ndarray, complexity_epsilon: float = 0.01) -> None:
        self.data = data_stream
        self.eps = complexity_epsilon
        self.hodge_theater: dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(RANDOM_SEED)

    def construct_hodge_theater(self) -> str:
        self.hodge_theater["universe_alpha"] = self.data
        self.hodge_theater["universe_beta"] = np.log(np.abs(self.data) + 1e-9)
        return "Theaters established: Log-Link enabled."

    def apply_theta_link(self, q_parameter: float) -> np.ndarray:
        if "universe_beta" not in self.hodge_theater:
            raise RuntimeError("construct_hodge_theater() を先に呼び出してください")
        link_noise = self._rng.normal(0, self.eps, len(self.data))
        self.hodge_theater["linked_data"] = (
            self.hodge_theater["universe_beta"] * q_parameter + link_noise
        )
        return self.hodge_theater["linked_data"]

    def reconstruct_structure(self) -> tuple[np.ndarray, float]:
        """位相のみ復元（IUTのθリンク: 構造不変量の抽出）。"""
        if "linked_data" not in self.hodge_theater:
            raise RuntimeError("apply_theta_link() を先に呼び出してください")

        linked = self.hodge_theater["linked_data"]
        freqs = fft(linked)
        phase_only = np.exp(1j * np.angle(freqs))
        reconstructed_log = ifft(phase_only).real
        final_value = np.exp(reconstructed_log).real

        p = np.abs(final_value)
        p = p / (p.sum() + 1e-9)
        entropy = float(-np.sum(p * np.log(p + 1e-9)))

        return final_value, entropy
