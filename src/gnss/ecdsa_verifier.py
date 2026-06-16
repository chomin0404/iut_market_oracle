"""ECDSA Verifier — ICD §5.4.4 / §5.5.2.

DSM-KROOT digital signature verification (ECDSA-P256 / P521) and
DSM-PKR Merkle Tree public key authentication for Galileo OSNMA.

Raw ICD signature format: r || s (big-endian, fixed width per curve).
  P-256: 64 bytes (32r + 32s)
  P-521: 132 bytes (66r + 66s)
"""

from __future__ import annotations

import hashlib
import logging

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ec import (
    ECDSA,
    SECP256R1,
    SECP521R1,
    EllipticCurvePublicKey,
)
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.hashes import SHA256, SHA512

from core.data_structures import DSMKROOTMessage, DSMPKRMessage, ECDSAType

_log = logging.getLogger(__name__)

# Raw signature sizes per curve (ICD §5.4.4 Table)
_DS_BYTES: dict[ECDSAType, int] = {
    ECDSAType.P256: 64,  # 32r + 32s
    ECDSAType.P521: 132,  # 66r + 66s
}
_SCALAR_BYTES: dict[ECDSAType, int] = {
    ECDSAType.P256: 32,
    ECDSAType.P521: 66,
}


class ECDSAVerifier:
    """ECDSA署名・Merkle Tree検証エンジン (ICD §5.4.4 / §5.5.2).

    DSM-KROOT デジタル署名の検証と DSM-PKR の Merkle Tree 公開鍵認証を行う。

    Usage::

        verifier = ECDSAVerifier(ECDSAType.P256)
        ok_kroot = verifier.verify_kroot(kroot_msg, nma_hdr_byte=0x05, public_key=pk)
        ok_pkr   = verifier.verify_public_key(pkr_msg, merkle_root=root)

    Args:
        ecdsa_type: 使用する ECDSA 曲線/ハッシュ組み合わせ (ECDSAType enum)。
    """

    def __init__(self, ecdsa_type: ECDSAType) -> None:
        self.ecdsa_type = ecdsa_type

    def verify_kroot(
        self,
        kroot: DSMKROOTMessage,
        nma_hdr_byte: int,
        public_key: EllipticCurvePublicKey,
    ) -> bool:
        """DSM-KROOT デジタル署名を検証する (ICD §5.4.4).

        M_KROOT = NMA_Header(1B) || DSM-KROOT body を構築し、
        ICD raw (r || s) フォーマットの署名を DER に変換してから検証する。

        Args:
            kroot:        デコード済み DSM-KROOT メッセージ。
            nma_hdr_byte: NMA ヘッダーバイト (ビット 7–0)。
            public_key:   ECDSA 公開鍵オブジェクト。

        Returns:
            署名が有効なら ``True``、無効または鍵の型が不一致なら ``False``。
        """
        if not self._check_key_curve(public_key):
            _log.warning("verify_kroot: key curve mismatch for ecdsa_type=%s", self.ecdsa_type)
            return False

        scalar = _SCALAR_BYTES[self.ecdsa_type]
        ds = kroot.ds
        expected_len = scalar * 2
        if len(ds) != expected_len:
            _log.warning("verify_kroot: ds length %d != expected %d", len(ds), expected_len)
            return False

        r = int.from_bytes(ds[:scalar], "big")
        s = int.from_bytes(ds[scalar:], "big")
        der_sig = encode_dss_signature(r, s)

        m_kroot = kroot.build_m_kroot(nma_hdr_byte)
        hash_alg = SHA256() if self.ecdsa_type == ECDSAType.P256 else SHA512()

        try:
            public_key.verify(der_sig, m_kroot, ECDSA(hash_alg))
            return True
        except InvalidSignature:
            _log.debug("verify_kroot: signature invalid")
            return False
        except (UnsupportedAlgorithm, ValueError, TypeError) as exc:
            _log.warning("verify_kroot: unexpected error — %s", exc)
            return False

    def verify_public_key(
        self,
        pkr: DSMPKRMessage,
        merkle_root: bytes,
    ) -> bool:
        """DSM-PKR の Merkle Tree 検証 (ICD §5.5.2).

        リーフ = SHA-256(PKID(1B) || PKTYPE(1B) || public_key)
        PKID の LSB が 0 なら左子・右に兄弟、1 なら右子・左に兄弟として
        ルートまでハッシュを辿る。

        Args:
            pkr:         デコード済み DSM-PKR メッセージ。
            merkle_root: 信頼された Merkle ルート (32 バイト, SHA-256)。

        Returns:
            Merkle 経路が ``merkle_root`` に到達すれば ``True``。
        """
        leaf = hashlib.sha256(bytes([pkr.pkid, pkr.pktype.value]) + pkr.public_key).digest()

        node, idx = leaf, pkr.pkid
        for sibling in pkr.merkle_nodes:
            if idx % 2 == 0:
                combined = node + sibling
            else:
                combined = sibling + node
            node = hashlib.sha256(combined).digest()
            idx //= 2

        ok = node == merkle_root
        if not ok:
            _log.debug("verify_public_key: Merkle root mismatch")
        return ok

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_key_curve(self, public_key: EllipticCurvePublicKey) -> bool:
        """公開鍵の曲線が ecdsa_type と一致するか確認する。"""
        curve = public_key.curve
        if self.ecdsa_type == ECDSAType.P256:
            return isinstance(curve, SECP256R1)
        return isinstance(curve, SECP521R1)
