"""MAC Engine — ICD §5.3 / §5.6.3.

HMAC-SHA-256 / CMAC-AES-128 authentication tag computation and verification
for Galileo OSNMA MACK section tags.

MACK タグ CTR フィールド (8 バイト, big-endian):
  ┌────────┬────────┬──────────┬────────┬───────┬────────────┐
  │PRN_A   │WN      │TOW       │CTR     │NMAS   │padding     │
  │ 8 bit  │12 bit  │20 bit    │ 8 bit  │ 2 bit │14 bit (=0) │
  └────────┴────────┴──────────┴────────┴───────┴────────────┘
  PRN_A: 認証衛星 SVID (ブロードキャスト元)
  WN   : GST Week Number = gst_sf // 604800
  TOW  : GST Time of Week = gst_sf % 604800
  CTR  : MACK ブロック内のタグカウンタ (0-indexed)
  NMAS : NMA Status (2 bit)

このフォーマットは §5.4.1 の自己認証タグ (SVID||GST[4B]||ADKD/COP||NMAS||NavData)
とは異なる。ADKD はタグ選択メタデータであり MAC 入力には含まれない。
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import logging
import struct

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.cmac import CMAC

from core.data_structures import ADKD, MACFunction

_log = logging.getLogger(__name__)

# GST 時刻分解定数
_SECONDS_PER_WEEK: int = 604800
_WN_MASK: int = 0xFFF  # 12-bit WN field
_TOW_MASK: int = 0xFFFFF  # 20-bit TOW field


class OSNMAMacEngine:
    """OSNMA MAC 計算・検証エンジン (ICD §5.3 / §5.6.3).

    MACK セクションの認証タグを HMAC-SHA-256 または CMAC-AES-128 で
    計算・検証する。タグは MSB 側から ``tag_size_bits`` ビットに切り捨てる。

    Usage::

        engine = OSNMAMacEngine(MACFunction.HMAC_SHA_256, tag_size_bits=40)
        tag = engine.compute_tag(key, nav_data, prn_a=1, prn_d=1,
                                 gst_sf=345600, adkd=ADKD.INAV_CED, ctr=0)
        ok = engine.verify_tag(tag, key, nav_data, prn_a=1, prn_d=1,
                               gst_sf=345600, adkd=ADKD.INAV_CED, ctr=0)

    Args:
        mac_func:      MAC アルゴリズム選択 (MACFunction enum)。
        tag_size_bits: タグ切り捨てビット数。8 の倍数でなければならない。
    """

    def __init__(self, mac_func: MACFunction, tag_size_bits: int) -> None:
        if tag_size_bits < 1:
            raise ValueError(f"tag_size_bits must be positive, got {tag_size_bits}")
        self.mac_func = mac_func
        self.tag_size_bits = tag_size_bits

    def compute_tag(
        self,
        key: bytes,
        nav_data: bytes,
        prn_a: int,
        prn_d: int,  # metadata only — not included in MAC input per §5.6.3
        gst_sf: int,
        adkd: ADKD | int,  # metadata only — not included in MAC input per §5.6.3
        ctr: int,
        nma_status: int = 0b01,
    ) -> bytes:
        """MACK タグを計算して返す。

        MAC 入力 = CTR_header(8B) || nav_data
        where CTR_header = PRN_A(8b) || WN(12b) || TOW(20b) || CTR(8b) || NMAS(2b) || pad(14b)

        Args:
            key:        TESLA 鍵 K_i。
            nav_data:   認証対象ナビゲーションデータ。
            prn_a:      認証衛星 SVID (ブロードキャスト元, 1–36)。
            prn_d:      データ提供衛星 SVID (クロス認証時に PRN_A と異なる場合あり)。
                        MAC 入力には含まれない — nav_data 選択のメタデータ。
            gst_sf:     サブフレーム開始 GST [s]。WN と TOW に分解される。
            adkd:       認証データ種別 (ADKD enum または int)。
                        MAC 入力には含まれない — nav_data 選択のメタデータ。
            ctr:        MACK ブロック内タグカウンタ (0–255)。
            nma_status: NMA ステータス (0–3)。デフォルト 1 = OPERATIONAL。

        Returns:
            ``tag_size_bits // 8`` バイトの切り捨て MAC タグ。
        """
        wn = (gst_sf // _SECONDS_PER_WEEK) & _WN_MASK
        tow = (gst_sf % _SECONDS_PER_WEEK) & _TOW_MASK
        ctr_header = struct.pack(
            ">Q",
            ((prn_a & 0xFF) << 56)
            | ((wn & _WN_MASK) << 44)
            | ((tow & _TOW_MASK) << 24)
            | ((ctr & 0xFF) << 16)
            | ((nma_status & 0x3) << 14),
        )
        return self._trunc_msb(self._raw_mac(key, ctr_header + nav_data), self.tag_size_bits)

    def verify_tag(
        self,
        received: bytes,
        key: bytes,
        nav_data: bytes,
        prn_a: int,
        prn_d: int,
        gst_sf: int,
        adkd: ADKD | int,
        ctr: int,
        nma_status: int = 0b01,
    ) -> bool:
        """受信タグを検証する (定数時間比較)。

        Args:
            received:   受信した MAC タグ (``tag_size_bits // 8`` バイト)。
            その他:     :meth:`compute_tag` と同じ。

        Returns:
            受信タグが期待値と一致すれば ``True``。
        """
        expected = self.compute_tag(key, nav_data, prn_a, prn_d, gst_sf, adkd, ctr, nma_status)
        return _hmac.compare_digest(received, expected)

    def _raw_mac(self, key: bytes, msg: bytes) -> bytes:
        """HMAC-SHA-256 または CMAC-AES-128 の生ダイジェストを返す。"""
        if self.mac_func == MACFunction.HMAC_SHA_256:
            return _hmac.new(key, msg, hashlib.sha256).digest()
        # CMAC-AES-128: 鍵を 16 バイトにパディング/切り捨て
        aes_key = key.ljust(16, b"\x00")[:16]
        c = CMAC(algorithms.AES(aes_key), backend=default_backend())
        c.update(msg)
        return c.finalize()

    @staticmethod
    def _trunc_msb(data: bytes, bits: int) -> bytes:
        """MSB 側から ``bits`` ビットを切り出す。

        bits が 8 の倍数でない場合、最終バイトの LSB 側をゼロマスクする。

        Args:
            data: 入力バイト列 (len(data)*8 >= bits を前提)。
            bits: 切り出すビット数。

        Returns:
            ``ceil(bits/8)`` バイトの切り捨て結果。
        """
        full, rem = divmod(bits, 8)
        result = bytearray(data[:full])
        if rem:
            result.append(data[full] & (0xFF << (8 - rem)) & 0xFF)
        return bytes(result)
