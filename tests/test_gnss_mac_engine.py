"""Tests for src/gnss/mac_engine.py — OSNMAMacEngine."""

from __future__ import annotations

import math
import struct

import pytest

from core.data_structures import ADKD, MACFunction
from gnss.mac_engine import _SECONDS_PER_WEEK, OSNMAMacEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HMAC_ENGINE = OSNMAMacEngine(MACFunction.HMAC_SHA_256, tag_size_bits=40)
CMAC_ENGINE = OSNMAMacEngine(MACFunction.CMAC_AES, tag_size_bits=40)

KEY_16 = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
NAV_DATA = b"\xde\xad\xbe\xef" * 4
GST_SF = 345630  # WN=4, TOW=6030
PRN_A = 1
PRN_D = 1
CTR = 0


# ---------------------------------------------------------------------------
# TestOSNMAMacEngineInit
# ---------------------------------------------------------------------------


class TestOSNMAMacEngineInit:
    def test_rejects_zero_bits(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            OSNMAMacEngine(MACFunction.HMAC_SHA_256, tag_size_bits=0)

    def test_accepts_icd_tag_sizes(self) -> None:
        # ICD §5.3 Table: TS=0→20b, TS=1→24b, TS=2→28b, TS=3→32b, TS=4→36b, TS=5→40b
        for bits in (20, 24, 28, 32, 36, 40):
            e = OSNMAMacEngine(MACFunction.HMAC_SHA_256, tag_size_bits=bits)
            assert e.tag_size_bits == bits


# ---------------------------------------------------------------------------
# TestComputeTag
# ---------------------------------------------------------------------------


class TestComputeTag:
    def test_output_length(self) -> None:
        tag = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert len(tag) == 40 // 8

    def test_deterministic(self) -> None:
        t1 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        t2 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert t1 == t2

    def test_different_prn_a_gives_different_tag(self) -> None:
        t1 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, 1, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        t2 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, 2, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert t1 != t2

    def test_different_gst_gives_different_tag(self) -> None:
        t1 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        t2 = HMAC_ENGINE.compute_tag(
            KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF + 30, ADKD.INAV_CED, CTR
        )
        assert t1 != t2

    def test_different_ctr_gives_different_tag(self) -> None:
        t1 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, 0)
        t2 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, 1)
        assert t1 != t2

    def test_different_key_gives_different_tag(self) -> None:
        key2 = bytes(reversed(KEY_16))
        t1 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        t2 = HMAC_ENGINE.compute_tag(key2, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert t1 != t2

    def test_different_nav_data_gives_different_tag(self) -> None:
        t1 = HMAC_ENGINE.compute_tag(KEY_16, b"\x00" * 16, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        t2 = HMAC_ENGINE.compute_tag(KEY_16, b"\xff" * 16, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert t1 != t2

    def test_adkd_does_not_affect_mac(self) -> None:
        """ADKD is nav-data selector metadata — NOT part of MAC input per §5.6.3."""
        t1 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        t2 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_TIMING, CTR)
        assert t1 == t2

    def test_prn_d_does_not_affect_mac(self) -> None:
        """PRN_D is cross-auth satellite selector — NOT part of MAC input per §5.6.3."""
        t1 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, 1, GST_SF, ADKD.INAV_CED, CTR)
        t2 = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, 7, GST_SF, ADKD.INAV_CED, CTR)
        assert t1 == t2

    def test_nma_status_affects_tag(self) -> None:
        t1 = HMAC_ENGINE.compute_tag(
            KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR, nma_status=0
        )
        t2 = HMAC_ENGINE.compute_tag(
            KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR, nma_status=1
        )
        assert t1 != t2

    def test_adkd_as_int(self) -> None:
        """Accept plain int for adkd (backward-compat)."""
        t_enum = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        t_int = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, 0, CTR)
        assert t_enum == t_int


# ---------------------------------------------------------------------------
# TestCTRHeaderLayout
# ---------------------------------------------------------------------------


class TestCTRHeaderLayout:
    """Verify the 8-byte CTR header bit layout against manual construction."""

    def _build_ctr_header(self, prn_a: int, gst_sf: int, ctr: int, nma_status: int) -> bytes:
        wn = (gst_sf // _SECONDS_PER_WEEK) & 0xFFF
        tow = (gst_sf % _SECONDS_PER_WEEK) & 0xFFFFF
        value = (
            ((prn_a & 0xFF) << 56)
            | ((wn & 0xFFF) << 44)
            | ((tow & 0xFFFFF) << 24)
            | ((ctr & 0xFF) << 16)
            | ((nma_status & 0x3) << 14)
        )
        return struct.pack(">Q", value)

    def _expected_tag(
        self,
        engine: OSNMAMacEngine,
        key: bytes,
        nav_data: bytes,
        prn_a: int,
        gst_sf: int,
        ctr: int,
        nma_status: int,
    ) -> bytes:
        hdr = self._build_ctr_header(prn_a, gst_sf, ctr, nma_status)
        import hashlib
        import hmac as _hmac

        digest = _hmac.new(key, hdr + nav_data, hashlib.sha256).digest()
        return engine._trunc_msb(digest, engine.tag_size_bits)

    def test_header_matches_manual(self) -> None:
        expected = self._expected_tag(HMAC_ENGINE, KEY_16, NAV_DATA, PRN_A, GST_SF, CTR, 1)
        actual = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert actual == expected

    def test_wn_tow_decomposition(self) -> None:
        """GST=0 → WN=0, TOW=0; GST=604800 → WN=1, TOW=0."""
        hdr_week0 = self._build_ctr_header(1, 0, 0, 1)
        hdr_week1 = self._build_ctr_header(1, _SECONDS_PER_WEEK, 0, 1)
        assert hdr_week0 != hdr_week1
        # WN field: bits 55-44 of 64-bit big-endian value
        val0 = struct.unpack(">Q", hdr_week0)[0]
        val1 = struct.unpack(">Q", hdr_week1)[0]
        wn0 = (val0 >> 44) & 0xFFF
        wn1 = (val1 >> 44) & 0xFFF
        assert wn0 == 0
        assert wn1 == 1

    def test_prn_a_in_top_byte(self) -> None:
        hdr = self._build_ctr_header(prn_a=0xAB, gst_sf=0, ctr=0, nma_status=0)
        assert hdr[0] == 0xAB

    def test_padding_bits_zero(self) -> None:
        """Lower 14 bits of CTR header must be zero (padding)."""
        hdr = self._build_ctr_header(prn_a=1, gst_sf=GST_SF, ctr=CTR, nma_status=1)
        val = struct.unpack(">Q", hdr)[0]
        assert (val & 0x3FFF) == 0


# ---------------------------------------------------------------------------
# TestVerifyTag
# ---------------------------------------------------------------------------


class TestVerifyTag:
    def test_verify_correct_tag(self) -> None:
        tag = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        ok = HMAC_ENGINE.verify_tag(tag, KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert ok

    def test_reject_tampered_tag(self) -> None:
        tag = bytearray(
            HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        )
        tag[0] ^= 0xFF
        assert not HMAC_ENGINE.verify_tag(
            bytes(tag), KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR
        )

    def test_reject_wrong_key(self) -> None:
        tag = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        wrong_key = bytes(b ^ 0x01 for b in KEY_16)
        assert not HMAC_ENGINE.verify_tag(
            tag, wrong_key, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR
        )

    def test_reject_tampered_nav_data(self) -> None:
        tag = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        bad_nav = bytes(b ^ 0x01 for b in NAV_DATA)
        assert not HMAC_ENGINE.verify_tag(
            tag, KEY_16, bad_nav, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR
        )

    def test_constant_time_comparison(self) -> None:
        """verify_tag uses hmac.compare_digest (constant-time)."""
        # Smoke test: timing attack prevention verified by using compare_digest.
        # We verify structural correctness here, not timing guarantees.
        tag = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        ok = HMAC_ENGINE.verify_tag(tag, KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert ok is True


# ---------------------------------------------------------------------------
# TestCMACVariant
# ---------------------------------------------------------------------------


class TestCMACVariant:
    def test_cmac_output_length(self) -> None:
        tag = CMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert len(tag) == 40 // 8

    def test_cmac_deterministic(self) -> None:
        t1 = CMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        t2 = CMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert t1 == t2

    def test_cmac_differs_from_hmac(self) -> None:
        t_hmac = HMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        t_cmac = CMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert t_hmac != t_cmac

    def test_cmac_verify_roundtrip(self) -> None:
        tag = CMAC_ENGINE.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        ok = CMAC_ENGINE.verify_tag(tag, KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert ok

    def test_cmac_short_key_padded(self) -> None:
        """Keys shorter than 16 bytes are zero-padded to AES-128 length."""
        short_key = b"\xaa" * 8
        tag = CMAC_ENGINE.compute_tag(short_key, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert len(tag) == 40 // 8

    def test_cmac_long_key_truncated(self) -> None:
        """Keys longer than 16 bytes are truncated to AES-128 length."""
        long_key = KEY_16 + b"\xff" * 16  # 32 bytes
        tag = CMAC_ENGINE.compute_tag(long_key, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert len(tag) == 40 // 8


# ---------------------------------------------------------------------------
# TestTruncMsb
# ---------------------------------------------------------------------------


class TestTruncMsb:
    def test_full_bytes_only(self) -> None:
        data = bytes(range(32))
        assert OSNMAMacEngine._trunc_msb(data, 24) == bytes(range(3))

    def test_40_bits_from_sha256(self) -> None:
        """Standard OSNMA tag truncation: 40 bits = 5 bytes."""
        data = b"\xab\xcd\xef\x12\x34\x56\x78"
        result = OSNMAMacEngine._trunc_msb(data, 40)
        assert result == b"\xab\xcd\xef\x12\x34"

    def test_partial_byte_4_bits(self) -> None:
        # 12 bits = 1 full byte + 4 bits of next byte
        data = b"\xab\xcd\xef"
        result = OSNMAMacEngine._trunc_msb(data, 12)
        assert len(result) == 2
        assert result[0] == 0xAB
        assert result[1] == 0xC0  # top 4 bits of 0xCD = 0xC, LSB masked off

    def test_partial_byte_1_bit(self) -> None:
        data = b"\xff\xff"
        result = OSNMAMacEngine._trunc_msb(data, 9)
        assert len(result) == 2
        assert result[0] == 0xFF
        assert result[1] == 0x80  # top bit of 0xFF

    def test_exact_byte_boundary(self) -> None:
        data = b"\xde\xad\xbe\xef"
        result = OSNMAMacEngine._trunc_msb(data, 32)
        assert result == data

    def test_single_byte(self) -> None:
        data = b"\xf0\x00"
        assert OSNMAMacEngine._trunc_msb(data, 8) == b"\xf0"


# ---------------------------------------------------------------------------
# TestTagSizeVariants
# ---------------------------------------------------------------------------


class TestTagSizeVariants:
    @pytest.mark.parametrize("bits", [20, 24, 40, 64, 80, 96])
    def test_output_length_matches_bits(self, bits: int) -> None:
        engine = OSNMAMacEngine(MACFunction.HMAC_SHA_256, tag_size_bits=bits)
        tag = engine.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert len(tag) == math.ceil(bits / 8)

    @pytest.mark.parametrize("bits", [20, 24, 40, 64])
    def test_verify_roundtrip_various_sizes(self, bits: int) -> None:
        engine = OSNMAMacEngine(MACFunction.HMAC_SHA_256, tag_size_bits=bits)
        tag = engine.compute_tag(KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
        assert engine.verify_tag(tag, KEY_16, NAV_DATA, PRN_A, PRN_D, GST_SF, ADKD.INAV_CED, CTR)
