"""Unit tests for src/gnss/parser/ and src/gnss/utils/gst_utils.py.

Coverage targets:
  - BitReader: read_uint, read_bytes, skip, seek, peek, read_bool, alignment guard
  - gst_utils: pack/unpack, gst_to_bytes, arithmetic, datetime round-trips
  - hkroot_parser: parse_hkroot_section, DsmKroot assembly, parse_dsm_kroot
  - mack_parser: parse_mack_section (normal, no-key, cross-tags)
  - inav_parser: OSNMAPage validation, _extract_bits, INavAccumulator end-to-end
"""

from __future__ import annotations

import struct

import pytest

from gnss.osnma_inav import (
    SUBFRAME_DURATION_S,
)
from gnss.parser._bit_io import BitReader
from gnss.parser.hkroot_parser import (
    DSM_BLOCKS_PER_MESSAGE,
    DSM_TOTAL_BITS,
    HKROOT_BITS,
    DsmKroot,
    parse_dsm_kroot,
    parse_hkroot_section,
)
from gnss.parser.hkroot_parser import (
    NMA_STATUS_OPERATIONAL as HP_NMA_OPERATIONAL,
)
from gnss.parser.inav_parser import (
    MACK_BITS,
    OSNMA_BYTES_PER_PAGE,
    PAGES_PER_SUBFRAME,
    DecodedSubframe,
    INavAccumulator,
    OSNMAPage,
    _extract_bits,
)
from gnss.parser.mack_parser import (
    ADKD_INAV_CED,
    parse_mack_section,
)
from gnss.parser.mack_parser import (
    MACK_BITS as MACK_BITS_M,
)
from gnss.utils.gst_utils import (
    GST_EPOCH,
    SECONDS_PER_WEEK,
    datetime_to_gst,
    gst_to_bytes,
    gst_to_datetime,
    gst_to_seconds_total,
    pack_gst,
    seconds_to_gst,
    subframe_aligned_gst,
    subframe_index,
    unpack_gst,
)

# ===========================================================================
# BitReader
# ===========================================================================


class TestBitReader:
    def test_read_nibbles(self) -> None:
        r = BitReader(b"\xab")
        assert r.read_uint(4) == 0xA
        assert r.read_uint(4) == 0xB

    def test_read_full_byte(self) -> None:
        r = BitReader(b"\xff")
        assert r.read_uint(8) == 0xFF

    def test_read_zero_byte(self) -> None:
        r = BitReader(b"\x00")
        assert r.read_uint(8) == 0

    def test_position_advances(self) -> None:
        r = BitReader(b"\xab\xcd")
        r.read_uint(4)
        assert r.position == 4
        r.read_uint(4)
        assert r.position == 8

    def test_remaining(self) -> None:
        r = BitReader(b"\xab\xcd")
        assert r.remaining == 16
        r.read_uint(4)
        assert r.remaining == 12

    def test_eof_raises(self) -> None:
        r = BitReader(b"\xff")
        with pytest.raises(EOFError):
            r.read_uint(9)

    def test_n_zero_raises(self) -> None:
        r = BitReader(b"\xff")
        with pytest.raises(ValueError):
            r.read_uint(0)

    def test_n_65_raises(self) -> None:
        r = BitReader(b"\xff" * 9)
        with pytest.raises(ValueError):
            r.read_uint(65)

    def test_read_bytes_aligned(self) -> None:
        r = BitReader(b"\xab\xcd")
        assert r.read_bytes(1) == b"\xab"
        assert r.read_bytes(1) == b"\xcd"

    def test_read_bytes_unaligned_raises(self) -> None:
        r = BitReader(b"\xab\xcd")
        r.read_uint(4)
        with pytest.raises(ValueError):
            r.read_bytes(1)

    def test_skip(self) -> None:
        r = BitReader(b"\x0f")  # 0000 1111
        r.skip(4)
        assert r.read_uint(4) == 0xF

    def test_skip_eof(self) -> None:
        r = BitReader(b"\xff")
        with pytest.raises(EOFError):
            r.skip(9)

    def test_seek(self) -> None:
        r = BitReader(b"\x0f")
        r.seek(4)
        assert r.position == 4
        assert r.read_uint(4) == 0xF

    def test_seek_out_of_range(self) -> None:
        r = BitReader(b"\xff")
        with pytest.raises(ValueError):
            r.seek(9)

    def test_peek_uint_does_not_advance(self) -> None:
        r = BitReader(b"\xab")
        v1 = r.peek_uint(4)
        v2 = r.peek_uint(4)
        assert v1 == v2 == 0xA
        assert r.position == 0

    def test_read_bool_true(self) -> None:
        r = BitReader(b"\x80")  # 1000 0000
        assert r.read_bool() is True

    def test_read_bool_false(self) -> None:
        r = BitReader(b"\x7f")  # 0111 1111
        assert r.read_bool() is False

    def test_read_bytes_unaligned_sequential(self) -> None:
        r = BitReader(b"\xab\xcd")
        b0 = r.read_bytes_unaligned(1)
        b1 = r.read_bytes_unaligned(1)
        assert b0 == b"\xab"
        assert b1 == b"\xcd"

    def test_msb_first_order(self) -> None:
        # 0b10110001 = 0xB1; bits should read 1,0,1,1,0,0,0,1
        r = BitReader(b"\xb1")
        bits = [r.read_uint(1) for _ in range(8)]
        assert bits == [1, 0, 1, 1, 0, 0, 0, 1]

    def test_is_exhausted(self) -> None:
        r = BitReader(b"\xff")
        assert not r.is_exhausted()
        r.read_uint(8)
        assert r.is_exhausted()


# ===========================================================================
# gst_utils
# ===========================================================================


class TestGstUtils:
    def test_pack_unpack_round_trip(self) -> None:
        wn, tow = 1234, 500000
        packed = pack_gst(wn, tow)
        assert unpack_gst(packed) == (wn, tow)

    def test_pack_zero(self) -> None:
        assert pack_gst(0, 0) == 0

    def test_pack_max(self) -> None:
        wn_max = 4095
        tow_max = SECONDS_PER_WEEK - 1
        packed = pack_gst(wn_max, tow_max)
        assert unpack_gst(packed) == (wn_max, tow_max)

    def test_pack_wn_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            pack_gst(4096, 0)

    def test_pack_tow_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            pack_gst(0, SECONDS_PER_WEEK)

    def test_gst_to_bytes_length(self) -> None:
        b = gst_to_bytes(pack_gst(1, 30))
        assert len(b) == 4

    def test_gst_to_bytes_big_endian(self) -> None:
        # pack_gst(1, 0) = 0x00100000
        b = gst_to_bytes(pack_gst(1, 0))
        assert b == struct.pack(">I", 1 << 20)

    def test_total_seconds_zero(self) -> None:
        assert gst_to_seconds_total(0, 0) == 0

    def test_total_seconds_one_week(self) -> None:
        assert gst_to_seconds_total(1, 0) == SECONDS_PER_WEEK

    def test_seconds_to_gst_round_trip(self) -> None:
        for total in [0, 100, SECONDS_PER_WEEK, SECONDS_PER_WEEK * 100 + 42]:
            wn, tow = seconds_to_gst(total)
            assert gst_to_seconds_total(wn, tow) == total

    def test_seconds_to_gst_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            seconds_to_gst(-1)

    def test_subframe_aligned_gst_exact(self) -> None:
        # TOW = 60 is exactly 2 subframes in
        wn, tow = subframe_aligned_gst(0, 60)
        assert tow == 60

    def test_subframe_aligned_gst_rounds_down(self) -> None:
        wn, tow = subframe_aligned_gst(0, 45)  # 45 → floor to 30
        assert tow == 30

    def test_subframe_aligned_gst_zero(self) -> None:
        wn, tow = subframe_aligned_gst(0, 0)
        assert tow == 0

    def test_subframe_index_zero(self) -> None:
        assert subframe_index(0, 0) == 0

    def test_subframe_index_one(self) -> None:
        assert subframe_index(0, 30) == 1

    def test_datetime_round_trip(self) -> None:

        dt = gst_to_datetime(100, 12345)
        wn2, tow2 = datetime_to_gst(dt)
        assert (wn2, tow2) == (100, 12345)

    def test_datetime_to_gst_naive_raises(self) -> None:
        from datetime import datetime

        with pytest.raises(ValueError):
            datetime_to_gst(datetime(2024, 1, 1))  # naive

    def test_gst_epoch_correct(self) -> None:
        """GST epoch = 1999-08-22 00:00:00 UTC."""
        from datetime import timezone

        assert GST_EPOCH.year == 1999
        assert GST_EPOCH.month == 8
        assert GST_EPOCH.day == 22
        assert GST_EPOCH.tzinfo == timezone.utc


# ===========================================================================
# _extract_bits helper
# ===========================================================================


class TestExtractBits:
    def test_full_buffer(self) -> None:
        data = b"\xab\xcd"
        result = _extract_bits(data, 0, 16)
        assert result == b"\xab\xcd"

    def test_first_byte(self) -> None:
        result = _extract_bits(b"\xab\xcd", 0, 8)
        assert result == b"\xab"

    def test_second_byte(self) -> None:
        result = _extract_bits(b"\xab\xcd", 8, 8)
        assert result == b"\xcd"

    def test_cross_byte_boundary(self) -> None:
        # 0b 10101011 11001101 = 0xABCD
        # bits 4-11: 0b1011_1100 = 0xBC
        result = _extract_bits(b"\xab\xcd", 4, 8)
        assert result == b"\xbc"

    def test_hkroot_mack_split(self) -> None:
        # Build a 75-byte (600-bit) buffer
        data = bytes(range(75))
        hkroot = _extract_bits(data, 0, HKROOT_BITS)  # 104 bits = 13 bytes
        mack = _extract_bits(data, HKROOT_BITS, MACK_BITS)  # 496 bits = 62 bytes
        assert len(hkroot) == HKROOT_BITS // 8
        assert len(mack) == MACK_BITS // 8

    def test_non_multiple_of_8(self) -> None:
        # Extract 4 bits → 1 byte (padded)
        result = _extract_bits(b"\xf0", 0, 4)
        assert len(result) == 1
        assert result[0] == 0xF0  # 1111 left-aligned


# ===========================================================================
# HkrootSection parser
# ===========================================================================


def _make_hkroot_bytes(
    nma_status: int = 1,
    chain_id: int = 0,
    cif: bool = False,
    cidx: int = 2,
    cpks: int = 3,
    dsm_id: int = 5,
    dsm_block_id: int = 7,
    dsm_data: bytes = b"\x00" * 9,
) -> bytes:
    """Construct a 13-byte HKROOT section by hand."""
    bits: list[int] = []
    # NMA_H (16 bits)
    bits += [(nma_status >> 1) & 1, nma_status & 1]
    bits += [(chain_id >> 1) & 1, chain_id & 1]
    bits += [1 if cif else 0]
    bits += [0] * 11  # reserved
    # TESLA_H (8 bits)
    for i in range(3, -1, -1):
        bits.append((cidx >> i) & 1)
    for i in range(3, -1, -1):
        bits.append((cpks >> i) & 1)
    # DSM header (8 bits)
    for i in range(3, -1, -1):
        bits.append((dsm_id >> i) & 1)
    for i in range(3, -1, -1):
        bits.append((dsm_block_id >> i) & 1)
    # DSM data (72 bits = 9 bytes)
    for byte in dsm_data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    assert len(bits) == HKROOT_BITS
    result = bytearray(HKROOT_BITS // 8)
    for i, bit in enumerate(bits):
        result[i // 8] |= bit << (7 - (i % 8))
    return bytes(result)


class TestHkrootParser:
    def _make_hkroot_bytes(self, **kwargs: object) -> bytes:
        return _make_hkroot_bytes(**kwargs)  # type: ignore[arg-type]

    def test_nma_status_operational(self) -> None:
        data = self._make_hkroot_bytes(nma_status=HP_NMA_OPERATIONAL)
        section = parse_hkroot_section(data)
        assert section.nma_status == HP_NMA_OPERATIONAL

    def test_chain_id(self) -> None:
        data = self._make_hkroot_bytes(chain_id=3)
        section = parse_hkroot_section(data)
        assert section.chain_id == 3

    def test_chain_in_force_true(self) -> None:
        data = self._make_hkroot_bytes(cif=True)
        section = parse_hkroot_section(data)
        assert section.chain_in_force is True

    def test_chain_in_force_false(self) -> None:
        data = self._make_hkroot_bytes(cif=False)
        section = parse_hkroot_section(data)
        assert section.chain_in_force is False

    def test_cidx(self) -> None:
        data = self._make_hkroot_bytes(cidx=7)
        section = parse_hkroot_section(data)
        assert section.cidx == 7

    def test_dsm_id(self) -> None:
        data = self._make_hkroot_bytes(dsm_id=11)
        section = parse_hkroot_section(data)
        assert section.dsm_id == 11

    def test_dsm_block_id(self) -> None:
        data = self._make_hkroot_bytes(dsm_block_id=13)
        section = parse_hkroot_section(data)
        assert section.dsm_block_id == 13

    def test_dsm_data_preserved(self) -> None:
        payload = bytes(range(9))
        data = self._make_hkroot_bytes(dsm_data=payload)
        section = parse_hkroot_section(data)
        assert section.dsm_data == payload

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_hkroot_section(b"\x00" * 12)

    def test_is_kroot_block(self) -> None:
        data = self._make_hkroot_bytes(dsm_id=5)
        section = parse_hkroot_section(data)
        assert section.is_kroot_block is True
        assert section.is_pkr_block is False

    def test_is_pkr_block(self) -> None:
        data = self._make_hkroot_bytes(dsm_id=12)
        section = parse_hkroot_section(data)
        assert section.is_pkr_block is True
        assert section.is_kroot_block is False


# ===========================================================================
# DsmKroot assembler
# ===========================================================================


class TestDsmKroot:
    def _build_complete_dsm(self) -> DsmKroot:
        dsm = DsmKroot(dsm_id=0)
        for i in range(DSM_BLOCKS_PER_MESSAGE):
            dsm.add_block(i, bytes([i] * 9))
        return dsm

    def test_is_complete_after_14_blocks(self) -> None:
        dsm = self._build_complete_dsm()
        assert dsm.is_complete()

    def test_is_incomplete_with_13_blocks(self) -> None:
        dsm = DsmKroot(dsm_id=0)
        for i in range(13):
            dsm.add_block(i, bytes(9))
        assert not dsm.is_complete()

    def test_missing_blocks(self) -> None:
        dsm = DsmKroot(dsm_id=0)
        for i in [0, 1, 3]:
            dsm.add_block(i, bytes(9))
        missing = dsm.missing_blocks()
        assert 2 in missing
        assert 0 not in missing

    def test_assembled_bytes_length(self) -> None:
        dsm = self._build_complete_dsm()
        raw = dsm.assembled_bytes()
        assert len(raw) == DSM_TOTAL_BITS // 8  # 126 bytes

    def test_assembled_bytes_order(self) -> None:
        dsm = DsmKroot(dsm_id=0)
        for i in range(DSM_BLOCKS_PER_MESSAGE):
            dsm.add_block(i, bytes([i]) * 9)
        raw = dsm.assembled_bytes()
        # block 0 is all zeros, block 1 is all 0x01, etc.
        for i in range(DSM_BLOCKS_PER_MESSAGE):
            assert raw[i * 9 : (i + 1) * 9] == bytes([i]) * 9

    def test_assembled_raises_if_incomplete(self) -> None:
        dsm = DsmKroot(dsm_id=0)
        with pytest.raises(RuntimeError):
            dsm.assembled_bytes()

    def test_block_id_out_of_range(self) -> None:
        dsm = DsmKroot(dsm_id=0)
        with pytest.raises(ValueError):
            dsm.add_block(14, bytes(9))

    def test_block_wrong_length(self) -> None:
        dsm = DsmKroot(dsm_id=0)
        with pytest.raises(ValueError):
            dsm.add_block(0, bytes(8))  # should be 9


# ===========================================================================
# DSM-KROOT parser (round-trip via synthesized payload)
# ===========================================================================


class TestParseDsmKroot:
    def _make_dsm_kroot_payload(
        self,
        nb_dk: int = 1,
        pkid: int = 0,
        cidx: int = 2,
        hf: int = 0,
        mf: int = 0,
        ks: int = 2,  # 128-bit key
        ts: int = 4,  # 40-bit tag
        maclt: int = 27,
        wn_k: int = 1200,
        tow_k: int = 345600,
        alpha: bytes = b"\x01\x02\x03\x04\x05\x06",
        kroot: bytes = b"\xde\xad" * 8,  # 16 bytes
        ds: bytes = b"\xca\xfe" * 32,  # 64 bytes
    ) -> bytes:
        """Build a 126-byte (1008-bit) DSM-KROOT payload from field values."""

        bits: list[int] = []

        def push(value: int, n: int) -> None:
            for i in range(n - 1, -1, -1):
                bits.append((value >> i) & 1)

        push(nb_dk, 4)
        push(pkid, 4)
        push(cidx, 4)
        push(hf, 4)
        push(mf, 4)
        push(ks, 4)
        push(ts, 4)
        push(maclt, 8)
        push(0, 12)  # reserved
        push(wn_k, 12)
        push(tow_k, 20)
        for byte in alpha:
            push(byte, 8)
        for byte in kroot:
            push(byte, 8)
        for byte in ds:
            push(byte, 8)

        # Pad to 1008 bits
        while len(bits) < DSM_TOTAL_BITS:
            bits.append(0)

        assert len(bits) == DSM_TOTAL_BITS
        result = bytearray(DSM_TOTAL_BITS // 8)
        for i, bit in enumerate(bits):
            result[i // 8] |= bit << (7 - (i % 8))
        return bytes(result)

    def test_basic_fields(self) -> None:
        payload = self._make_dsm_kroot_payload()
        parsed = parse_dsm_kroot(payload)
        assert parsed.nb_dk == 1
        assert parsed.pkid == 0
        assert parsed.cidx == 2
        assert parsed.ks == 2
        assert parsed.ts == 4
        assert parsed.wn_k == 1200
        assert parsed.tow_k == 345600

    def test_alpha_preserved(self) -> None:
        alpha = b"\xaa\xbb\xcc\xdd\xee\xff"
        payload = self._make_dsm_kroot_payload(alpha=alpha)
        parsed = parse_dsm_kroot(payload)
        assert parsed.alpha == alpha

    def test_kroot_preserved(self) -> None:
        kroot = bytes(range(16))
        payload = self._make_dsm_kroot_payload(kroot=kroot)
        parsed = parse_dsm_kroot(payload)
        assert parsed.kroot == kroot

    def test_key_size_bits_resolved(self) -> None:
        payload = self._make_dsm_kroot_payload(ks=2)  # KS=2 → 128 bits
        parsed = parse_dsm_kroot(payload)
        assert parsed.key_size_bits == 128

    def test_tag_size_bits_resolved(self) -> None:
        payload = self._make_dsm_kroot_payload(ts=4)  # TS=4 → 40 bits
        parsed = parse_dsm_kroot(payload)
        assert parsed.tag_size_bits == 40

    def test_wrong_payload_length_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_dsm_kroot(b"\x00" * 125)

    def test_unknown_ks_raises(self) -> None:
        payload = self._make_dsm_kroot_payload(ks=15)  # invalid
        with pytest.raises(ValueError, match="Unknown KS"):
            parse_dsm_kroot(payload)

    def test_unknown_ts_raises(self) -> None:
        payload = self._make_dsm_kroot_payload(ts=15)  # invalid
        with pytest.raises(ValueError, match="Unknown TS"):
            parse_dsm_kroot(payload)


# ===========================================================================
# MACK parser
# ===========================================================================


def _make_mack_bytes(
    nma_status: int = 1,
    chain_id: int = 0,
    has_key: bool = True,
    tag0: bytes = b"\xab\xcd\xef\x01\x02",
    tag0_adkd: int = ADKD_INAV_CED,
    tag0_cop: int = 0,
    tesla_key: bytes = b"\x00" * 16,
    key_size_bits: int = 128,
    tag_size_bits: int = 40,
) -> bytes:
    """Build 62 bytes of MACK section from field values."""
    bits: list[int] = []

    def push(value: int, n: int) -> None:
        for i in range(n - 1, -1, -1):
            bits.append((value >> i) & 1)

    def push_bytes(data: bytes) -> None:
        for b in data:
            push(b, 8)

    # HF (12 bits)
    push(nma_status, 2)
    push(chain_id, 2)
    push(1 if has_key else 0, 1)
    push(0, 7)  # reserved

    # tag-0
    push_bytes(tag0)
    push(tag0_adkd, 4)
    push(tag0_cop, 4)

    # Cross-auth tags: fill with zeros to reach key start
    key_start_bit = MACK_BITS_M - key_size_bits
    while len(bits) < key_start_bit:
        bits.append(0)

    # TESLA key
    if has_key:
        push_bytes(tesla_key)
    else:
        push(0, key_size_bits)

    # Pad to MACK_BITS
    while len(bits) < MACK_BITS_M:
        bits.append(0)

    assert len(bits) == MACK_BITS_M
    result = bytearray(MACK_BITS_M // 8)
    for i, bit in enumerate(bits):
        result[i // 8] |= bit << (7 - (i % 8))
    return bytes(result)


class TestMackParser:
    def _make_mack_bytes(self, **kwargs: object) -> bytes:
        return _make_mack_bytes(**kwargs)  # type: ignore[arg-type]

    def test_nma_status(self) -> None:
        data = self._make_mack_bytes(nma_status=1)
        parsed = parse_mack_section(data)
        assert parsed.nma_status == 1

    def test_chain_id(self) -> None:
        data = self._make_mack_bytes(chain_id=2)
        parsed = parse_mack_section(data)
        assert parsed.chain_id == 2

    def test_has_key_true(self) -> None:
        key = bytes(range(16))
        data = self._make_mack_bytes(has_key=True, tesla_key=key)
        parsed = parse_mack_section(data)
        assert parsed.has_key is True
        assert parsed.tesla_key == key

    def test_has_key_false_returns_none(self) -> None:
        data = self._make_mack_bytes(has_key=False)
        parsed = parse_mack_section(data)
        assert parsed.has_key is False
        assert parsed.tesla_key is None

    def test_tag0_bytes(self) -> None:
        tag = b"\xde\xad\xbe\xef\x00"
        data = self._make_mack_bytes(tag0=tag)
        parsed = parse_mack_section(data)
        assert parsed.tag0.tag == tag

    def test_tag0_adkd(self) -> None:
        data = self._make_mack_bytes(tag0_adkd=ADKD_INAV_CED)
        parsed = parse_mack_section(data)
        assert parsed.tag0.adkd == ADKD_INAV_CED

    def test_tag0_is_self_auth(self) -> None:
        data = self._make_mack_bytes()
        parsed = parse_mack_section(data)
        assert parsed.tag0.is_self_auth is True

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_mack_section(b"\x00" * 61)

    def test_odd_tag_size_raises(self) -> None:
        with pytest.raises(ValueError, match="partial-byte"):
            parse_mack_section(b"\x00" * 62, tag_size_bits=20)

    def test_tesla_key_preserved(self) -> None:
        key = bytes(range(16))
        data = self._make_mack_bytes(tesla_key=key)
        parsed = parse_mack_section(data, key_size_bits=128, tag_size_bits=40)
        assert parsed.tesla_key == key


# ===========================================================================
# OSNMAPage validation
# ===========================================================================


class TestOSNMAPage:
    def test_valid_page(self) -> None:
        page = OSNMAPage(svid=1, wn=0, tow=0, page_idx=0, osnma_bits=bytes(5))
        assert page.svid == 1

    def test_wrong_osnma_length(self) -> None:
        with pytest.raises(ValueError):
            OSNMAPage(svid=1, wn=0, tow=0, page_idx=0, osnma_bits=bytes(4))

    def test_invalid_svid_low(self) -> None:
        with pytest.raises(ValueError):
            OSNMAPage(svid=0, wn=0, tow=0, page_idx=0, osnma_bits=bytes(5))

    def test_invalid_svid_high(self) -> None:
        with pytest.raises(ValueError):
            OSNMAPage(svid=37, wn=0, tow=0, page_idx=0, osnma_bits=bytes(5))

    def test_invalid_page_idx(self) -> None:
        with pytest.raises(ValueError):
            OSNMAPage(svid=1, wn=0, tow=0, page_idx=15, osnma_bits=bytes(5))


# ===========================================================================
# INavAccumulator — end-to-end with synthesized OSNMA data
# ===========================================================================


class TestINavAccumulator:
    """End-to-end accumulator tests using synthetic OSNMA pages.

    We build 75-byte (600-bit) OSNMA blocks from scratch so that the
    HKROOT and MACK parsers can decode them correctly.
    """

    def _make_600bit_block(
        self, nma_status: int = 1, dsm_id: int = 5, dsm_block_id: int = 0
    ) -> bytes:
        """Build a valid 75-byte OSNMA block (600 bits)."""
        hkroot_bytes = _make_hkroot_bytes(
            nma_status=nma_status,
            dsm_id=dsm_id,
            dsm_block_id=dsm_block_id,
        )  # 13 bytes = 104 bits
        mack_bytes = _make_mack_bytes(
            nma_status=nma_status,
        )  # 62 bytes = 496 bits

        return hkroot_bytes + mack_bytes  # 75 bytes = 600 bits

    def _make_pages(self, svid: int, wn: int, tow_sf: int, block_600: bytes) -> list[OSNMAPage]:
        """Split a 600-bit block into 15 OSNMAPage objects."""
        pages = []
        for page_idx in range(PAGES_PER_SUBFRAME):
            start = page_idx * OSNMA_BYTES_PER_PAGE
            chunk = block_600[start : start + OSNMA_BYTES_PER_PAGE]
            pages.append(
                OSNMAPage(
                    svid=svid,
                    wn=wn,
                    tow=tow_sf + page_idx * 2,
                    page_idx=page_idx,
                    osnma_bits=chunk,
                )
            )
        return pages

    def test_emits_none_until_complete(self) -> None:
        acc = INavAccumulator(svid=1)
        block = self._make_600bit_block()
        pages = self._make_pages(1, 0, 0, block)
        for page in pages[:-1]:
            assert acc.add_page(page) is None

    def test_emits_subframe_on_last_page(self) -> None:
        acc = INavAccumulator(svid=1)
        block = self._make_600bit_block()
        pages = self._make_pages(1, 0, 0, block)
        result = None
        for page in pages:
            result = acc.add_page(page)
        assert isinstance(result, DecodedSubframe)

    def test_decoded_svid(self) -> None:
        acc = INavAccumulator(svid=7)
        block = self._make_600bit_block()
        pages = self._make_pages(7, 0, 0, block)
        result = None
        for page in pages:
            result = acc.add_page(page)
        assert result is not None
        assert result.svid == 7

    def test_decoded_tow_sf(self) -> None:
        acc = INavAccumulator(svid=1)
        block = self._make_600bit_block()
        pages = self._make_pages(1, 0, 30, block)  # subframe starting at TOW=30
        result = None
        for page in pages:
            result = acc.add_page(page)
        assert result is not None
        assert result.tow_sf == 30

    def test_raw_600_length(self) -> None:
        acc = INavAccumulator(svid=1)
        block = self._make_600bit_block()
        pages = self._make_pages(1, 0, 0, block)
        result = None
        for page in pages:
            result = acc.add_page(page)
        assert result is not None
        assert len(result.raw_600) == 75

    def test_wrong_svid_raises(self) -> None:
        acc = INavAccumulator(svid=1)
        page = OSNMAPage(svid=2, wn=0, tow=0, page_idx=0, osnma_bits=bytes(5))
        with pytest.raises(ValueError):
            acc.add_page(page)

    def test_crc_fail_skips_page(self) -> None:
        acc = INavAccumulator(svid=1)
        block = self._make_600bit_block()
        pages = self._make_pages(1, 0, 0, block)
        # Make all pages fail CRC — no subframe should complete
        bad_pages = [
            OSNMAPage(
                svid=p.svid,
                wn=p.wn,
                tow=p.tow,
                page_idx=p.page_idx,
                osnma_bits=p.osnma_bits,
                crc_ok=False,
            )
            for p in pages
        ]
        for page in bad_pages:
            assert acc.add_page(page) is None

    def test_new_subframe_resets_buffer(self) -> None:
        acc = INavAccumulator(svid=1)
        block = self._make_600bit_block()
        # Feed 5 pages of subframe 0, then switch to subframe 1
        pages_sf0 = self._make_pages(1, 0, 0, block)[:5]
        pages_sf1 = self._make_pages(1, 0, 30, block)
        for page in pages_sf0:
            acc.add_page(page)
        result = None
        for page in pages_sf1:
            result = acc.add_page(page)
        # subframe 1 should complete normally
        assert isinstance(result, DecodedSubframe)
        assert result.tow_sf == 30

    def test_invalid_svid_constructor(self) -> None:
        with pytest.raises(ValueError):
            INavAccumulator(svid=0)

    def test_dsm_builder_accumulates(self) -> None:
        """DSM blocks from consecutive subframes accumulate in the builder."""
        acc = INavAccumulator(svid=1)
        for block_id in range(DSM_BLOCKS_PER_MESSAGE):
            block = self._make_600bit_block(dsm_id=3, dsm_block_id=block_id)
            pages = self._make_pages(1, 0, block_id * SUBFRAME_DURATION_S, block)
            for page in pages:
                acc.add_page(page)
        # All 14 blocks delivered → DSM should be complete
        completed = acc.completed_dsm()
        assert 3 in completed
        assert completed[3].is_complete()
