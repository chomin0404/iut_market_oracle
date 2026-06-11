"""Galileo I/NAV double-page OSNMA accumulator.

Each Galileo I/NAV page (2 seconds) carries a 40-bit OSNMA field embedded
in the reserved bits of word types 0 through 14.  This module accumulates
those 40-bit payloads across 15 consecutive pages to form one complete
600-bit OSNMA block per subframe, then splits it into the 104-bit HKROOT
section and 496-bit MACK section.

ICD references
--------------
  Galileo OS SIS ICD v2.1 §4.3 — I/NAV page structure
  Galileo OSNMA SIS ICD §5.2   — OSNMA placement within I/NAV

I/NAV double-page OSNMA field placement:
  Even page: 240 bits total
    bits  0:   page type (0 = even)
    bits  1:6:  word type (WT, 6 bits)
    bits  7:48: data fields (vary by WT)
    ...
    bits 208:248: Reserved_1 (40 bits) ← OSNMA field

  The OSNMA bit offset within the even-page body varies by word type.
  For operational use, receivers extract exactly 40 bits per page period.

  In this research implementation the parser accepts pre-extracted
  ``OSNMAPage`` objects (SVID + GST + 5-byte OSNMA payload) rather than
  raw I/NAV bitstreams, matching the format produced by OSNMAlib and the
  Septentrio SBF OSNMA block.

Subframe assembly:
  15 consecutive pages (page_idx 0-14) contribute one 40-bit block each.
  Concatenating blocks 0-14 in order gives 600 bits:
    bits   0-103:  HKROOT section (parsed by hkroot_parser)
    bits 104-599:  MACK section   (parsed by mack_parser)
"""

from __future__ import annotations

from dataclasses import dataclass

from gnss.parser._bit_io import BitReader
from gnss.parser.hkroot_parser import (
    HKROOT_BITS,
    DsmKroot,
    HkrootSection,
    parse_hkroot_section,
)
from gnss.parser.mack_parser import MACK_BITS, ParsedMack, parse_mack_section
from gnss.utils.gst_utils import (
    pack_gst,
    subframe_aligned_gst,
    subframe_index,
)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

#: OSNMA bits contributed by each I/NAV page (40 bits = 5 bytes).
OSNMA_BITS_PER_PAGE: int = 40

#: Pages per subframe (15 pages × 2 s = 30 s).
PAGES_PER_SUBFRAME: int = 15

#: Total OSNMA bits per subframe.
OSNMA_BITS_PER_SUBFRAME: int = OSNMA_BITS_PER_PAGE * PAGES_PER_SUBFRAME  # 600

assert HKROOT_BITS + MACK_BITS == OSNMA_BITS_PER_SUBFRAME, (
    f"HKROOT({HKROOT_BITS}) + MACK({MACK_BITS}) != {OSNMA_BITS_PER_SUBFRAME}"
)

#: OSNMA bytes per page.
OSNMA_BYTES_PER_PAGE: int = OSNMA_BITS_PER_PAGE // 8  # 5

#: Re-export for convenience
MACK_BITS = MACK_BITS


# ---------------------------------------------------------------------------
# Data class for one I/NAV OSNMA page
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OSNMAPage:
    """One I/NAV page's OSNMA data (40-bit field), keyed by SVID and GST.

    In a real receiver, this is produced by extracting the ``Reserved_1``
    field from each I/NAV even-page burst.  In research/test contexts it
    can be synthesised directly from the ``INavOSNMASimulator``.

    Attributes:
        svid:       Galileo SVID (1-36).
        wn:         Galileo week number.
        tow:        Time of week at the start of this page [s].
        page_idx:   0-based page index within the subframe (0-14).
        osnma_bits: 5 bytes (40 bits) of OSNMA data for this page.
        crc_ok:     True if the I/NAV CRC passed (False → ignore page).
    """

    svid: int
    wn: int
    tow: int
    page_idx: int
    osnma_bits: bytes  # 5 bytes = 40 bits
    crc_ok: bool = True

    def __post_init__(self) -> None:
        if len(self.osnma_bits) != OSNMA_BYTES_PER_PAGE:
            raise ValueError(
                f"osnma_bits must be {OSNMA_BYTES_PER_PAGE} bytes, got {len(self.osnma_bits)}"
            )
        if not 1 <= self.svid <= 36:
            raise ValueError(f"SVID {self.svid} out of range [1, 36]")
        if not 0 <= self.page_idx < PAGES_PER_SUBFRAME:
            raise ValueError(f"page_idx {self.page_idx} out of range [0, {PAGES_PER_SUBFRAME})")


# ---------------------------------------------------------------------------
# Decoded subframe output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedSubframe:
    """OSNMA data for one (SVID, subframe) pair, fully decoded.

    Produced by :class:`INavAccumulator` once all 15 pages have arrived.

    Attributes:
        svid:           Galileo SVID (1-36).
        wn:             Galileo week number.
        tow_sf:         TOW at the start of this subframe [s] (multiple of 30).
        subframe_idx:   0-based subframe index from GST epoch.
        hkroot_section: Decoded HKROOT (NMA status, DSM header, DSM data block).
        mack:           Decoded MACK (tag-0, cross-tags, TESLA key).
        raw_600:        Raw 75 bytes (600 bits) of assembled OSNMA data.
    """

    svid: int
    wn: int
    tow_sf: int
    subframe_idx: int
    hkroot_section: HkrootSection
    mack: ParsedMack
    raw_600: bytes  # 75 bytes


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


class INavAccumulator:
    """Accumulates I/NAV OSNMA pages and emits complete subframes.

    One accumulator instance tracks a single SVID.  Feed pages via
    :meth:`add_page`; the method returns a :class:`DecodedSubframe` when all
    15 pages for a subframe have arrived, otherwise ``None``.

    Also maintains a :class:`DsmKroot` assembler so that KROOT blocks are
    aggregated across subframes and returned once complete (one complete DSM
    takes at most 14 subframes = 7 minutes).

    Args:
        svid:          Galileo SVID to track (1-36).
        key_size_bits: TESLA key size from DSM-KROOT (default 128).
        tag_size_bits: MAC tag size from DSM-KROOT (default 40).
    """

    def __init__(
        self,
        svid: int,
        key_size_bits: int = 128,
        tag_size_bits: int = 40,
    ) -> None:
        if not 1 <= svid <= 36:
            raise ValueError(f"SVID {svid} out of range [1, 36]")
        self._svid = svid
        self._key_size_bits = key_size_bits
        self._tag_size_bits = tag_size_bits

        # Current subframe accumulation buffer: page_idx → 5 bytes
        self._current_sf_gst: int | None = None  # packed GST of current subframe start
        self._page_buf: dict[int, bytes] = {}  # page_idx → 5 bytes

        # DSM-KROOT assembler: dsm_id → DsmKroot
        self._dsm_builders: dict[int, DsmKroot] = {}
        self._completed_dsm: dict[int, DsmKroot] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def svid(self) -> int:
        return self._svid

    def add_page(self, page: OSNMAPage) -> DecodedSubframe | None:
        """Feed one I/NAV OSNMA page to the accumulator.

        Args:
            page: An :class:`OSNMAPage` for this accumulator's SVID.

        Returns:
            A :class:`DecodedSubframe` if this page completes the subframe,
            otherwise ``None``.

        Raises:
            ValueError: if page.svid does not match this accumulator.
        """
        if page.svid != self._svid:
            raise ValueError(f"Page SVID {page.svid} does not match accumulator SVID {self._svid}")
        if not page.crc_ok:
            return None  # discard pages with CRC errors

        # Determine the subframe-aligned GST for this page
        wn_sf, tow_sf = subframe_aligned_gst(page.wn, page.tow)
        sf_gst = pack_gst(wn_sf, tow_sf)

        # Reset buffer if we moved to a new subframe
        if sf_gst != self._current_sf_gst:
            self._current_sf_gst = sf_gst
            self._page_buf = {}

        self._page_buf[page.page_idx] = page.osnma_bits

        # Emit a decoded subframe when all 15 pages have arrived
        if len(self._page_buf) == PAGES_PER_SUBFRAME:
            return self._assemble_subframe(wn_sf, tow_sf)
        return None

    def completed_dsm(self) -> dict[int, DsmKroot]:
        """Return a snapshot of all fully assembled DSM-KROOT builders."""
        return dict(self._completed_dsm)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assemble_subframe(self, wn_sf: int, tow_sf: int) -> DecodedSubframe:
        """Assemble the 600-bit OSNMA block and decode HKROOT + MACK."""
        # Concatenate pages 0-14 in order
        raw_600 = b"".join(self._page_buf[i] for i in range(PAGES_PER_SUBFRAME))

        # Split: HKROOT = bits 0-103 (13 bytes), MACK = bits 104-599 (62 bytes)
        hkroot_bytes = _extract_bits(raw_600, 0, HKROOT_BITS)
        mack_bytes = _extract_bits(raw_600, HKROOT_BITS, MACK_BITS)

        hkroot_section = parse_hkroot_section(hkroot_bytes)
        mack = parse_mack_section(
            mack_bytes,
            key_size_bits=self._key_size_bits,
            tag_size_bits=self._tag_size_bits,
        )

        # Update DSM builder
        if hkroot_section.is_kroot_block:
            self._update_dsm_builder(hkroot_section)

        sf_idx = subframe_index(wn_sf, tow_sf)

        return DecodedSubframe(
            svid=self._svid,
            wn=wn_sf,
            tow_sf=tow_sf,
            subframe_idx=sf_idx,
            hkroot_section=hkroot_section,
            mack=mack,
            raw_600=raw_600,
        )

    def _update_dsm_builder(self, hkroot: HkrootSection) -> None:
        """Store one DSM-KROOT block; move to completed if all 14 arrived."""
        dsm_id = hkroot.dsm_id
        if dsm_id not in self._dsm_builders:
            self._dsm_builders[dsm_id] = DsmKroot(dsm_id=dsm_id)
        builder = self._dsm_builders[dsm_id]
        builder.add_block(hkroot.dsm_block_id, hkroot.dsm_data)
        if builder.is_complete():
            self._completed_dsm[dsm_id] = builder
            # Keep the builder in place for re-verification


# ---------------------------------------------------------------------------
# Bit-slicing helper
# ---------------------------------------------------------------------------


def _extract_bits(data: bytes, start_bit: int, n_bits: int) -> bytes:
    """Extract a contiguous ``n_bits`` slice starting at ``start_bit``.

    The result is packed MSB-first into ``ceil(n_bits / 8)`` bytes.
    If ``n_bits`` is not a multiple of 8, the last byte is zero-padded
    on the LSB side.

    Args:
        data:       Source byte buffer.
        start_bit:  0-based starting bit index (MSB=0 convention).
        n_bits:     Number of bits to extract.

    Returns:
        Byte buffer of length ``ceil(n_bits / 8)``.

    Raises:
        EOFError: if the slice extends beyond the source data.
    """
    r = BitReader(data)
    r.seek(start_bit)

    n_full_bytes = n_bits // 8
    remainder = n_bits % 8
    result = bytearray(n_full_bytes + (1 if remainder else 0))

    for i in range(n_full_bytes):
        result[i] = r.read_uint(8)

    if remainder:
        partial = r.read_uint(remainder)
        result[n_full_bytes] = partial << (8 - remainder)  # left-align

    return bytes(result)
