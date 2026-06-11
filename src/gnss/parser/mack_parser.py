"""Galileo OSNMA MACK section parser.

Decodes the 496-bit MACK section transmitted in each I/NAV subframe.

ICD references
--------------
  Galileo OSNMA SIS ICD OS-SIS-ICD-OSNMA §5.4 (MACK structure)
  Table 14:  MACK section field layout

MACK section layout (496 bits per subframe, for KS=128b and TS=40b):
    [0:12]     HF          (12 b)  — header flags
    [12:52]    tag-0       (40 b)  — self-authentication tag (ADKD_INAV_CED)
    [52:56]    ADKD_0      (4 b)   — ADKD for tag-0
    [56:60]    COP_0       (4 b)   — Continuity-of-protection for tag-0
    for each subsequent cross-auth tag (ADKD≠0, PRN_A-addressed):
        [+0:+40]  tag_k    (40 b)  — cross-authentication tag
        [+40:+48] PRN_A    (8 b)   — authenticated satellite PRN (1-36)
        [+48:+52] ADKD_k   (4 b)   — ADKD type
        [+52:+56] COP_k    (4 b)   — Continuity-of-protection
    [496-128:496] TESLA_KEY (128 b) — disclosed K_{sf_idx − TESLA_DELAY}

The number of cross-authentication tags depends on the key size and tag size.
For KS=128b, TS=40b:
    header + tag-0+info = 12 + 56 = 68 bits
    TESLA key           = 128 bits
    remaining for cross tags = 496 − 68 − 128 = 300 bits
    each cross tag = 40 + 8 + 4 + 4 = 56 bits → 5 cross-auth tags

Note: for the first ``TESLA_DELAY`` subframes the disclosed key is absent.
The parser returns ``tesla_key=None`` when ``has_key=False`` in HF.
"""

from __future__ import annotations

from dataclasses import dataclass

from gnss.parser._bit_io import BitReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Total MACK section size in bits.
MACK_BITS: int = 496

#: MACK header flags field size in bits.
MACK_HF_BITS: int = 12

# Subfields within HF (12 bits total):
#   NMA_S  [0:2]  — NMA status (matches HKROOT)
#   CHAIN_ID[2:4] — chain identifier
#   has_key [4]   — 1 if TESLA key is present in this block
#   reserved[5:12]— reserved (7 bits)
_HF_NMA_S_BITS: int = 2
_HF_CHAIN_ID_BITS: int = 2
_HF_HAS_KEY_BITS: int = 1
_HF_RESERVED_BITS: int = 7

#: Size of tag-0 adkd+cop info field in bits (ICD §5.4.1).
_TAG0_INFO_BITS: int = 8  # ADKD_0[4b] + COP_0[4b]

#: Size of cross-auth tag info field in bits (ICD §5.4.2).
_CROSS_TAG_PRN_BITS: int = 8  # PRN_A (8 bits)
_CROSS_TAG_INFO_BITS: int = 8  # ADKD_k[4b] + COP_k[4b]

# ADKD codes (per ICD §5.4.1)
ADKD_INAV_CED: int = 0   # I/NAV clock & ephemeris data
ADKD_INAV_TIMING: int = 4  # I/NAV timing parameters
ADKD_SLOW_MAC: int = 12   # slow MAC (cross-constellation)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TagEntry:
    """One authentication tag entry in the MACK section.

    Attributes:
        tag:   Tag bytes (tag_size_bits // 8, or partial bytes if not multiple of 8).
        adkd:  ADKD type (0, 4, or 12).
        cop:   Continuity-of-protection nibble (0–15).
        prn_a: Authenticated satellite PRN (1-36) for cross-auth tags;
               0 for tag-0 (self-authentication).
    """

    tag: bytes
    adkd: int
    cop: int
    prn_a: int = 0  # 0 = tag-0 (self-auth)

    @property
    def is_self_auth(self) -> bool:
        """True if this is the tag-0 self-authentication tag (prn_a == 0)."""
        return self.prn_a == 0


@dataclass(frozen=True)
class ParsedMack:
    """Decoded MACK section for one subframe.

    Produced by :func:`parse_mack_section`.

    Attributes:
        nma_status:   NMA status from HF (should match HKROOT).
        chain_id:     Chain identifier from HF.
        has_key:      True if TESLA key is disclosed in this subframe.
        tag0:         Self-authentication tag-0.
        cross_tags:   Cross-authentication tags (may be empty).
        tesla_key:    Disclosed TESLA key bytes, or ``None`` if not present.
    """

    nma_status: int
    chain_id: int
    has_key: bool
    tag0: TagEntry
    cross_tags: tuple[TagEntry, ...]
    tesla_key: bytes | None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_mack_section(
    data: bytes,
    key_size_bits: int = 128,
    tag_size_bits: int = 40,
) -> ParsedMack:
    """Parse a 496-bit MACK section.

    The MACK section is the second part of the 600-bit per-subframe OSNMA
    block, starting at bit 104 (after the 104-bit HKROOT section).

    Args:
        data:          Exactly 62 bytes (496 bits) of MACK section data.
        key_size_bits: TESLA key size in bits, from the DSM-KROOT ``KS`` field
                       (default 128, matching ``osnma_inav.KEY_SIZE_BITS``).
        tag_size_bits: Authentication tag size in bits, from DSM-KROOT ``TS``
                       (default 40, matching ``osnma_inav.MAC_TAG_BITS``).

    Returns:
        Decoded :class:`ParsedMack`.

    Raises:
        ValueError: if ``data`` is not exactly 62 bytes or tag_size_bits is not
                    a multiple of 8 (partial-byte tags are not yet supported).
    """
    expected_bytes = MACK_BITS // 8
    if len(data) != expected_bytes:
        raise ValueError(
            f"Expected {expected_bytes} bytes for MACK, got {len(data)}"
        )
    if tag_size_bits % 8 != 0:
        raise ValueError(
            f"tag_size_bits={tag_size_bits}: partial-byte tags are not supported"
        )
    tag_bytes = tag_size_bits // 8
    key_bytes = key_size_bits // 8

    r = BitReader(data)

    # --- Header flags (12 bits) ---
    nma_status = r.read_uint(_HF_NMA_S_BITS)
    chain_id = r.read_uint(_HF_CHAIN_ID_BITS)
    has_key = r.read_bool()
    r.skip(_HF_RESERVED_BITS)

    # --- Tag-0 (self-authentication) ---
    tag0_bytes = r.read_bytes_unaligned(tag_bytes)
    tag0_adkd = r.read_uint(4)
    tag0_cop = r.read_uint(4)
    tag0 = TagEntry(tag=tag0_bytes, adkd=tag0_adkd, cop=tag0_cop, prn_a=0)

    # --- TESLA key at the end ---
    # Key occupies the last key_bytes bytes of the 62-byte MACK section.
    # Compute bits available for cross-auth tags between tag-0 info and key.
    bits_consumed_so_far = MACK_HF_BITS + tag_size_bits + _TAG0_INFO_BITS
    key_start_bit = MACK_BITS - key_size_bits
    cross_tag_total_bits = key_start_bit - bits_consumed_so_far

    # Each cross-auth tag: tag[ts_b] + prn_a[8b] + adkd[4b] + cop[4b]
    cross_tag_entry_bits = tag_size_bits + _CROSS_TAG_PRN_BITS + _CROSS_TAG_INFO_BITS
    n_cross = cross_tag_total_bits // cross_tag_entry_bits if cross_tag_entry_bits > 0 else 0

    cross_tags: list[TagEntry] = []
    for _ in range(n_cross):
        tag_k = r.read_bytes_unaligned(tag_bytes)
        prn_a = r.read_uint(_CROSS_TAG_PRN_BITS)
        adkd_k = r.read_uint(4)
        cop_k = r.read_uint(4)
        cross_tags.append(TagEntry(tag=tag_k, adkd=adkd_k, cop=cop_k, prn_a=prn_a))

    # --- Seek to key position and read TESLA key ---
    r.seek(key_start_bit)
    tesla_key: bytes | None = None
    if has_key:
        tesla_key = r.read_bytes_unaligned(key_bytes)

    return ParsedMack(
        nma_status=nma_status,
        chain_id=chain_id,
        has_key=has_key,
        tag0=tag0,
        cross_tags=tuple(cross_tags),
        tesla_key=tesla_key,
    )
