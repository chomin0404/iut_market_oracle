"""GNSS OSNMA bit-exact parser sub-package.

Modules
-------
_bit_io      — BitReader / BitWriter primitives
hkroot_parser — HKROOT section + DSM-KROOT/PKR decoding (ICD Table 8/9)
mack_parser   — MACK section decoding (ICD Table 14)
inav_parser   — Galileo I/NAV double-page accumulator → SubframeData
"""

from gnss.parser.hkroot_parser import (
    DSM_BLOCKS_PER_MESSAGE,
    HKROOT_BITS,
    DsmKroot,
    DsmPkr,
    HkrootSection,
    ParsedHkroot,
    ParsedPkr,
    parse_dsm_kroot,
    parse_dsm_pkr,
    parse_hkroot_section,
    parsed_pkr_to_message,
)
from gnss.parser.inav_parser import (
    MACK_BITS,
    OSNMA_BITS_PER_PAGE,
    PAGES_PER_SUBFRAME,
    INavAccumulator,
    OSNMAPage,
)
from gnss.parser.mack_parser import (
    ParsedMack,
    TagEntry,
    parse_mack_section,
)

__all__ = [
    # hkroot
    "DSM_BLOCKS_PER_MESSAGE",
    "HKROOT_BITS",
    "DsmKroot",
    "DsmPkr",
    "HkrootSection",
    "ParsedHkroot",
    "ParsedPkr",
    "parse_dsm_kroot",
    "parse_dsm_pkr",
    "parse_hkroot_section",
    "parsed_pkr_to_message",
    # inav
    "MACK_BITS",
    "OSNMA_BITS_PER_PAGE",
    "PAGES_PER_SUBFRAME",
    "INavAccumulator",
    "OSNMAPage",
    # mack
    "ParsedMack",
    "TagEntry",
    "parse_mack_section",
]
