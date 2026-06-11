"""Galileo System Time (GST) conversion utilities.

GST encoding per Galileo OS SIS ICD §5.1.3:
    GST = WN[12 bits, big-endian MSB] || TOW[20 bits]  packed into 4 bytes BE

GST epoch: 1999-08-22 00:00:00 UTC (= GPS week 1024 Sunday)

The 4-byte packed integer is used directly in the TESLA key derivation:
    K_i = SHA-256( K_{i+1} || pack_gst(wn, tow)[4B,BE] || alpha[6B] )

Note: the osnma_inav.py simulation uses ``gst_sf & 0xFFFFFFFF`` as the GST
integer.  That representation is correct when WN=0 (simulation starting at
epoch 0), because TOW = total_seconds in that case.  For real Galileo data,
always use ``pack_gst(wn, tow)`` to ensure the WN is included.
"""

from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

#: GST epoch in UTC (Galileo System Time zero point).
GST_EPOCH: datetime = datetime(1999, 8, 22, 0, 0, 0, tzinfo=timezone.utc)

#: Seconds per GST week.
SECONDS_PER_WEEK: int = 7 * 24 * 3600  # 604800

#: Bit widths
WN_BITS: int = 12
TOW_BITS: int = 20

#: Maximum TOW value (exclusive)
TOW_MAX: int = SECONDS_PER_WEEK  # TOW ∈ [0, 604799]

#: Subframe duration in seconds (15 pages × 2 s/page).
SUBFRAME_DURATION_S: int = 30


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def pack_gst(wn: int, tow: int) -> int:
    """Pack Galileo week number and time of week into a 32-bit GST integer.

    Bit layout:  WN[31..20] (12 bits) || TOW[19..0] (20 bits)

    Args:
        wn:  Galileo week number (0..4095).
        tow: Time of week in seconds (0..604799).

    Returns:
        32-bit unsigned integer representing GST.

    Raises:
        ValueError: if wn or tow are out of range.
    """
    if not 0 <= wn < (1 << WN_BITS):
        raise ValueError(f"WN {wn} out of range [0, {1 << WN_BITS})")
    if not 0 <= tow < TOW_MAX:
        raise ValueError(f"TOW {tow} out of range [0, {TOW_MAX})")
    return ((wn & 0xFFF) << TOW_BITS) | (tow & 0xFFFFF)


def unpack_gst(gst_int: int) -> tuple[int, int]:
    """Unpack a 32-bit GST integer into ``(WN, TOW)``.

    Args:
        gst_int: 32-bit GST integer from ``pack_gst``.

    Returns:
        Tuple ``(wn, tow)``.
    """
    wn = (gst_int >> TOW_BITS) & 0xFFF
    tow = gst_int & 0xFFFFF
    return wn, tow


def gst_to_bytes(gst_int: int) -> bytes:
    """Serialize a GST integer to 4 bytes, big-endian.

    Used directly in TESLA key derivation:
        SHA-256( K_{i+1} || gst_to_bytes(pack_gst(wn, tow)) || alpha )
    """
    return struct.pack(">I", gst_int & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------


def gst_to_seconds_total(wn: int, tow: int) -> int:
    """Total GST seconds elapsed since the GST epoch.

    Args:
        wn:  Galileo week number.
        tow: Time of week [s].
    """
    return wn * SECONDS_PER_WEEK + tow


def seconds_to_gst(total_seconds: int) -> tuple[int, int]:
    """Convert total GST seconds to ``(WN, TOW)``.

    Args:
        total_seconds: Seconds elapsed since GST epoch (≥ 0).

    Returns:
        Tuple ``(wn, tow)``.
    """
    if total_seconds < 0:
        raise ValueError(f"total_seconds must be ≥ 0, got {total_seconds}")
    wn = total_seconds // SECONDS_PER_WEEK
    tow = total_seconds % SECONDS_PER_WEEK
    return wn, tow


def subframe_aligned_gst(wn: int, tow: int) -> tuple[int, int]:
    """Round (WN, TOW) down to the nearest 30-second subframe boundary.

    Returns:
        ``(wn_sf, tow_sf)`` of the subframe start.
    """
    total_s = gst_to_seconds_total(wn, tow)
    sf_start = (total_s // SUBFRAME_DURATION_S) * SUBFRAME_DURATION_S
    return seconds_to_gst(sf_start)


def subframe_index(wn: int, tow: int) -> int:
    """Compute the 0-based subframe counter from the GST epoch.

    Subframes are numbered sequentially: subframe 0 starts at GST epoch.
    """
    return gst_to_seconds_total(wn, tow) // SUBFRAME_DURATION_S


# ---------------------------------------------------------------------------
# datetime conversion
# ---------------------------------------------------------------------------


def gst_to_datetime(wn: int, tow: int) -> datetime:
    """Convert GST ``(WN, TOW)`` to a timezone-aware UTC ``datetime``.

    Args:
        wn:  Galileo week number.
        tow: Time of week [s].

    Returns:
        UTC ``datetime`` with ``tzinfo=timezone.utc``.
    """
    total_s = gst_to_seconds_total(wn, tow)
    return GST_EPOCH + timedelta(seconds=total_s)


def datetime_to_gst(dt: datetime) -> tuple[int, int]:
    """Convert a UTC ``datetime`` to GST ``(WN, TOW)``.

    Args:
        dt: Timezone-aware datetime (UTC recommended).

    Returns:
        Tuple ``(wn, tow)``.

    Raises:
        ValueError: if ``dt`` is timezone-naive.
    """
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (got naive)")
    delta = dt.astimezone(timezone.utc) - GST_EPOCH
    total_s = int(delta.total_seconds())
    if total_s < 0:
        raise ValueError(f"datetime {dt} is before GST epoch {GST_EPOCH}")
    return seconds_to_gst(total_s)
