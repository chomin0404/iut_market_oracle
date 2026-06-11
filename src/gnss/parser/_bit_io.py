"""Bit-level I/O primitives for OSNMA binary parsing.

The Galileo OSNMA ICD specifies bit fields aligned to arbitrary bit
boundaries, not to byte boundaries.  This module provides a minimal
``BitReader`` that reads MSB-first from a ``bytes`` buffer.

Example::

    reader = BitReader(b"\\xAB\\xCD")
    assert reader.read_uint(4) == 0xA  # first nibble
    assert reader.read_uint(4) == 0xB  # second nibble
    assert reader.read_bytes(1) == b"\\xCD"
"""

from __future__ import annotations


class BitReader:
    """MSB-first bit reader over an immutable ``bytes`` buffer.

    Bits are numbered from 0 (MSB of byte 0) to ``8*len(data)-1``
    (LSB of the last byte), consistent with the Galileo ICD convention.

    Args:
        data: Source bytes.  Must not be empty.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos: int = 0  # current bit position (0 = MSB of byte 0)
        self._total_bits: int = len(data) * 8

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> int:
        """Current bit position (0-based, MSB-first)."""
        return self._pos

    @property
    def remaining(self) -> int:
        """Remaining bits available for reading."""
        return self._total_bits - self._pos

    def is_exhausted(self) -> bool:
        """Return True if all bits have been consumed."""
        return self._pos >= self._total_bits

    # ------------------------------------------------------------------
    # Core read operations
    # ------------------------------------------------------------------

    def read_uint(self, n: int) -> int:
        """Read ``n`` bits MSB-first and return as an unsigned integer.

        Args:
            n: Number of bits to read (1 ≤ n ≤ 64).

        Returns:
            Unsigned integer value of the next ``n`` bits.

        Raises:
            ValueError: if ``n`` is out of range.
            EOFError:   if fewer than ``n`` bits remain.
        """
        if n < 1 or n > 64:
            raise ValueError(f"n must be in [1, 64], got {n}")
        if self._pos + n > self._total_bits:
            raise EOFError(
                f"Cannot read {n} bits: only {self.remaining} remain "
                f"(position={self._pos}, total={self._total_bits})"
            )
        result: int = 0
        for _ in range(n):
            byte_idx = self._pos >> 3  # self._pos // 8
            bit_idx = 7 - (self._pos & 7)  # MSB first within each byte
            result = (result << 1) | ((self._data[byte_idx] >> bit_idx) & 1)
            self._pos += 1
        return result

    def read_bytes(self, n: int) -> bytes:
        """Read ``n`` complete bytes.

        The reader must be byte-aligned at the time of this call.

        Args:
            n: Number of bytes to read.

        Returns:
            ``bytes`` of length ``n``.

        Raises:
            ValueError: if the reader is not byte-aligned.
            EOFError:   if fewer than ``n * 8`` bits remain.
        """
        if self._pos & 7:
            raise ValueError(
                f"read_bytes requires byte alignment (position={self._pos})"
            )
        if self._pos + n * 8 > self._total_bits:
            raise EOFError(
                f"Cannot read {n} bytes: only {self.remaining} bits remain"
            )
        start = self._pos >> 3
        self._pos += n * 8
        return self._data[start : start + n]

    def skip(self, n: int) -> None:
        """Advance the position by ``n`` bits without returning data.

        Args:
            n: Number of bits to skip.

        Raises:
            EOFError: if fewer than ``n`` bits remain.
        """
        if self._pos + n > self._total_bits:
            raise EOFError(
                f"Cannot skip {n} bits: only {self.remaining} remain"
            )
        self._pos += n

    def seek(self, pos: int) -> None:
        """Seek to an absolute bit position.

        Args:
            pos: Target bit position (0-based).

        Raises:
            ValueError: if ``pos`` is out of ``[0, total_bits]``.
        """
        if pos < 0 or pos > self._total_bits:
            raise ValueError(
                f"Seek position {pos} out of range [0, {self._total_bits}]"
            )
        self._pos = pos

    def peek_uint(self, n: int) -> int:
        """Peek at the next ``n`` bits without advancing the position.

        Args:
            n: Number of bits to peek (1 ≤ n ≤ 64).

        Returns:
            Unsigned integer value of the next ``n`` bits.
        """
        saved = self._pos
        value = self.read_uint(n)
        self._pos = saved
        return value

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def read_bool(self) -> bool:
        """Read one bit as a boolean."""
        return self.read_uint(1) == 1

    def read_bytes_unaligned(self, n_bytes: int) -> bytes:
        """Read ``n_bytes`` bytes from any bit position (no alignment required).

        Collects bits in groups of 8, padding the last byte with zero bits
        on the right if the total available bits are not a multiple of 8.

        Args:
            n_bytes: Number of bytes to read.
        """
        result = bytearray(n_bytes)
        for i in range(n_bytes):
            result[i] = self.read_uint(8)
        return bytes(result)
