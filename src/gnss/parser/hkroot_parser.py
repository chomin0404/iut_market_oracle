"""Galileo OSNMA HKROOT section parser.

Parses the 104-bit HKROOT section transmitted in each I/NAV subframe and
assembles complete DSM-KROOT messages from multiple subframe blocks.

ICD references
--------------
  Galileo OSNMA SIS ICD OS-SIS-ICD-OSNMA §5.3 (HKROOT structure)
  Table 8:   HKROOT section bit layout (NMA_H + TESLA_H + DSM header + data)
  Table 9:   DSM-KROOT message field layout

HKROOT section layout (104 bits per subframe):
    [0:2]    NMA_STATUS  (2 b)  — 0=TEST, 1=OPERATIONAL, 2=DONT_USE, 3=RESERVED
    [2:4]    CHAIN_ID    (2 b)  — identifies which TESLA chain is in force
    [4]      CIF         (1 b)  — Chain In Force (1 = new chain starts this SF)
    [5:16]   NMA_H_RSVD (11 b)  — reserved (ignore)
    [16:20]  CIDX        (4 b)  — Chain Index (same as chain_id but scoped to tesla)
    [20:24]  CPKS        (4 b)  — Chain Phase and Key Size encoding
    [24:28]  DSM_ID      (4 b)  — DSM block type (0-11 = DSM-KROOT, 12 = DSM-PKR)
    [28:32]  DSM_BLOCK_ID(4 b)  — sequential 0-based block number within DSM (0-13)
    [32:104] DSM_DATA   (72 b)  — one 72-bit fragment of the DSM message

A complete DSM is assembled from 14 blocks (BLOCK_ID 0-13) of the same DSM_ID.
Total DSM payload = 14 × 72 = 1008 bits.

DSM-KROOT field layout (assembled, variable total length):
    [0:4]    NB_DK       (4 b)  — number of distinguished keys
    [4:8]    PKID        (4 b)  — public key ID
    [8:12]   CIDX        (4 b)  — chain index
    [12:16]  HF          (4 b)  — hash function (0=SHA-256)
    [16:20]  MF          (4 b)  — MAC function  (0=HMAC-SHA256)
    [20:24]  KS          (4 b)  — key size selector (see _KEY_SIZE_BITS)
    [24:28]  TS          (4 b)  — tag size selector (see _TAG_SIZE_BITS)
    [28:36]  MACLT       (8 b)  — MAC lookup table ID
    [36:48]  reserved   (12 b)
    [48:60]  WN_K       (12 b)  — Galileo week of K_ROOT
    [60:80]  TOW_K      (20 b)  — time of week of K_ROOT [s]
    [80:128] ALPHA      (48 b)  — 6-byte TESLA nonce
    [128:128+ks] K_ROOT (ks b)  — root TESLA key
    [...+ds] DS          (nb)   — digital signature (512 b for ECDSA-P256)
    [...+p]  P_K         (nb)   — zero padding to fill 1008-bit DSM
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.data_structures import DSMPKRMessage, ECDSAType
from gnss.parser._bit_io import BitReader

# ---------------------------------------------------------------------------
# ICD-defined lookup tables
# ---------------------------------------------------------------------------

#: Key size in bits, indexed by KS field value (ICD Table 10).
KEY_SIZE_BITS_BY_KS: dict[int, int] = {
    0: 96,
    1: 104,
    2: 128,
    3: 160,
    4: 192,
    5: 224,
    6: 256,
}
# Legacy private alias — kept for internal backward compatibility.
_KEY_SIZE_BITS = KEY_SIZE_BITS_BY_KS

#: Tag size in bits, indexed by TS field value (ICD Table 11).
_TAG_SIZE_BITS: dict[int, int] = {
    0: 20,
    1: 24,
    2: 28,
    3: 32,
    4: 40,
}

#: Default digital signature size (ECDSA-P256, bits).
_DS_BITS_DEFAULT: int = 512

# ---------------------------------------------------------------------------
# Structure constants
# ---------------------------------------------------------------------------

#: Bits in one HKROOT section (per subframe).
HKROOT_BITS: int = 104

#: Bits in one DSM data block (HKROOT[32:104]).
DSM_BLOCK_BITS: int = 72

#: Number of blocks to reassemble a complete DSM message.
DSM_BLOCKS_PER_MESSAGE: int = 14

#: Total DSM message bits.
DSM_TOTAL_BITS: int = DSM_BLOCKS_PER_MESSAGE * DSM_BLOCK_BITS  # 1008

# DSM-PKR: compressed EC public key size in bits (ICD §5.5 Table).
_PK_SIZE_BITS: dict[ECDSAType, int] = {
    ECDSAType.P256: 264,  # 33 bytes (compressed P-256)
    ECDSAType.P521: 536,  # 67 bytes (compressed P-521)
}

#: SHA-256 hash size per Intermediate Tree Node (ICD §5.5.2).
_ITN_NODE_BITS: int = 256

#: Fixed header bits consumed before NPK in DSM-PKR:
#: NB_DPK(4) + MID(4) + NPKID(4) + NPKT(4) = 16 bits.
_DSM_PKR_FIXED_HEADER_BITS: int = 16

# NMA_STATUS codes (ICD §5.3.1)
NMA_STATUS_TEST: int = 0
NMA_STATUS_OPERATIONAL: int = 1
NMA_STATUS_DONT_USE: int = 2
NMA_STATUS_RESERVED: int = 3

# DSM_ID ranges
DSM_ID_KROOT_MAX: int = 11  # 0-11 → DSM-KROOT
DSM_ID_PKR: int = 12  # 12   → DSM-PKR (public key registration)


# ---------------------------------------------------------------------------
# Parsed data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HkrootSection:
    """Decoded 104-bit HKROOT section from one I/NAV subframe.

    Produced by :func:`parse_hkroot_section`.

    Attributes:
        nma_status:   NMA_STATUS field (0–3).
        chain_id:     CHAIN_ID field (0–3).
        chain_in_force: CIF flag (True = new chain starts this subframe).
        cidx:         Chain phase index CIDX (0–15).
        cpks:         Chain Phase and Key Size (4-bit, raw).
        dsm_id:       DSM block type (0-11 = KROOT, 12 = PKR).
        dsm_block_id: Sequential block index within DSM (0-13).
        dsm_data:     72-bit DSM data block (9 bytes).
    """

    nma_status: int
    chain_id: int
    chain_in_force: bool
    cidx: int
    cpks: int
    dsm_id: int
    dsm_block_id: int
    dsm_data: bytes  # 9 bytes = 72 bits

    @property
    def is_kroot_block(self) -> bool:
        """True if this block carries DSM-KROOT data (DSM_ID 0-11)."""
        return self.dsm_id <= DSM_ID_KROOT_MAX

    @property
    def is_pkr_block(self) -> bool:
        """True if this block carries DSM-PKR data (DSM_ID 12)."""
        return self.dsm_id == DSM_ID_PKR


@dataclass
class DsmKroot:
    """Partially or fully assembled DSM-KROOT message.

    Holds up to ``DSM_BLOCKS_PER_MESSAGE`` (14) blocks of 72 bits each.
    Call :meth:`is_complete` to check whether all blocks have arrived,
    then :meth:`assembled_bytes` to get the raw payload for
    :func:`parse_dsm_kroot`.

    Args:
        dsm_id: DSM_ID value identifying this chain/key.
    """

    dsm_id: int
    _blocks: dict[int, bytes] = field(default_factory=dict, repr=False)

    def add_block(self, block_id: int, data: bytes) -> None:
        """Store one 72-bit DSM data block.

        Args:
            block_id: Sequential block index (0-13).
            data:     9 bytes (72 bits) of DSM_DATA from the HKROOT section.

        Raises:
            ValueError: if block_id is out of range or data is the wrong length.
        """
        if not 0 <= block_id < DSM_BLOCKS_PER_MESSAGE:
            raise ValueError(f"block_id {block_id} out of range [0, {DSM_BLOCKS_PER_MESSAGE})")
        if len(data) != DSM_BLOCK_BITS // 8:
            raise ValueError(f"Expected {DSM_BLOCK_BITS // 8} bytes, got {len(data)}")
        self._blocks[block_id] = data

    def is_complete(self) -> bool:
        """Return True once all 14 blocks (IDs 0-13) have been received."""
        return len(self._blocks) == DSM_BLOCKS_PER_MESSAGE

    def missing_blocks(self) -> list[int]:
        """Return list of block IDs not yet received."""
        return [i for i in range(DSM_BLOCKS_PER_MESSAGE) if i not in self._blocks]

    def assembled_bytes(self) -> bytes:
        """Concatenate blocks 0-13 in order into a 126-byte (1008-bit) payload.

        Raises:
            RuntimeError: if the DSM is not yet complete.
        """
        if not self.is_complete():
            missing = self.missing_blocks()
            raise RuntimeError(f"DSM_ID={self.dsm_id} is incomplete: missing blocks {missing}")
        return b"".join(self._blocks[i] for i in range(DSM_BLOCKS_PER_MESSAGE))


@dataclass(frozen=True)
class ParsedHkroot:
    """Fully decoded DSM-KROOT message (assembled from 14 subframe blocks).

    Produced by :func:`parse_dsm_kroot`.  Field names match the ICD notation.

    Attributes:
        nb_dk:      Number of distinguished keys (NB_DK).
        pkid:       Public key ID (PKID).
        cidx:       Chain index (CIDX).
        hf:         Hash function selector (0=SHA-256).
        mf:         MAC function selector (0=HMAC-SHA256).
        ks:         Key size selector (see _KEY_SIZE_BITS).
        ts:         Tag size selector (see _TAG_SIZE_BITS).
        maclt:      MAC lookup table ID.
        wn_k:       Galileo week number of K_ROOT epoch.
        tow_k:      Time of week of K_ROOT epoch [s].
        alpha:      TESLA chain nonce (6 bytes).
        kroot:      Root TESLA key bytes (length = key_size_bits // 8).
        ds:         Digital signature bytes (typically 64 bytes for ECDSA-P256).
        key_size_bits: Actual key size resolved from KS.
        tag_size_bits: Actual tag size resolved from TS.
    """

    nb_dk: int
    pkid: int
    cidx: int
    hf: int
    mf: int
    ks: int
    ts: int
    maclt: int
    wn_k: int
    tow_k: int
    alpha: bytes  # 6 bytes
    kroot: bytes  # key_size_bits // 8 bytes
    ds: bytes
    key_size_bits: int
    tag_size_bits: int


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_hkroot_section(data: bytes) -> HkrootSection:
    """Parse a 104-bit HKROOT section from 13 bytes of OSNMA data.

    The HKROOT section occupies the first 104 bits of the 600-bit OSNMA
    block assembled from 15 I/NAV pages × 40 bits/page.

    Args:
        data: Exactly 13 bytes (104 bits) of HKROOT section data.

    Returns:
        Decoded :class:`HkrootSection`.

    Raises:
        ValueError: if ``data`` is not 13 bytes.
    """
    expected_bytes = HKROOT_BITS // 8
    if len(data) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes for HKROOT, got {len(data)}")
    r = BitReader(data)

    # NMA_H (16 bits)
    nma_status = r.read_uint(2)
    chain_id = r.read_uint(2)
    chain_in_force = r.read_bool()
    r.skip(11)  # NMA_H_RESERVED

    # TESLA_H (8 bits)
    cidx = r.read_uint(4)
    cpks = r.read_uint(4)

    # DSM header (8 bits)
    dsm_id = r.read_uint(4)
    dsm_block_id = r.read_uint(4)

    # DSM_DATA (72 bits = 9 bytes)
    dsm_data = r.read_bytes(9)

    return HkrootSection(
        nma_status=nma_status,
        chain_id=chain_id,
        chain_in_force=chain_in_force,
        cidx=cidx,
        cpks=cpks,
        dsm_id=dsm_id,
        dsm_block_id=dsm_block_id,
        dsm_data=dsm_data,
    )


def parse_dsm_kroot(
    dsm_payload: bytes,
    ds_bits: int = _DS_BITS_DEFAULT,
) -> ParsedHkroot:
    """Parse a fully assembled DSM-KROOT payload (1008 bits = 126 bytes).

    The payload is ``DsmKroot.assembled_bytes()`` after all 14 blocks arrive.

    Args:
        dsm_payload: 126 bytes of assembled DSM data (blocks 0–13 concatenated).
        ds_bits:     Digital signature size in bits (default 512 = ECDSA-P256).

    Returns:
        Decoded :class:`ParsedHkroot`.

    Raises:
        ValueError: if ``dsm_payload`` is not 126 bytes or KS/TS are unknown.
    """
    expected_bytes = DSM_TOTAL_BITS // 8
    if len(dsm_payload) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes for DSM-KROOT, got {len(dsm_payload)}")
    r = BitReader(dsm_payload)

    nb_dk = r.read_uint(4)
    pkid = r.read_uint(4)
    cidx = r.read_uint(4)
    hf = r.read_uint(4)
    mf = r.read_uint(4)
    ks = r.read_uint(4)
    ts = r.read_uint(4)
    maclt = r.read_uint(8)
    r.skip(12)  # reserved
    wn_k = r.read_uint(12)
    tow_k = r.read_uint(20)
    alpha = r.read_bytes(6)

    if ks not in _KEY_SIZE_BITS:
        raise ValueError(f"Unknown KS value {ks}; known: {list(_KEY_SIZE_BITS)}")
    if ts not in _TAG_SIZE_BITS:
        raise ValueError(f"Unknown TS value {ts}; known: {list(_TAG_SIZE_BITS)}")

    key_bits = _KEY_SIZE_BITS[ks]
    tag_bits = _TAG_SIZE_BITS[ts]

    # K_ROOT: key_bits long (must be byte-multiple for keys we support)
    if key_bits % 8:
        raise ValueError(f"key_size_bits={key_bits} is not a multiple of 8")
    kroot = r.read_bytes(key_bits // 8)

    # DS: ds_bits long (default 512 = 64 bytes)
    if ds_bits % 8:
        raise ValueError(f"ds_bits={ds_bits} is not a multiple of 8")
    ds = r.read_bytes(ds_bits // 8)

    # Remaining bits are P_K padding (ignored).

    return ParsedHkroot(
        nb_dk=nb_dk,
        pkid=pkid,
        cidx=cidx,
        hf=hf,
        mf=mf,
        ks=ks,
        ts=ts,
        maclt=maclt,
        wn_k=wn_k,
        tow_k=tow_k,
        alpha=alpha,
        kroot=kroot,
        ds=ds,
        key_size_bits=key_bits,
        tag_size_bits=tag_bits,
    )


# ---------------------------------------------------------------------------
# DSM-PKR structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedPkr:
    """Fully decoded DSM-PKR message (assembled from subframe blocks).

    Produced by :func:`parse_dsm_pkr`.  Field names follow ICD §5.5 notation.

    Attributes:
        nb_dpk:  NB_DPK field value; actual block count = nb_dpk + 1.
        mid:     Broadcast message ID (MID field; identifies transmission slot).
        npkid:   New Public Key ID — leaf index in the Merkle tree (0–15).
        npkt:    New Public Key Type (:class:`~core.data_structures.ECDSAType`).
        npk:     Compressed EC public key bytes (33 B for P-256, 67 B for P-521).
        itn:     Intermediate Tree Nodes — Merkle authentication path from leaf
                 toward root.  Each entry is 32 bytes (one SHA-256 hash).
                 Length = ``(actual_bits − 16 − npk_bits) // 256``, where
                 ``actual_bits = (nb_dpk + 1) × 72``.
    """

    nb_dpk: int
    mid: int
    npkid: int
    npkt: ECDSAType
    npk: bytes  # len = _PK_SIZE_BITS[npkt] // 8
    itn: tuple[bytes, ...]  # each entry is _ITN_NODE_BITS // 8 = 32 bytes


def parsed_pkr_to_message(pkr: ParsedPkr) -> DSMPKRMessage:
    """Convert a parser-layer :class:`ParsedPkr` to a verifier-layer :class:`DSMPKRMessage`.

    Field mapping (ICD §5.5 notation → semantic names used by the verifier):

    ======== ========== ====================
    ParsedPkr            DSMPKRMessage
    ======== ========== ====================
    npkid    → pkid      Merkle leaf index
    npkt     → pktype    ECDSAType enum
    npk      → public_key Compressed EC key
    itn      → merkle_nodes Auth path nodes
    ======== ========== ====================

    Args:
        pkr: Decoded DSM-PKR produced by :func:`parse_dsm_pkr`.

    Returns:
        :class:`~core.data_structures.DSMPKRMessage` ready for
        :meth:`~gnss.ecdsa_verifier.ECDSAVerifier.verify_public_key`.
    """
    return DSMPKRMessage(
        pkid=pkr.npkid,
        pktype=pkr.npkt,
        public_key=pkr.npk,
        merkle_nodes=pkr.itn,
    )


@dataclass
class DsmPkr:
    """Partially or fully assembled DSM-PKR message.

    Holds up to ``DSM_BLOCKS_PER_MESSAGE`` (14) blocks of 72 bits each.
    Analogous to :class:`DsmKroot` but for DSM-PKR data (DSM_ID = 12).

    Call :meth:`is_complete` to check whether all blocks have arrived,
    then :meth:`assembled_bytes` to get the raw payload for
    :func:`parse_dsm_pkr`.

    Args:
        dsm_id: DSM_ID value — should be 12 (:data:`DSM_ID_PKR`).
    """

    dsm_id: int
    _blocks: dict[int, bytes] = field(default_factory=dict, repr=False)

    def add_block(self, block_id: int, data: bytes) -> None:
        """Store one 72-bit DSM data block.

        Args:
            block_id: Sequential block index (0-13).
            data:     9 bytes (72 bits) of DSM_DATA from the HKROOT section.

        Raises:
            ValueError: if block_id is out of range or data is the wrong length.
        """
        if not 0 <= block_id < DSM_BLOCKS_PER_MESSAGE:
            raise ValueError(f"block_id {block_id} out of range [0, {DSM_BLOCKS_PER_MESSAGE})")
        if len(data) != DSM_BLOCK_BITS // 8:
            raise ValueError(f"Expected {DSM_BLOCK_BITS // 8} bytes, got {len(data)}")
        self._blocks[block_id] = data

    def is_complete(self) -> bool:
        """Return True once all 14 blocks (IDs 0-13) have been received."""
        return len(self._blocks) == DSM_BLOCKS_PER_MESSAGE

    def missing_blocks(self) -> list[int]:
        """Return list of block IDs not yet received."""
        return [i for i in range(DSM_BLOCKS_PER_MESSAGE) if i not in self._blocks]

    def assembled_bytes(self) -> bytes:
        """Concatenate blocks 0-13 in order into a 126-byte (1008-bit) payload.

        Raises:
            RuntimeError: if the DSM is not yet complete.
        """
        if not self.is_complete():
            missing = self.missing_blocks()
            raise RuntimeError(f"DSM_ID={self.dsm_id} is incomplete: missing blocks {missing}")
        return b"".join(self._blocks[i] for i in range(DSM_BLOCKS_PER_MESSAGE))


# ---------------------------------------------------------------------------
# DSM-PKR parser
# ---------------------------------------------------------------------------


def parse_dsm_pkr(dsm_payload: bytes) -> ParsedPkr:
    """Parse a fully assembled DSM-PKR payload (1008 bits = 126 bytes).

    The payload is ``DsmPkr.assembled_bytes()`` after all 14 blocks arrive.

    Field layout (ICD §5.5):
        NB_DPK  (4 b)  — number of valid blocks minus one
        MID     (4 b)  — broadcast message ID
        NPKID   (4 b)  — new public key ID (Merkle leaf index, 0–15)
        NPKT    (4 b)  — new public key type (0 = P-256, 1 = P-521)
        NPK     (var)  — compressed EC public key (264 b / P-256; 536 b / P-521)
        ITN     (var)  — Intermediate Tree Nodes; n_itn × 256 bits where
                         n_itn = (actual_bits − 16 − npk_bits) // 256 and
                         actual_bits = (NB_DPK + 1) × 72
        P_DP    (var)  — zero padding (ignored)

    .. note::
        The bit layout above is derived from code inspection and community
        documentation.  It has **not** been independently verified against a
        licensed copy of the Galileo OSNMA SIS ICD OS-SIS-ICD-OSNMA §5.5.
        Treat the field offsets as best-effort until cross-checked with the
        official ESA document.

    Args:
        dsm_payload: 126 bytes of assembled DSM data (blocks 0–13 concatenated).

    Returns:
        Decoded :class:`ParsedPkr`.

    Raises:
        ValueError: if ``dsm_payload`` is not 126 bytes, NB_DPK is out of range,
                    NPKT is unknown, or the payload is too small for the key.
    """
    expected_bytes = DSM_TOTAL_BITS // 8
    if len(dsm_payload) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes for DSM-PKR, got {len(dsm_payload)}")
    r = BitReader(dsm_payload)

    nb_dpk = r.read_uint(4)
    mid = r.read_uint(4)
    npkid = r.read_uint(4)
    npkt_raw = r.read_uint(4)

    if nb_dpk >= DSM_BLOCKS_PER_MESSAGE:
        raise ValueError(f"NB_DPK={nb_dpk} out of range [0, {DSM_BLOCKS_PER_MESSAGE - 1}]")

    try:
        npkt = ECDSAType(npkt_raw)
    except ValueError:
        raise ValueError(f"Unknown NPKT value {npkt_raw}; known: {list(ECDSAType)}")

    npk_bits = _PK_SIZE_BITS[npkt]
    actual_bits = (nb_dpk + 1) * DSM_BLOCK_BITS
    available_for_itn = actual_bits - _DSM_PKR_FIXED_HEADER_BITS - npk_bits
    if available_for_itn < 0:
        raise ValueError(
            f"NB_DPK={nb_dpk} ({actual_bits} bits) insufficient for "
            f"NPKT={npkt.name} key ({npk_bits} bits)"
        )

    n_itn = available_for_itn // _ITN_NODE_BITS  # floor; remainder = intra-ITN padding

    npk = r.read_bytes(npk_bits // 8)

    itn = tuple(r.read_bytes(_ITN_NODE_BITS // 8) for _ in range(n_itn))

    # Remaining bits are P_DP padding (ignored).

    return ParsedPkr(
        nb_dpk=nb_dpk,
        mid=mid,
        npkid=npkid,
        npkt=npkt,
        npk=npk,
        itn=itn,
    )
