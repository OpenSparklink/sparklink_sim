"""Control length semantics, TXS-10002-2025 §6.4.1.3/.4 and §6.4.2.2.2.

A2/A3 count PHY information plus any MIC, excluding data CRC. Non-retransmitted
B1 counts those bytes AND data CRC. Raw zero B1 explicitly has no data CRC.
MCS15/CBG retransmission requires previous scheduling context and is not decoded
by this helper. Its caller must reject it before interpreting the bitmap as bytes.
"""

__all__ = ["control_data_length"]


def control_data_length(raw_length: int, control_format: str, crc_bits: int, *,
                        contract: str = "standard") -> tuple[int, bool]:
    """Return (PHY information/MIC bytes, data CRC present), without CRC-based retry.

    The legacy contract is an explicit pre-receive choice for already frozen,
    nonstandard B1 model inputs. It must never be inferred from received CRCs.
    """
    if (type(raw_length) is not int or not 0 <= raw_length <= 2047
            or type(crc_bits) is not int or crc_bits not in (24, 32)
            or control_format not in ("A2", "A3", "B1")
            or contract not in ("standard", "legacy-b1-payload-length")
            or (contract != "standard" and control_format != "B1")):
        raise ValueError("Need a bounded control length, CRC24/32 and explicit length contract")
    if control_format != "B1" or contract != "standard":
        return raw_length, True
    if raw_length == 0:
        return 0, False
    if raw_length < crc_bits // 8:
        raise ValueError("Nonzero B1 wire length is shorter than its data CRC")
    return raw_length - crc_bits // 8, True
