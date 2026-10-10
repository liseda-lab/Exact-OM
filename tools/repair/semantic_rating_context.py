"""Recover exact native rating contexts without modifying the frozen wire encoder."""

from exact.repair.records import read_record
from tools.repair.semantic_packet_encoding import decode, encode


def rating_context(packet, reference):
    """Do not confuse pair-specific transport headers with semantic evidence.

    Both directions and the complete native record must agree before its text
    enters the repeated-rating context identity. Raw response validation and
    comparison identities still use the unchanged wire packet.
    """
    original = read_record(reference["original_packet"])
    proof = reference["proof"]
    if decode(packet, proof) != original or encode(original) != (packet, proof):
        raise ValueError("Rating context lacks the exact lossless native round trip")
    return original
