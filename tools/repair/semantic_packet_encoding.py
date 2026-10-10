"""Lossless namespace dictionary for complete semantic packet transport.

This is a presentation encoding, never an ontology alias, projection or omission.
The complete expanded packet must equal its original digest before admission.
"""

from dataclasses import replace
import json
from difflib import SequenceMatcher
import re

from exact.repair.records import canonical_hash

REVISION = "semantic-packet-lossless-text-dictionary/v1"
NAMESPACE = re.compile(r"urn:exact:generated:[0-9a-f]{16}:")
MARKER = "~N"
BLOCK_MARKER = "~B"


def _texts(packet):
    return (
        *packet.original_observation,
        *(value["text"] for value in packet.evidence.values()),
        *packet.local_context,
        *packet.plan_a.content,
        *packet.plan_b.content,
    )


def _map_texts(packet, transform):
    return replace(
        packet,
        original_observation=tuple(map(transform, packet.original_observation)),
        evidence={
            key: {**value, "text": transform(value["text"])}
            for key, value in packet.evidence.items()
        },
        local_context=tuple(map(transform, packet.local_context)),
        plan_a=replace(packet.plan_a, content=tuple(map(transform, packet.plan_a.content))),
        plan_b=replace(packet.plan_b, content=tuple(map(transform, packet.plan_b.content))),
    )


def header(dictionary):
    return (
        "Lossless namespace dictionary; representation only. In every observation, evidence "
        "text, context and complete plan content, expand each ~N<number>~ token to its "
        "exact namespace below before interpreting an IRI. This notation does not rename "
        "ontology entities, omit axioms or change either repair. Full theories remain present. "
        + json.dumps(dictionary, sort_keys=True, separators=(",", ":"))
    )


def block_headers(blocks):
    return tuple(
        "Shared literal full-theory block "
        + key
        + " (substitute these exact characters wherever the token occurs; no omitted axioms):\n"
        + value
        for key, value in blocks.items()
    )


def encode(packet):
    texts = _texts(packet)
    if any(MARKER in text or BLOCK_MARKER in text for text in texts):
        raise ValueError("Packet already contains a namespace dictionary marker")
    namespaces = sorted({match for text in texts for match in NAMESPACE.findall(text)})
    dictionary = {f"{MARKER}{i}~": ns for i, ns in enumerate(namespaces)}
    reverse = {v: k for k, v in dictionary.items()}
    compact = _map_texts(packet, lambda text: NAMESPACE.sub(lambda m: reverse[m.group()], text))
    # Share repeated literal full-theory blocks once. Both plans still expand
    # exactly; this is never a theory slice, extracted module or projection.
    left, right = compact.plan_a.content[0], compact.plan_b.content[0]
    a, b = left.splitlines(keepends=True), right.splitlines(keepends=True)
    shared = [
        "".join(a[match.a : match.a + match.size])
        for match in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
        if match.size
    ]
    shared = sorted(set(block for block in shared if len(block) >= 128), key=lambda x: (-len(x), x))
    blocks = {}
    for block in shared:
        # Earlier replacements cannot be hidden inside a later dictionary value.
        if block not in compact.plan_a.content[0] or block not in compact.plan_b.content[0]:
            continue
        marker = f"{BLOCK_MARKER}{len(blocks)}~"
        blocks[marker] = block
        compact = _map_texts(compact, lambda text: text.replace(block, marker))
    context = (header(dictionary), *block_headers(blocks), *compact.local_context)
    compact = replace(compact, local_context=context)
    proof = dict(
        schema=REVISION,
        dictionary=dictionary,
        blocks=blocks,
        original_packet_hash=packet.content_hash,
        encoded_packet_hash=compact.content_hash,
        complete_plans=True,
        ontology_identities_changed=False,
        axiom_omissions=0,
    )
    if decode(compact, proof).to_dict() != packet.to_dict():
        raise ValueError("Namespace dictionary failed exact full-packet round trip")
    return compact, proof


def decode(packet, proof):
    if proof.get("schema") != REVISION or packet.content_hash != proof["encoded_packet_hash"]:
        raise ValueError("Encoded packet identity changed")
    dictionary = proof["dictionary"]
    if any(
        not re.fullmatch(r"~N\d+~", key) or not NAMESPACE.fullmatch(value)
        for key, value in dictionary.items()
    ):
        raise ValueError("Invalid lossless namespace dictionary")
    if not packet.local_context or packet.local_context[0] != header(dictionary):
        raise ValueError("Lossless dictionary header changed")
    blocks = proof["blocks"]
    if any(
        not re.fullmatch(r"~B\d+~", key) or BLOCK_MARKER in value for key, value in blocks.items()
    ):
        raise ValueError("Invalid shared full-theory block dictionary")
    expected_headers = block_headers(blocks)
    if tuple(packet.local_context[1 : 1 + len(blocks)]) != expected_headers:
        raise ValueError("Shared complete-theory context changed")
    value = replace(packet, local_context=packet.local_context[1 + len(blocks) :])

    def expand(text):
        for key, block in blocks.items():
            text = text.replace(key, block)
        for key, namespace in dictionary.items():
            text = text.replace(key, namespace)
        return text

    value = _map_texts(value, expand)
    if value.content_hash != proof["original_packet_hash"]:
        raise ValueError("Expanded packet differs from original native packet")
    return value


def encoding_identity():
    return canonical_hash((REVISION, NAMESPACE.pattern, MARKER))
