"""Independent prepared-data truth and actual decoder-position receipts for V8.

The oracle reads immutable physical records and the pinned byte vocabulary. It
does not call encode_record, the window iterator, the target planner, or a decoder.
Runtime receipts are populated at the actual categorical/lexical loss calls.
Only labels, indices, masks and public supports are retained; never activations,
predictions, logits, routing assignments or parameter values.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict

from megatoken.contracts import ROLE_TO_ID
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.training.target_inventory_v8 import inventory_episode_source_id

POSITION_SCHEMA = "semantic-v8-optimizer-position-exposure/1"
TEXT_PROFILE = "admitted-v8-plain-text/1"
POSITION_LOSSES = ("lexical", "reconstruction", "structure", "terminal", "modality")


def _require(condition, message):
    if not condition:
        raise ValueError("V8 position audit: " + message)


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _seal(value):
    body = {key: child for key, child in value.items() if key != "digest"}
    return {**body, "digest": stable_algorithm_hash(body)}


def _integer_support(values):
    """Canonical disjoint half-open runs, preserving every allowed class."""
    values = list(values)
    _require(
        bool(values)
        and all(type(value) is int and value >= 0 for value in values)
        and len(values) == len(set(values)),
        "categorical support contains missing, invalid or duplicate classes",
    )
    runs = []
    for value in sorted(values):
        if runs and runs[-1][1] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, value + 1])
    return runs


def _lexical_alphabet(codec):
    # UTF-8 and raw-byte text have the same byte-token support. Derive it from
    # the pinned vocabulary rather than the decoder's mutable alphabet buffer.
    mask = bytearray(codec.vocab_size)
    for token in codec.valid_token_ids:
        mask[token] = 1
    return _hash_bytes(bytes(mask))


def independent_text_target(
    record,
    *,
    codec,
    generated_capacity,
    max_capacity,
    modality_support=(0,),
    _alphabet_sha256=None,
):
    """Reconstruct one ordinary V8 text target without any planner labels.

    Cached physical partitions and recorded emitted IDs are authoritative input
    identities. Re-encoding their bytes could erase a valid noncanonical emitted
    tokenization. Byte boundaries and eligibility are rebuilt independently.
    """
    _require(record.policy_mask is True, "observations cannot receive target-position credit")
    _require(record.role in ROLE_TO_ID, "target role is unknown")
    _require(
        type(record.logical_id) is str
        and bool(record.logical_id)
        and type(record.record_id) is str
        and bool(record.record_id),
        "prepared target identity is missing",
    )
    _require(
        record.media_type.startswith("text/")
        and type(record.payload) is bytes
        and type(record.part_index) is int
        and type(record.part_count) is int
        and 0 <= record.part_index < record.part_count,
        "plain-text target metadata is invalid",
    )
    _require(
        type(generated_capacity) is int
        and type(max_capacity) is int
        and min(generated_capacity, max_capacity) > 0,
        "target geometry is invalid",
    )
    ids = record.emitted_ids if record.emitted_ids is not None else record.cached_ids
    if ids is not None:
        _require(record.codec_fingerprint == codec.fingerprint, "record tokenizer changed")
        ids = list(ids)
    else:
        ids = list(codec.encode(record.payload))
    _require(
        all(type(token) is int and 0 <= token < codec.vocab_size for token in ids),
        "prepared token IDs are invalid",
    )
    pieces = [codec.token_bytes(token) for token in ids]
    _require(b"".join(pieces) == record.payload, "prepared token IDs do not spell the exact bytes")
    _require(
        1 <= len(ids) <= min(generated_capacity, max_capacity),
        "target length is outside the public semantic text support",
    )
    try:
        record.payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("V8 position audit: target is not valid semantic UTF-8") from error
    edges = [0]
    for piece in pieces:
        edges.append(edges[-1] + len(piece))
    emitted_mask = record.emitted_free_mask
    if emitted_mask is not None:
        _require(
            record.emitted_ids is not None
            and len(emitted_mask) == len(ids)
            and all(type(value) is bool for value in emitted_mask),
            "recorded emission mask is invalid",
        )
    spans = record.free_spans
    if spans is not None:
        previous = 0
        for span in spans:
            _require(
                type(span.start) is int
                and type(span.end) is int
                and previous <= span.start <= span.end <= len(record.payload),
                "free byte spans are invalid, overlapping or reordered",
            )
            previous = span.end
    mask = [
        (spans is None or any(span.start <= left and right <= span.end for span in spans))
        and (emitted_mask is None or emitted_mask[index])
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True))
    ]
    support = tuple(modality_support)
    _require(0 in support and set(support) <= {0, 1, 2, 3}, "public text modality is unavailable")
    return _seal(
        {
            "profile": TEXT_PROFILE,
            "logicalTargetId": record.logical_id,
            "physicalRecordId": record.record_id,
            "partIndex": record.part_index,
            "partCount": record.part_count,
            "payloadSha256": _hash_bytes(record.payload),
            "payloadBytes": len(record.payload),
            "codecFingerprint": codec.fingerprint,
            "tokenIds": ids,
            "tokenByteEdges": edges,
            "freeMask": mask,
            "tokenSequenceDigest": stable_algorithm_hash(ids),
            "alphabetSha256": _lexical_alphabet(codec)
            if _alphabet_sha256 is None
            else _alphabet_sha256,
            "vocabularySize": codec.vocab_size,
            "capacity": min(1 << (max(1, len(ids)) - 1).bit_length(), max_capacity),
            "roleId": ROLE_TO_ID[record.role],
            "terminal": record.part_index + 1 == record.part_count,
            "supports": {
                "length": [[1, min(generated_capacity, max_capacity) + 1]],
                "mode": [[0, 1]],
                "terminal": [[0, 2]],
                "modality": _integer_support(support),
            },
        }
    )


class TextReadoutReceipt:
    """One executed prior or reconstruction readout, checked against raw truth."""

    def __init__(self, oracle, branch):
        _require(branch in {"prior", "reconstruction"}, "unknown decoder branch")
        self.oracle = oracle
        self.branch = branch
        self.started = False
        self.decisions = {}
        self.positions = []
        self.labels = []
        self.alphabet = None
        self.closed = False

    def begin(self, payloads, ids, capacities):
        _require(not self.started and not self.closed, "decoder readout was duplicated")
        _require(
            len(payloads) == len(ids) == len(capacities) == 1
            and type(payloads[0]) is bytes
            and all(type(token) is int for token in ids[0])
            and type(capacities[0]) is int,
            "decoder readout does not identify one exact text target",
        )
        _require(
            _hash_bytes(payloads[0]) == self.oracle["payloadSha256"]
            and list(ids[0]) == self.oracle["tokenIds"]
            and capacities[0] == self.oracle["capacity"],
            "decoder input bytes, token sequence or workspace differ from prepared truth",
        )
        self.started = True

    def categorical(self, name, rows, local_labels, support):
        _require(self.started and not self.closed, "categorical receipt is outside a readout")
        _require(name in {"length", "mode", "terminal"}, "unknown plain-text decision")
        _require(name not in self.decisions, "categorical decision was duplicated")
        _require(
            rows == [0] and all(type(row) is int for row in rows) and len(local_labels) == 1,
            "categorical row membership changed",
        )
        runs = _integer_support(support)
        local = local_labels[0]
        _require(type(local) is int and 0 <= local < len(support), "categorical label is invalid")
        label = support[local]
        expected = {
            "length": len(self.oracle["tokenIds"]),
            "mode": 0,
            "terminal": int(self.oracle["terminal"]),
        }[name]
        _require(
            label == expected and runs == self.oracle["supports"][name],
            "executed categorical label or support differs from prepared truth",
        )
        self.decisions[name] = {"label": label, "support": runs}

    def lexical(self, kind, coordinates, labels, alphabet):
        _require(self.started and not self.closed, "lexical receipt is outside a readout")
        _require(kind == "utf8", "executed lexical alphabet kind differs from semantic text")
        _require(len(coordinates) == len(labels) and bool(labels), "lexical rows and labels differ")
        _require(
            str(alphabet.dtype) == "torch.bool"
            and alphabet.ndim == 1
            and alphabet.numel() == self.oracle["vocabularySize"],
            "executed lexical alphabet mask has invalid geometry",
        )
        mask_hash = _hash_bytes(alphabet.detach().cpu().numpy().tobytes())
        _require(
            mask_hash == self.oracle["alphabetSha256"], "executed lexical alphabet mask changed"
        )
        self.alphabet = mask_hash
        for coordinate, label in zip(coordinates, labels, strict=True):
            _require(
                len(coordinate) == 2
                and type(coordinate[0]) is int
                and coordinate[0] == 0
                and type(coordinate[1]) is int
                and type(label) is int,
                "executed lexical coordinate or label is invalid",
            )
            position = coordinate[1]
            _require(position not in self.positions, "executed lexical position was duplicated")
            _require(
                0 <= position < len(self.oracle["tokenIds"])
                and self.oracle["freeMask"][position]
                and label == self.oracle["tokenIds"][position],
                "executed lexical position, mask or label differs from prepared truth",
            )
            self.positions.append(position)
            self.labels.append(label)

    def finish(self):
        _require(self.started and not self.closed, "decoder readout is missing or duplicated")
        expected = [index for index, eligible in enumerate(self.oracle["freeMask"]) if eligible]
        _require(
            sorted(self.positions) == expected, "decoder omitted one or more eligible positions"
        )
        _require(
            set(self.decisions) == {"length", "mode", "terminal"},
            "decoder omitted a categorical decision",
        )
        self.closed = True
        rows = sorted(zip(self.positions, self.labels, strict=True))
        return {
            "branch": self.branch,
            "tokenSequenceDigest": self.oracle["tokenSequenceDigest"],
            "capacity": self.oracle["capacity"],
            "positions": [position for position, _ in rows],
            "labels": [label for _, label in rows],
            "mask": [index in self.positions for index in range(len(self.oracle["tokenIds"]))],
            "alphabetSha256": self.alphabet,
            "decisions": {name: self.decisions[name] for name in sorted(self.decisions)},
        }


def installed_modality_ids(model):
    from megatoken.model.media import REALIZERS

    return [
        index
        for index, name in enumerate(("text", "audio", "image", "video"))
        if name == "text" or REALIZERS[name] in model.media_realizers.packs
    ]


def position_objective(config, model_config, codec, modalities):
    """Bind audit history to the same objective identity used by target inventory."""
    identity = asdict(config)
    for key in (
        "output",
        "max_steps",
        "checkpoint_every",
        "keep_checkpoints",
        "log_every",
        "device",
    ):
        identity.pop(key, None)
    weights = dict(config.loss_weights)
    weights["router_balance"] = 0.0
    _require(
        all(
            type(name) is str
            and type(weight) in {int, float}
            and math.isfinite(weight)
            and weight >= 0
            for name, weight in weights.items()
        ),
        "objective weights are invalid",
    )
    return {
        "trainingConfigDigest": stable_algorithm_hash(identity),
        "modelConfigDigest": stable_algorithm_hash(model_config.to_dict()),
        "codecFingerprint": codec.fingerprint,
        "installedModalities": list(modalities),
        "effectiveLossWeights": weights,
        "routingBalanceCoefficient": config.loss_weights["router_balance"],
    }


def independent_episode_targets(episode, *, codec, config, model_config, modalities, records=None):
    """All physical policy targets, before looking at windows or prepared plans."""
    annotations = json.loads(episode.supervision.annotations_json)
    reasons = []
    if json.loads(episode.tools_json):
        reasons.append("structured tool-action masks need a separate public-schema oracle")
    if annotations.get("native_action_tapes"):
        reasons.append("native action-tape/public-support branches need a separate oracle")
    events = {event.record_id: event for group in episode.meridians for event in group.events}
    selected_records = episode.records if records is None else records
    targets = {}
    # Derive the pinned alphabet once for this immutable prepared input, without
    # sharing a mutable decoder buffer or repeatedly scanning its vocabulary.
    alphabet_sha256 = _lexical_alphabet(codec)
    for record in selected_records:
        if not record.policy_mask:
            continue
        _require(record.record_id not in targets, "prepared corpus duplicated a physical target")
        event = events.get(record.record_id)
        local_reasons = list(reasons)
        if event is None or event.modality != "text" or not record.media_type.startswith("text/"):
            local_reasons.append(
                "non-text or unbound event positions need their own modality oracle"
            )
        if local_reasons:
            targets[record.record_id] = {
                "logicalTargetId": record.logical_id,
                "physicalRecordId": record.record_id,
                "status": "unmeasured",
                "reasons": local_reasons,
                "oracle": None,
            }
            continue
        settings = json.loads(event.settings_json)
        allowed = settings.get("allowed_modalities")
        names = ("text", "audio", "image", "video")
        if allowed is not None:
            _require(
                type(allowed) is list
                and bool(allowed)
                and len(allowed) == len(set(allowed))
                and all(type(name) is str and name in names for name in allowed),
                "prepared public modality constraints are invalid",
            )
        support = [index for index in modalities if allowed is None or names[index] in allowed]
        targets[record.record_id] = {
            "logicalTargetId": record.logical_id,
            "physicalRecordId": record.record_id,
            "status": "verified",
            "reasons": [],
            "oracle": independent_text_target(
                record,
                codec=codec,
                generated_capacity=config.generated_capacity,
                max_capacity=model_config.max_capacity,
                modality_support=support,
                _alphabet_sha256=alphabet_sha256,
            ),
        }
    return targets


class PositionStepAudit:
    """Labels selected by one real grouped loss execution, before optimizer commit."""

    def __init__(self, targets, weights):
        self.targets = targets
        self.weights = weights
        self.supported = all(target["status"] == "verified" for target in targets)
        self.sinks = {}
        self.modality_rows = None
        self.input_rows = None
        self.finished = False

    def decoder_sinks(self, branch):
        if not self.supported:
            return None
        _require(branch not in self.sinks, "grouped decoder branch was duplicated")
        sinks = [TextReadoutReceipt(target["oracle"], branch) for target in self.targets]
        self.sinks[branch] = sinks
        return sinks

    def inputs(self, event_ids, roles):
        if not self.supported:
            return
        _require(self.input_rows is None, "grouped target identity receipt was duplicated")
        actual_roles = roles.detach().cpu().tolist()
        _require(
            list(event_ids) == [target["physicalRecordId"] for target in self.targets]
            and actual_roles == [[target["oracle"]["roleId"] for target in self.targets]],
            "actual grouped target identity or role differs from prepared truth",
        )
        self.input_rows = actual_roles[0]

    def modality(self, labels, mask):
        if not self.supported:
            return
        _require(self.modality_rows is None, "modality loss receipt was duplicated")
        labels = labels.detach().cpu().tolist()
        mask = mask.detach().cpu().tolist()
        _require(
            len(labels) == len(mask) == 1 and len(labels[0]) == len(mask[0]) == len(self.targets),
            "modality loss has changed target membership",
        )
        rows = []
        for target, label, bits in zip(self.targets, labels[0], mask[0], strict=True):
            _require(
                type(label) is int
                and label == 0
                and len(bits) == 4
                and all(type(bit) is bool for bit in bits),
                "actual modality label or mask is invalid",
            )
            support = _integer_support(index for index, bit in enumerate(bits) if bit)
            _require(
                support == target["oracle"]["supports"]["modality"],
                "actual modality support differs from prepared public capabilities",
            )
            rows.append({"label": label, "support": support})
        self.modality_rows = rows

    def finish(self, losses):
        _require(not self.finished, "grouped position audit was duplicated")
        self.finished = True
        counts = {name: value[1] for name, value in losses.items()}
        _require(
            all(type(count) is int and count >= 0 for count in counts.values()),
            "actual objective denominators are malformed",
        )
        actual = []
        if self.supported:
            _require(
                set(self.sinks) == {"prior", "reconstruction"}
                and self.input_rows is not None
                and self.modality_rows is not None,
                "an actual decoder pass, target binding or modality loss was not observed",
            )
            observed_counts = dict.fromkeys(POSITION_LOSSES, 0)
            for index, target in enumerate(self.targets):
                readouts = {
                    branch: self.sinks[branch][index].finish()
                    for branch in ("prior", "reconstruction")
                }
                observed_counts["lexical"] += len(readouts["prior"]["positions"])
                observed_counts["reconstruction"] += len(readouts["reconstruction"]["positions"])
                observed_counts["structure"] += 4
                observed_counts["terminal"] += 2
                observed_counts["modality"] += 1
                actual.append(
                    {**target, "readouts": readouts, "modality": self.modality_rows[index]}
                )
            _require(
                all(counts.get(name, 0) == count for name, count in observed_counts.items()),
                "actual loss counts differ from the executed position/decision receipts",
            )
        else:
            actual = [
                {
                    **target,
                    "status": "unmeasured",
                    "reasons": target["reasons"]
                    or ["a grouped peer requires an unsupported position oracle"],
                    "oracle": None,
                    "readouts": None,
                    "modality": None,
                }
                for target in self.targets
            ]
        unmeasured = {
            name: count
            for name, count in counts.items()
            if count
            and self.weights.get(name, 0) > 0
            and (name not in POSITION_LOSSES or not self.supported)
        }
        return actual, unmeasured


class PositionWindowAudit:
    """Match actual step receipts to independently enumerated window input records."""

    def __init__(self, batch, *, codec, config, model_config, modalities, objective):
        _require(len(batch.windows) == 1, "position audit requires one independent episode")
        item = batch.windows[0]
        self.registry = config.semantic_v8_policy["registryDigest"]
        self.source = inventory_episode_source_id(item.episode, registry_digest=self.registry)
        self.window_index = item.window.index
        self.objective = objective
        independent = independent_episode_targets(
            item.episode,
            codec=codec,
            config=config,
            model_config=model_config,
            modalities=modalities,
            records=item.episode.records[item.window.start : item.window.end],
        )
        self.expected = {
            record.record_id: independent[record.record_id]
            for record in item.episode.records[item.window.start : item.window.end]
            if record.policy_mask
        }
        self.seen = {}
        self.unmeasured = {}

    def step(self, records):
        ids = [entry.record.record_id for entry in records]
        _require(len(ids) == len(set(ids)), "a grouped execution duplicated a physical target")
        _require(
            all(identity in self.expected and identity not in self.seen for identity in ids),
            "execution invented, observed or repeated a target outside the prepared window",
        )
        return PositionStepAudit(
            [self.expected[identity] for identity in ids], self.objective["effectiveLossWeights"]
        )

    def accept(self, step, losses):
        records, unmeasured = step.finish(losses)
        for record in records:
            identity = record["physicalRecordId"]
            _require(identity not in self.seen, "executed physical target was duplicated")
            self.seen[identity] = record
        for name, count in unmeasured.items():
            self.unmeasured[name] = self.unmeasured.get(name, 0) + count

    def finish(self):
        _require(set(self.seen) == set(self.expected), "execution omitted a prepared window target")
        return {
            "schema": POSITION_SCHEMA,
            "registryDigest": self.registry,
            "objectiveDigest": stable_algorithm_hash(self.objective),
            "sourceId": self.source,
            "windowIndex": self.window_index,
            "records": [self.seen[key] for key in sorted(self.seen)],
            "unmeasuredObjectives": dict(sorted(self.unmeasured.items())),
        }


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_target_receipt(record, *, logical_id, record_id):
    fields = {
        "logicalTargetId",
        "physicalRecordId",
        "status",
        "reasons",
        "oracle",
        "readouts",
        "modality",
    }
    _require(
        type(record) is dict
        and set(record) == fields
        and record["logicalTargetId"] == logical_id
        and record["physicalRecordId"] == record_id
        and record["status"] in {"verified", "unmeasured"}
        and type(record["reasons"]) is list
        and all(type(reason) is str and reason for reason in record["reasons"]),
        "retained target position receipt has changed fields or identity",
    )
    if record["status"] == "unmeasured":
        _require(
            bool(record["reasons"])
            and record["oracle"] is None
            and record["readouts"] is None
            and record["modality"] is None,
            "unsupported target falsely carries verified positional credit",
        )
        return
    oracle = record["oracle"]
    oracle_fields = {
        "profile",
        "logicalTargetId",
        "physicalRecordId",
        "partIndex",
        "partCount",
        "payloadSha256",
        "payloadBytes",
        "codecFingerprint",
        "tokenIds",
        "tokenByteEdges",
        "freeMask",
        "tokenSequenceDigest",
        "alphabetSha256",
        "vocabularySize",
        "capacity",
        "roleId",
        "terminal",
        "supports",
        "digest",
    }
    _require(
        not record["reasons"]
        and type(oracle) is dict
        and set(oracle) == oracle_fields
        and oracle["profile"] == TEXT_PROFILE
        and oracle["logicalTargetId"] == logical_id
        and oracle["physicalRecordId"] == record_id
        and _seal(oracle) == oracle
        and all(
            _digest(oracle[key])
            for key in (
                "payloadSha256",
                "codecFingerprint",
                "tokenSequenceDigest",
                "alphabetSha256",
            )
        ),
        "retained prepared position oracle has invalid identity or seal",
    )
    ids, mask, edges = oracle["tokenIds"], oracle["freeMask"], oracle["tokenByteEdges"]
    _require(
        type(ids) is list
        and bool(ids)
        and type(mask) is list
        and len(mask) == len(ids)
        and type(edges) is list
        and len(edges) == len(ids) + 1
        and type(oracle["vocabularySize"]) is int
        and oracle["vocabularySize"] > 0
        and all(type(token) is int and 0 <= token < oracle["vocabularySize"] for token in ids)
        and all(type(bit) is bool for bit in mask)
        and all(type(edge) is int for edge in edges)
        and edges[0] == 0
        and edges[-1] == oracle["payloadBytes"]
        and all(left < right for left, right in zip(edges[:-1], edges[1:], strict=True))
        and type(oracle["payloadBytes"]) is int
        and oracle["payloadBytes"] > 0
        and type(oracle["capacity"]) is int
        and oracle["capacity"] >= len(ids)
        and type(oracle["partIndex"]) is int
        and type(oracle["partCount"]) is int
        and 0 <= oracle["partIndex"] < oracle["partCount"]
        and type(oracle["roleId"]) is int
        and oracle["roleId"] in ROLE_TO_ID.values()
        and type(oracle["terminal"]) is bool
        and oracle["terminal"] == (oracle["partIndex"] + 1 == oracle["partCount"])
        and oracle["tokenSequenceDigest"] == stable_algorithm_hash(ids),
        "retained prepared token positions, byte edges or masks are malformed",
    )
    supports = oracle["supports"]
    _require(
        type(supports) is dict and set(supports) == {"length", "mode", "terminal", "modality"},
        "retained public supports are malformed",
    )
    for runs in supports.values():
        _require(
            type(runs) is list
            and bool(runs)
            and all(
                type(run) is list
                and len(run) == 2
                and all(type(x) is int for x in run)
                and 0 <= run[0] < run[1]
                for run in runs
            )
            and all(left[1] < right[0] for left, right in zip(runs[:-1], runs[1:], strict=True)),
            "retained public support is noncanonical or duplicated",
        )
    _require(
        supports["mode"] == [[0, 1]]
        and supports["terminal"] == [[0, 2]]
        and len(supports["length"]) == 1
        and supports["length"][0][0] == 1
        and len(ids) < supports["length"][0][1]
        and supports["modality"][0][0] == 0
        and supports["modality"][-1][1] <= 4,
        "retained semantic categorical supports changed",
    )
    readouts = record["readouts"]
    _require(
        type(readouts) is dict and set(readouts) == {"prior", "reconstruction"},
        "retained decoder passes are missing",
    )
    expected_positions = [index for index, eligible in enumerate(mask) if eligible]
    for branch, row in readouts.items():
        _require(
            type(row) is dict
            and set(row)
            == {
                "branch",
                "tokenSequenceDigest",
                "capacity",
                "positions",
                "labels",
                "mask",
                "alphabetSha256",
                "decisions",
            }
            and row["branch"] == branch
            and row["tokenSequenceDigest"] == oracle["tokenSequenceDigest"]
            and type(row["capacity"]) is int
            and row["capacity"] == oracle["capacity"]
            and type(row["positions"]) is list
            and all(type(x) is int for x in row["positions"])
            and row["positions"] == expected_positions
            and type(row["labels"]) is list
            and all(type(x) is int for x in row["labels"])
            and row["labels"] == [ids[index] for index in expected_positions]
            and type(row["mask"]) is list
            and all(type(x) is bool for x in row["mask"])
            and row["mask"] == mask
            and row["alphabetSha256"] == (oracle["alphabetSha256"] if expected_positions else None),
            "retained executed token positions, labels or masks differ from the oracle",
        )
        expected_decisions = {
            name: {"label": label, "support": supports[name]}
            for name, label in (
                ("length", len(ids)),
                ("mode", 0),
                ("terminal", int(oracle["terminal"])),
            )
        }
        _require(
            stable_algorithm_hash(row["decisions"]) == stable_algorithm_hash(expected_decisions),
            "retained executed categorical decisions differ from the oracle",
        )
    _require(
        stable_algorithm_hash(record["modality"])
        == stable_algorithm_hash({"label": 0, "support": supports["modality"]}),
        "retained actual modality decision differs from the oracle",
    )


def empty_position_exposure(
    *, registry_digest, objective, starting_global_step=0, previous_stage=None
):
    _require(_digest(registry_digest), "registry digest is invalid")
    _require(
        type(starting_global_step) is int and starting_global_step >= 0, "starting step is invalid"
    )
    return _seal(
        {
            "schema": POSITION_SCHEMA,
            "registryDigest": registry_digest,
            "objective": objective,
            "previousDigest": None,
            "previousStage": previous_stage,
            "startingGlobalStep": starting_global_step,
            "lastGlobalStep": starting_global_step,
            "optimizerUpdates": 0,
            "sources": {},
            "unmeasuredObjectives": {},
        }
    )


def validate_position_exposure(state, *, registry_digest, objective):
    _require(
        type(state) is dict
        and set(state)
        == {
            "schema",
            "registryDigest",
            "objective",
            "previousDigest",
            "previousStage",
            "startingGlobalStep",
            "lastGlobalStep",
            "optimizerUpdates",
            "sources",
            "unmeasuredObjectives",
            "digest",
        }
        and state["schema"] == POSITION_SCHEMA
        and state["registryDigest"] == registry_digest
        and stable_algorithm_hash(state["objective"]) == stable_algorithm_hash(objective)
        and _seal(state) == state
        and all(
            type(state[name]) is int
            for name in ("startingGlobalStep", "lastGlobalStep", "optimizerUpdates")
        )
        and 0 <= state["startingGlobalStep"] <= state["lastGlobalStep"]
        and state["optimizerUpdates"] == state["lastGlobalStep"] - state["startingGlobalStep"]
        and (
            state["previousDigest"] is None
            if state["optimizerUpdates"] == 0
            else _digest(state["previousDigest"])
        )
        and type(state["sources"]) is dict
        and type(state["unmeasuredObjectives"]) is dict
        and all(
            type(name) is str and type(count) is int and count > 0
            for name, count in state["unmeasuredObjectives"].items()
        ),
        "checkpoint position history is malformed, unsealed or bound to another objective",
    )
    previous_stage = state["previousStage"]
    if previous_stage is not None:
        _require(type(previous_stage) is dict, "archived position stage is malformed")
        validate_position_exposure(
            previous_stage,
            registry_digest=registry_digest,
            objective=previous_stage.get("objective"),
        )
        _require(
            previous_stage["lastGlobalStep"] == state["startingGlobalStep"],
            "archived position stage does not meet the current stage boundary",
        )
    for source_id, source in state["sources"].items():
        _require(
            type(source_id) is str
            and bool(source_id)
            and type(source) is dict
            and set(source) == {"split", "logicalTargets"}
            and source["split"] == "train"
            and type(source["logicalTargets"]) is dict,
            "retained position source identity is invalid",
        )
        for logical_id, target in source["logicalTargets"].items():
            _require(
                type(logical_id) is str
                and bool(logical_id)
                and type(target) is dict
                and set(target) == {"records"}
                and type(target["records"]) is dict,
                "retained logical target identity is invalid",
            )
            for record_id, row in target["records"].items():
                _require(
                    type(record_id) is str
                    and bool(record_id)
                    and type(row) is dict
                    and set(row)
                    == {
                        "receipt",
                        "windowIndices",
                        "scoredOccurrences",
                        "optimizerUpdates",
                        "firstGlobalStep",
                        "lastGlobalStep",
                    }
                    and type(row["windowIndices"]) is list
                    and bool(row["windowIndices"])
                    and all(type(index) is int and index >= 0 for index in row["windowIndices"])
                    and row["windowIndices"] == sorted(set(row["windowIndices"]))
                    and all(
                        type(row[name]) is int
                        for name in (
                            "scoredOccurrences",
                            "optimizerUpdates",
                            "firstGlobalStep",
                            "lastGlobalStep",
                        )
                    )
                    and len(row["windowIndices"]) <= row["scoredOccurrences"]
                    and 1
                    <= row["optimizerUpdates"]
                    <= min(row["scoredOccurrences"], state["optimizerUpdates"])
                    and state["startingGlobalStep"]
                    < row["firstGlobalStep"]
                    <= row["lastGlobalStep"]
                    <= state["lastGlobalStep"],
                    "retained target position occurrences or step bounds are invalid",
                )
                _validate_target_receipt(row["receipt"], logical_id=logical_id, record_id=record_id)
    return state


def restore_position_exposure(
    payload, *, registry_digest, objective, global_step, phase_transition=False
):
    _require(type(global_step) is int and global_step >= 0, "restore step is invalid")
    if payload is None:
        return empty_position_exposure(
            registry_digest=registry_digest, objective=objective, starting_global_step=global_step
        )
    previous = payload.get("train_config", {}).get("semantic_v8_policy")
    saved = payload.get("semantic_v8_position_exposure")
    if previous is None:
        _require(
            phase_transition and saved is None,
            "position history can begin only at an explicit fresh stage",
        )
        return empty_position_exposure(
            registry_digest=registry_digest, objective=objective, starting_global_step=global_step
        )
    _require(saved is not None, "V8 checkpoint is missing its position-exposure history")
    if phase_transition:
        validate_position_exposure(
            saved,
            registry_digest=registry_digest,
            objective=saved.get("objective"),
        )
        _require(
            saved["lastGlobalStep"] == global_step,
            "archived position stage differs from checkpoint step",
        )
        return empty_position_exposure(
            registry_digest=registry_digest,
            objective=objective,
            starting_global_step=global_step,
            previous_stage=saved,
        )
    state = validate_position_exposure(saved, registry_digest=registry_digest, objective=objective)
    _require(
        state["lastGlobalStep"] == global_step, "position history differs from checkpoint step"
    )
    return state


def advance_position_exposure(state, exposures, *, registry_digest, objective, optimizer_step):
    validate_position_exposure(state, registry_digest=registry_digest, objective=objective)
    _require(
        type(optimizer_step) is int and optimizer_step == state["lastGlobalStep"] + 1,
        "position history may advance only with the next optimizer update",
    )
    updated = json.loads(json.dumps(state))
    updated.update(
        previousDigest=state["digest"],
        lastGlobalStep=optimizer_step,
        optimizerUpdates=state["optimizerUpdates"] + 1,
    )
    touched = set()
    for exposure in exposures:
        _require(
            type(exposure) is dict
            and set(exposure)
            == {
                "schema",
                "registryDigest",
                "objectiveDigest",
                "sourceId",
                "windowIndex",
                "records",
                "unmeasuredObjectives",
            }
            and exposure["schema"] == POSITION_SCHEMA
            and exposure["registryDigest"] == registry_digest
            and exposure["objectiveDigest"] == stable_algorithm_hash(objective)
            and type(exposure["sourceId"]) is str
            and bool(exposure["sourceId"])
            and type(exposure["windowIndex"]) is int
            and exposure["windowIndex"] >= 0
            and type(exposure["records"]) is list
            and type(exposure["unmeasuredObjectives"]) is dict,
            "distributed position receipts have changed source/objective bindings",
        )
        source_id = exposure["sourceId"]
        for receipt in exposure["records"]:
            logical_id, record_id = receipt.get("logicalTargetId"), receipt.get("physicalRecordId")
            _require(
                type(logical_id) is str and logical_id and type(record_id) is str and record_id,
                "position receipt target identity is missing",
            )
            _validate_target_receipt(receipt, logical_id=logical_id, record_id=record_id)
            source = updated["sources"].setdefault(
                source_id, {"split": "train", "logicalTargets": {}}
            )
            target = source["logicalTargets"].setdefault(logical_id, {"records": {}})
            row = target["records"].get(record_id)
            if row is None:
                row = target["records"][record_id] = {
                    "receipt": receipt,
                    "windowIndices": [],
                    "scoredOccurrences": 0,
                    "optimizerUpdates": 0,
                    "firstGlobalStep": optimizer_step,
                    "lastGlobalStep": optimizer_step,
                }
            _require(
                row["receipt"] == receipt,
                "same physical target changed its prepared or executed positions",
            )
            row["windowIndices"] = sorted(set(row["windowIndices"]) | {exposure["windowIndex"]})
            row["scoredOccurrences"] += 1
            row["lastGlobalStep"] = optimizer_step
            key = source_id, logical_id, record_id
            if key not in touched:
                row["optimizerUpdates"] += 1
                touched.add(key)
        for name, count in exposure["unmeasuredObjectives"].items():
            _require(
                type(name) is str and type(count) is int and count > 0,
                "unmeasured objective count is malformed",
            )
            updated["unmeasuredObjectives"][name] = (
                updated["unmeasuredObjectives"].get(name, 0) + count
            )
    return validate_position_exposure(
        _seal(updated), registry_digest=registry_digest, objective=objective
    )


def _position_counts(receipt, weights):
    counts = dict.fromkeys(POSITION_LOSSES, 0)
    if receipt["status"] == "verified":
        counts.update(
            lexical=len(receipt["readouts"]["prior"]["positions"]),
            reconstruction=len(receipt["readouts"]["reconstruction"]["positions"]),
            structure=4,
            terminal=2,
            modality=1,
        )
    return {name: count for name, count in counts.items() if weights.get(name, 0) > 0}


def position_exposure_report(state):
    counts = dict.fromkeys(POSITION_LOSSES, 0)
    verified = unmeasured = occurrences = logical = physical = 0
    for source in state["sources"].values():
        logical += len(source["logicalTargets"])
        for target in source["logicalTargets"].values():
            for row in target["records"].values():
                physical += 1
                verified += row["receipt"]["status"] == "verified"
                unmeasured += row["receipt"]["status"] == "unmeasured"
                occurrences += row["scoredOccurrences"]
                for name, count in _position_counts(
                    row["receipt"], state["objective"]["effectiveLossWeights"]
                ).items():
                    counts[name] += count * row["scoredOccurrences"]
    return {
        "schema": POSITION_SCHEMA,
        "digest": state["digest"],
        "startingGlobalStep": state["startingGlobalStep"],
        "lastGlobalStep": state["lastGlobalStep"],
        "optimizerUpdates": state["optimizerUpdates"],
        "previousStageDigest": None
        if state["previousStage"] is None
        else state["previousStage"]["digest"],
        "sourceCount": len(state["sources"]),
        "logicalTargetCount": logical,
        "physicalTargetRecordCount": physical,
        "verifiedPhysicalTargets": verified,
        "unmeasuredPhysicalTargets": unmeasured,
        "scoredOccurrences": occurrences,
        "positiveWeightPositionCounts": counts,
        "unmeasuredObjectiveCounts": state["unmeasuredObjectives"],
        "positionScope": "Actual plain-text lexical/reconstruction rows and length/mode/terminal/modality categorical labels and public supports",
        "routerInternalPositionCoverage": "unmeasured",
        "otherObjectiveScope": "Routing regularizers, representation coordinates, media/structured-action masks and transducer/retrieval losses are not granted token-position credit by this audit",
        "learnedSuccess": None,
    }


def independent_position_inventory(
    episodes,
    *,
    codec,
    config,
    model_config,
    modalities,
    prepared_manifest_fingerprint,
    expected_source_ids,
):
    """Enumerate physical prepared policy records without windows or loss plans.

    One complete corpus pass requires each distinct physical target exactly once.
    This independent multiplicity also catches a planner and execution path that
    agree with each other while both repeating or dropping the same target.
    """
    registry = config.semantic_v8_policy["registryDigest"]
    objective = position_objective(config, model_config, codec, modalities)
    sources = {}
    for episode in episodes:
        source_id = inventory_episode_source_id(episode, registry_digest=registry)
        _require(source_id not in sources, "independent corpus contains a duplicate source")
        source = sources[source_id] = {"split": "train", "logicalTargets": {}}
        for identity, target in independent_episode_targets(
            episode,
            codec=codec,
            config=config,
            model_config=model_config,
            modalities=modalities,
        ).items():
            logical = source["logicalTargets"].setdefault(
                target["logicalTargetId"], {"records": {}}
            )
            logical["records"][identity] = target
    _require(
        sorted(sources) == sorted(expected_source_ids)
        and len(expected_source_ids) == len(set(expected_source_ids)),
        "independent prepared source population differs from admitted training split",
    )
    records = [
        row
        for source in sources.values()
        for target in source["logicalTargets"].values()
        for row in target["records"].values()
    ]
    _require(bool(records), "independent prepared corpus contains no policy targets")
    position_counts = dict.fromkeys(POSITION_LOSSES, 0)
    for target in records:
        if target["status"] == "verified":
            count = sum(target["oracle"]["freeMask"])
            for name, value in {
                "lexical": count,
                "reconstruction": count,
                "structure": 4,
                "terminal": 2,
                "modality": 1,
            }.items():
                if objective["effectiveLossWeights"].get(name, 0) > 0:
                    position_counts[name] += value
    return _seal(
        {
            "schema": "semantic-v8-independent-prepared-positions/1",
            "registryDigest": registry,
            "preparedManifestFingerprint": prepared_manifest_fingerprint,
            "objective": objective,
            "sourceIds": sorted(sources),
            "sourceCount": len(sources),
            "logicalTargetCount": sum(len(source["logicalTargets"]) for source in sources.values()),
            "physicalTargetRecordCount": len(records),
            "verifiedPhysicalTargets": sum(row["status"] == "verified" for row in records),
            "unmeasuredPhysicalTargets": sum(row["status"] == "unmeasured" for row in records),
            "positiveWeightPositionCounts": position_counts,
            "sources": sources,
            "independentOf": ["target planner", "window iterator", "decoder", "optimizer receipts"],
        }
    )


def _flat_position_rows(sources):
    return {
        (source_id, logical_id, record_id): record
        for source_id, source in sources.items()
        for logical_id, target in source["logicalTargets"].items()
        for record_id, record in target["records"].items()
    }


def compare_position_inventory(inventory, state, *, target_exposure, global_step):
    """Compare actual receipts with independently re-read prepared physical truth.

    The caller constructs ``inventory`` afresh from the admitted prepared corpus;
    it must never substitute an inventory copied from an optimizer checkpoint.
    """
    _require(
        inventory["schema"] == "semantic-v8-independent-prepared-positions/1"
        and _seal(inventory) == inventory,
        "independent position inventory is malformed or unsealed",
    )
    validate_position_exposure(
        state,
        registry_digest=inventory["registryDigest"],
        objective=inventory["objective"],
    )
    _require(
        type(global_step) is int
        and global_step >= 1
        and state["startingGlobalStep"] == target_exposure["startingGlobalStep"] == 0
        and state["previousStage"] is None
        and state["lastGlobalStep"] == target_exposure["lastGlobalStep"] == global_step
        and state["optimizerUpdates"] == target_exposure["optimizerUpdates"] == global_step
        and target_exposure["registryDigest"] == inventory["registryDigest"],
        "position and target histories do not bind the same fresh optimizer sequence",
    )
    expected = _flat_position_rows(inventory["sources"])
    actual = _flat_position_rows(state["sources"])
    presence = _flat_position_rows(target_exposure["sources"])
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    oracle_mismatches = []
    occurrence_mismatches = []
    target_census_mismatches = []
    unmeasured = []
    for key in sorted(expected.keys() | actual.keys()):
        if key in expected and expected[key]["status"] != "verified":
            unmeasured.append(key)
        if key not in actual or key not in expected:
            continue
        row = actual[key]
        receipt = row["receipt"]
        actual_oracle = {field: receipt[field] for field in expected[key]}
        if stable_algorithm_hash(actual_oracle) != stable_algorithm_hash(expected[key]):
            oracle_mismatches.append(key)
        if receipt["status"] != "verified" and key not in unmeasured:
            unmeasured.append(key)
        # The expected multiplicity is physical corpus membership, not a count
        # supplied by the planner or by the actual decoder.
        if row["scoredOccurrences"] != 1:
            occurrence_mismatches.append([list(key), 1, row["scoredOccurrences"]])
        target_row = presence.get(key)
        if target_row is None or any(
            row[field] != target_row[field] for field in ("windowIndices", "scoredOccurrences")
        ):
            target_census_mismatches.append(key)
    target_census_extra = sorted(presence.keys() - actual.keys())
    summary = position_exposure_report(state)
    count_match = (
        summary["positiveWeightPositionCounts"] == inventory["positiveWeightPositionCounts"]
    )
    complete = (
        not any(
            (
                missing,
                unexpected,
                oracle_mismatches,
                occurrence_mismatches,
                target_census_mismatches,
                target_census_extra,
                unmeasured,
            )
        )
        and count_match
    )
    return {
        "independentPreparedPhysicalTargets": len(expected),
        "observedPhysicalTargets": len(actual),
        "missingPhysicalTargets": [list(key) for key in missing],
        "unexpectedPhysicalTargets": [list(key) for key in unexpected],
        "preparedOracleMismatches": [list(key) for key in oracle_mismatches],
        "independentOccurrenceMismatches": occurrence_mismatches,
        "targetCensusWindowOrOccurrenceMismatches": [list(key) for key in target_census_mismatches],
        "targetCensusWithoutPositionReceipt": [list(key) for key in target_census_extra],
        "unmeasuredPhysicalTargets": [list(key) for key in unmeasured],
        "positiveWeightPositionCountsMatch": count_match,
        "exactOnePassPreparedTextPositions": complete,
        "routerInternalPositionCoverage": "unmeasured",
        "learnedSuccess": None,
    }


def qualify_position_history(first, resumed, *, inventory, first_metadata, resumed_metadata):
    """Verify bounded step-one/resume history and its observed prepared-text subset."""
    from types import SimpleNamespace

    from megatoken.model import ModelConfig
    from megatoken.training.config import TrainConfig

    states = []
    for payload, metadata, step in ((first, first_metadata, 1), (resumed, resumed_metadata, 2)):
        identity = position_objective(
            TrainConfig.from_dict(payload["train_config"]),
            ModelConfig.from_dict(payload["model_config"]),
            SimpleNamespace(fingerprint=payload["codec_fingerprint"]),
            inventory["objective"]["installedModalities"],
        )
        _require(
            stable_algorithm_hash(identity) == stable_algorithm_hash(inventory["objective"]),
            "lifecycle checkpoint changed the independently inventoried position objective",
        )
        state = validate_position_exposure(
            payload.get("semantic_v8_position_exposure"),
            registry_digest=inventory["registryDigest"],
            objective=identity,
        )
        _require(
            state["startingGlobalStep"] == 0
            and state["lastGlobalStep"]
            == state["optimizerUpdates"]
            == payload["global_step"]
            == step
            and metadata.get("semantic_v8_position_digest") == state["digest"],
            "lifecycle checkpoint metadata or steps do not bind the actual position history",
        )
        states.append(state)
    before, after = states
    _require(
        after["previousDigest"] == before["digest"], "resumed position history changed its parent"
    )
    _require(
        all(
            after["unmeasuredObjectives"].get(name, 0) >= count
            for name, count in before["unmeasuredObjectives"].items()
        ),
        "resumed position history dropped unmeasured objective events",
    )
    previous_rows, resumed_rows = (_flat_position_rows(state["sources"]) for state in states)
    for key, previous in previous_rows.items():
        current = resumed_rows.get(key)
        _require(
            current is not None
            and stable_algorithm_hash(current["receipt"])
            == stable_algorithm_hash(previous["receipt"])
            and set(previous["windowIndices"]) <= set(current["windowIndices"])
            and current["scoredOccurrences"] >= previous["scoredOccurrences"]
            and current["optimizerUpdates"] >= previous["optimizerUpdates"]
            and current["firstGlobalStep"] == previous["firstGlobalStep"]
            and current["lastGlobalStep"] >= previous["lastGlobalStep"],
            "resumed position history dropped or changed an earlier actual receipt",
        )
    comparisons = [
        compare_position_inventory(
            inventory,
            state,
            target_exposure=payload["semantic_v8_target_exposure"],
            global_step=step,
        )
        for state, payload, step in ((before, first, 1), (after, resumed, 2))
    ]
    for comparison in comparisons:
        _require(
            not any(
                comparison[name]
                for name in (
                    "unexpectedPhysicalTargets",
                    "preparedOracleMismatches",
                    "targetCensusWindowOrOccurrenceMismatches",
                    "targetCensusWithoutPositionReceipt",
                )
            ),
            "lifecycle actual position subset differs from independently read prepared data",
        )
    return {
        "registryDigest": inventory["registryDigest"],
        "historyContinued": True,
        "firstCheckpointDigest": before["digest"],
        "resumedCheckpointDigest": after["digest"],
        "firstCheckpointCounts": position_exposure_report(before),
        "resumedCounts": position_exposure_report(after),
        "independentPreparedPositionInventoryDigest": inventory["digest"],
        "independentPreparedPhysicalTargetCount": inventory["physicalTargetRecordCount"],
        "fullPopulationComparison": comparisons[-1],
        "fullPopulationQualified": False,
        "scope": "Only the actual two-update prepared-text subset and its exact resume history; a dedicated fresh complete pass is required for full qualification",
        "routerInternalPositionCoverage": "unmeasured",
        "learnedSuccess": None,
    }
