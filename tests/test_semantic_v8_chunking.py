"""Physical observations use the retained training policy, not model capacity."""

import pytest

from megatoken.contracts import ROLE_TO_ID
from megatoken.data.chunking import ChunkingConfig
from megatoken.data.native_v8 import prepare_native_v8_training
from megatoken.semantic_model_v8 import NativeSemanticInferenceV8
from tests.test_prediction_prefix_v8 import examples
from tests.test_semantic_model_v8 import FakeRuntime
from tests.test_semantic_v8_serving_restore import select, target


@pytest.mark.parametrize("physical_limit", [128, 256])
def test_observation_bytes_roles_events_and_fragment_budget_match_prepared_records(physical_limit):
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    runtime = FakeRuntime([(target(example, 0), True)])
    runtime.generator.generator.config.max_capacity = 512
    runtime.chunking = ChunkingConfig().with_hard_limit(physical_limit)
    adapter = NativeSemanticInferenceV8(runtime, model_identity="a" * 64, max_tokens=512)
    assert adapter.max_tokens == physical_limit
    adapter.initialize(
        seed=17, checkpointIdentity="a" * 64, tokenizerIdentity=runtime.codec.fingerprint
    )
    select(adapter, example, 0)
    episode, _ = prepare_native_v8_training(
        example, codec=runtime.codec, max_positions=physical_limit, max_history=128
    )
    expected = []
    for row in episode.records:
        if row.policy_mask:
            break
        expected.append(
            (row.payload, ROLE_TO_ID[row.role], row.record_id, row.part_index + 1 == row.part_count)
        )
    assert [
        (row.payload, row.role_id, row.event_id, row.terminal) for row in runtime.observations
    ] == expected
