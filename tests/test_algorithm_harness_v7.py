"""Native causal admission tests using fresh semantic/7 producer artifacts."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatoken.data.algorithm_context_v7 import (
    AlgorithmContextError,
    restore_algorithm_continuation_v7,
    stable_algorithm_hash,
)
from megatoken.data.algorithm_harness_v7 import AlgorithmHarnessError, adapt_algorithm_harness_v7
from megatoken.algorithm_model_adapter_v7 import AlgorithmModelAdapterV7
from megatoken.controls import build_request_control
from megatoken.data.algorithm_context_v7 import algorithm_execution_metadata, algorithm_selection_instruction


ROOT = Path(__file__).resolve().parents[2]


JS_ARTIFACTS = r"""
import {AlgorithmExecutionHarness} from './tracing-js-self/src/semantic/harness.js';
import {evaluateTerm} from './tracing-js-self/src/semantic/operations.js';
import {decodeAlgorithmRecord} from './tracing-js-self/src/semantic/trace-codec.js';
import {hashAlgorithmData} from './tracing-js-self/src/semantic/program.js';

async function make({source,entry='run',args=[],readValue=7,rejectRead=false,alternate=false}){
  const preliminary = new AlgorithmExecutionHarness({source,entry,args,
    capabilities:['demo.readAsync','demo.store']});
  const capabilities = [...new Set(preliminary.program.boundaries.map(row=>row.operation))];
  const harness = new AlgorithmExecutionHarness({source,entry,args,capabilities});
  const contexts=[];
  const emit=record=>{
    contexts.push(harness.modelContext());
    const result=harness.select(record);
    if(!result.claim.consistent)throw new Error('fixture emitted a false claim');
    return result;
  };
  if(alternate){
    const step=harness.controller.nextStep(),term=step.term;
    const left=term.args[0].args[0],right=term.args[0].args[1];
    emit({kind:'compute',id:'extraLeft',term:left,result:['num',3]});
    emit({kind:'compute',id:'extraRight',term:right,result:['num',7]});
    const decomposition={op:'add',args:[
      {op:'multiply',args:[{op:'ref',id:'extraLeft'},{op:'ref',id:'extraRight'}]},
      term.args[1]
    ]};
    emit({kind:'compute',id:step.id,term:decomposition,result:['num',22]});
  }
  while(!harness.controller.completion){
    const step=harness.controller.nextStep();
    const environment=harness.controller.environment();
    if(step.kind==='compute')emit({...step,result:evaluateTerm(step.term,environment)});
    else if(step.kind==='request'){
      const selected=emit({...step,arguments:step.arguments.map(term=>evaluateTerm(term,environment))});
      await harness.dispatch(selected.dispatch.id,async(_args,context)=>{
        if(context.operation==='demo.readAsync'){
          if(rejectRead)throw 'denied';
          return readValue;
        }
        return undefined;
      });
    }else if(step.kind==='return'||step.kind==='throw')
      emit({kind:step.kind,value:evaluateTerm(step.value,environment)});
    else throw new Error('unhandled fixture step '+step.kind);
  }
  const publicEpisode=harness.exportEpisode();
  const certificates=[];
  for(let index=0;index<harness.controller.history.length;index++){
    const row=harness.controller.history[index];
    if(row.mode!=='generated')continue;
    const record=decodeAlgorithmRecord(row.line);
    certificates.push({record:index,
      rule:record.kind==='request'?'certified-effect-boundary/1':'checked-algorithm-operation/1',
      digest:hashAlgorithmData(row.line),sourceDigest:harness.program.sourceDigest,
      definitionDigest:harness.program.definitionDigest});
  }
  const sourceArtifact={public:{schema:'semantic-execution/6',timeline:[]},
    sidecar:{schema:'semantic-supervision/2',identityDigest:'b'.repeat(64),
      sourceFamily:'algorithm-test-family',reference:{sourceChecks:{status:'agreed'},
        outcome:{kind:'return'}}}};
  const sidecar={schema:'semantic-algorithm-evidence/2',admission:'locally_verified',
    referenceStatus:'agreed',selectionOrigin:'owned-interpreter-teacher',
    profile:'algorithm-execution/1',representation:'semantic-trace/7',
    sourceDigest:harness.program.sourceDigest,
    coreDigest:harness.program.coreDigest,definitionDigest:harness.program.definitionDigest,
    sourceArtifact,sourceArtifactDigest:hashAlgorithmData(sourceArtifact),
    sourceEvidence:{identityDigest:'b'.repeat(64),sourceChecks:{status:'agreed'},
      referenceOutcome:{kind:'return'}},
    archiveDigest:hashAlgorithmData(harness.program.source+'\narchive'),
    family:'algorithm-test-family',sourceFamily:'algorithm-test-family',
    fixtureDigest:'a'.repeat(64),provenance:{},receipt:{},certificates,
    publicDigest:hashAlgorithmData(publicEpisode),contextDigest:hashAlgorithmData(contexts),
    nativeAdmission:null,optimizerSteps:0,learningClaim:null};
  return {public:publicEpisode,contexts,sidecar};
}

const arithmetic={source:'"use strict";function run(a,b){return a+b;}',args:[2,3]};
const alternate={source:'"use strict";function run(a,b,c,d){return (a+b)*(c+d)+1;}',args:[1,2,3,4]};
const counter={source:'"use strict";async function run(){const x=await host.readAsync("counter");return x+1;}',args:[]};
const badFinally='"use strict";async function run(){const x=await host.readAsync("counter");return x+1;}';
const values={
  arithmetic:await make(arithmetic),
  alternate:await make({...alternate,alternate:true}),
  counter7:await make({...counter,readValue:7}),
  counter99:await make({...counter,readValue:99}),
  rejected:await make({...counter,rejectRead:true}),
  checkpoint:undefined,
};
const pendingProgram=new AlgorithmExecutionHarness({source:counter.source,entry:'run',args:[],capabilities:['demo.readAsync','demo.store']});
const pendingCapabilities=[...new Set(pendingProgram.program.boundaries.map(row=>row.operation))];
const pending=new AlgorithmExecutionHarness({source:counter.source,entry:'run',args:[],capabilities:pendingCapabilities});
const first=pending.controller.nextStep();
const selected=pending.select({...first,arguments:first.arguments.map(term=>evaluateTerm(term,pending.controller.environment()))});
values.checkpoint=pending.controller.checkpoint();
console.log(JSON.stringify(values));
"""


@pytest.fixture(scope="module")
def artifacts():
    if not (ROOT / "tracing-js-self" / "src" / "semantic" / "harness.js").is_file():
        pytest.skip("Fresh semantic/7 JavaScript producer is unavailable")
    result = subprocess.run(
        ["node", "--input-type=module", "-e", JS_ARTIFACTS],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _clone(value):
    return copy.deepcopy(value)


def _rehash(export):
    export["sidecar"]["publicDigest"] = stable_algorithm_hash(export["public"])
    export["sidecar"]["contextDigest"] = stable_algorithm_hash(export["contexts"])


def _refresh_certificate_for_first_generated(export):
    generated = next(row for row in export["public"]["timeline"] if row.get("kind") == "semantic")
    certificate = export["sidecar"]["certificates"][0]
    certificate["digest"] = stable_algorithm_hash(generated["line"])


def test_fresh_js_artifact_becomes_a_native_episode_with_loss_only_on_generated_records(artifacts):
    export = artifacts["arithmetic"]
    episode, receipt = adapt_algorithm_harness_v7(export, allow_local=True)

    assert receipt.generated == len(export["contexts"])
    assert receipt.completion == export["public"]["completion"]
    assert receipt.records > len(export["public"]["timeline"])
    targets = [record for record in episode.records if record.policy_mask]
    assert len(targets) == receipt.generated
    assert all(record.role == "assistant" and record.provenance == "unknown" for record in targets)
    assert all(record.role != "assistant" for record in episode.records if not record.policy_mask)
    assert all(record.payload != json.dumps(export["sidecar"]).encode() for record in episode.records)


def test_alternate_truthful_subterm_claims_are_replayed_and_retained_as_live_context(artifacts):
    export = artifacts["alternate"]
    episode, _ = adapt_algorithm_harness_v7(export, allow_local=True)

    generated = [record.text for record in episode.records if record.policy_mask]
    assert any("extraLeft" in line for line in generated)
    assert any("extraRight" in line for line in generated)
    assert any("%extraLeft" in line and "%extraRight" in line for line in generated)
    assert [row[0] for row in export["contexts"][1]["state"]["symbols"]] == ["extraLeft"]
    assert [row[0] for row in export["contexts"][2]["state"]["symbols"]] == [
        "extraLeft",
        "extraRight",
    ]


def test_external_observations_are_unmasked_environment_results_and_counterfactual_future_isolation(artifacts):
    low, high = artifacts["counter7"], artifacts["counter99"]
    low_timeline, high_timeline = low["public"]["timeline"], high["public"]["timeline"]
    first_difference = next(i for i, (a, b) in enumerate(zip(low_timeline, high_timeline)) if a != b)
    assert low_timeline[first_difference]["kind"] == high_timeline[first_difference]["kind"] == "observation"
    assert low_timeline[first_difference]["line"] != high_timeline[first_difference]["line"]
    assert low["contexts"][:1] == high["contexts"][:1]

    episode, receipt = adapt_algorithm_harness_v7(low, allow_local=True)
    assert receipt.actions == 1
    observations = [record for record in episode.records if record.result_of]
    assert len(observations) == 1
    assert observations[0].role == "environment" and not observations[0].policy_mask
    dispatch_rows = [row for row in low_timeline if row["kind"] == "dispatch"]
    assert len(dispatch_rows) == 1
    assert all(record.role == "environment" and not record.policy_mask for record in episode.records
               if record.record_id.startswith("algorithm:v7:observation:"))


def test_rejected_external_observation_is_not_relabelled_as_generated_and_still_restores_completion(artifacts):
    export = artifacts["rejected"]
    episode, receipt = adapt_algorithm_harness_v7(export, allow_local=True)
    observed = [row for row in export["public"]["timeline"] if row["kind"] == "observation"]
    assert len(observed) == 1 and observed[0]["mode"] == "observed"
    assert receipt.completion["kind"] == "throw"
    assert any(not record.policy_mask and record.role == "environment" and record.result_of
               for record in episode.records)


def test_exact_current_prefix_context_and_consumer_definition_cache_are_mandatory(artifacts):
    valid = _clone(artifacts["counter7"])
    adapt_algorithm_harness_v7(valid, allow_local=True)

    stale_cutoff = _clone(valid)
    stale_cutoff["contexts"][1]["visibleThrough"] += 1
    _rehash(stale_cutoff)
    with pytest.raises(AlgorithmHarnessError, match="cutoff|context"):
        adapt_algorithm_harness_v7(stale_cutoff, allow_local=True)

    cache_tamper = _clone(valid)
    cache_tamper["contexts"][0]["program"]["core"]["steps"][0]["id"] = "forged"
    _rehash(cache_tamper)
    with pytest.raises(AlgorithmHarnessError, match="context|definition|publication"):
        adapt_algorithm_harness_v7(cache_tamper, allow_local=True)

    republish = _clone(valid)
    republish["contexts"][1]["program"] = _clone(republish["contexts"][0]["program"])
    _rehash(republish)
    with pytest.raises(AlgorithmHarnessError, match="republish|context"):
        adapt_algorithm_harness_v7(republish, allow_local=True)


def test_old_discriminator_changed_and_future_records_are_rejected_even_when_public_hashes_are_refreshed(artifacts):
    old = _clone(artifacts["arithmetic"])
    del old["public"]["representation"]
    _rehash(old)
    with pytest.raises(AlgorithmHarnessError, match="fields|explicit"):
        adapt_algorithm_harness_v7(old, allow_local=True)

    changed = _clone(artifacts["arithmetic"])
    first = next(row for row in changed["public"]["timeline"] if row["kind"] == "semantic")
    first["line"] = first["line"].rsplit(" => ", 1)[0] + " => 123"
    changed["public"]["firstViolation"] = {"index": 0, "kind": "compute"}
    _refresh_certificate_for_first_generated(changed)
    _rehash(changed)
    with pytest.raises(AlgorithmHarnessError, match="context|false|claim|selection"):
        adapt_algorithm_harness_v7(changed, allow_local=True)

    wrong_role = _clone(artifacts["counter7"])
    observation = next(row for row in wrong_role["public"]["timeline"] if row["kind"] == "observation")
    observation["mode"] = "generated"
    _rehash(wrong_role)
    with pytest.raises(AlgorithmHarnessError, match="observed|observation|authority"):
        adapt_algorithm_harness_v7(wrong_role, allow_local=True)

    future = _clone(artifacts["arithmetic"])
    future["public"]["timeline"].append(_clone(future["public"]["timeline"][-1]))
    _rehash(future)
    with pytest.raises(AlgorithmHarnessError):
        adapt_algorithm_harness_v7(future, allow_local=True)


def test_current_context_from_a_different_future_response_is_rejected(artifacts):
    left, right = _clone(artifacts["counter7"]), _clone(artifacts["counter99"])
    left["contexts"][2] = _clone(right["contexts"][2])
    _rehash(left)
    with pytest.raises(AlgorithmHarnessError, match="prefix|state|context"):
        adapt_algorithm_harness_v7(left, allow_local=True)


def test_continuation_restore_replays_public_prefix_without_dispatch(artifacts):
    restored = restore_algorithm_continuation_v7(artifacts["checkpoint"])
    assert restored.pending["id"] == "r0"
    assert restored.history[0]["mode"] == "generated"

    packet = _clone(artifacts["checkpoint"])
    packet["state"]["pending"]["payload"] = "[]"
    body = {key: value for key, value in packet.items() if key != "integrity"}
    packet["integrity"] = stable_algorithm_hash(body)
    with pytest.raises(AlgorithmContextError, match="state|prefix"):
        restore_algorithm_continuation_v7(packet)


class _FakeNativeRuntime:
    """Exercise the actual native serving call contract without fixture answer repair."""

    def __init__(self, pieces=()):
        self.generator = SimpleNamespace(
            generator=SimpleNamespace(config=SimpleNamespace(max_capacity=128))
        )
        self.pieces = list(pieces)
        self.sessions = []
        self.calls = []

    def open(self, **kwargs):
        assert kwargs["metadata"] == algorithm_execution_metadata()
        assert kwargs["memory_writes"] is False
        session = SimpleNamespace(
            context_archive=SimpleNamespace(events={}),
            visible_events=set(),
            observations=[],
        )
        self.sessions.append(session)
        return session

    def observe_text(self, session, track, payload, *, role, **kwargs):
        assert track == "assistant"
        identity = f"event{len(session.observations)}"
        session.observations.append((identity, role, payload))
        session.context_archive.events[identity] = payload
        session.visible_events.add(identity)
        return 1

    def context_status(self, session, *, track):
        assert track == "assistant"
        return {"visible_events": tuple(session.visible_events)}

    def generate(self, session, *, tracks, specs, controls, effort, continuation_mode):
        assert continuation_mode == "encoded_text"
        assert len(tracks) == len(specs) == len(controls) == 1
        assert specs[0].text_kind == "utf8"
        assert controls[0] in {
            build_request_control(instruction=algorithm_selection_instruction()),
            build_request_control(instruction=algorithm_selection_instruction(continuation=True)),
        }
        self.calls.append(controls[0])
        payload, terminal = self.pieces.pop(0)
        identity = f"event{len(session.observations)}"
        session.observations.append((identity, "assistant", payload))
        session.context_archive.events[identity] = payload
        session.visible_events.add(identity)
        return SimpleNamespace(
            tape=SimpleNamespace(
                valid=(True,),
                modalities=("text",),
                tracks=tracks,
                event_ids=(identity,),
                payloads=(payload,),
                terminal=(terminal,),
            )
        )


def test_native_serving_uses_matching_context_controls_continuation_and_rejects_stale_definitions(artifacts):
    export = artifacts["arithmetic"]
    lines = [row["line"] for row in export["public"]["timeline"] if row.get("kind") == "semantic"]
    split = len(lines[0].encode()) // 2
    pieces = [(lines[0].encode()[:split], False), (lines[0].encode()[split:], True), (lines[1].encode(), True)]
    runtime = _FakeNativeRuntime(pieces)
    adapter = AlgorithmModelAdapterV7(runtime, verify_context=True)
    adapter.open(
        "serving1",
        export["public"]["timeline"][0]["program"],
        export["public"]["timeline"][1]["graph"],
        invocation_id=export["contexts"][0]["invocationId"],
    )

    first = adapter.select("serving1")
    assert first["line"] == lines[0]
    assert len(runtime.calls) == 2
    assert runtime.calls == [
        build_request_control(instruction=algorithm_selection_instruction()),
        build_request_control(instruction=algorithm_selection_instruction(continuation=True)),
    ]
    stale = first["context"]
    stale["state"]["cursor"] = 99
    with pytest.raises(Exception, match="stale|altered"):
        adapter.select("serving1", context=stale)
    second = adapter.select("serving1")
    assert second["line"] == lines[1]
    episode, _ = adapt_algorithm_harness_v7(export, allow_local=True)
    target_events = [event for group in episode.meridians for event in group.events
                     if episode.records[next(i for i, record in enumerate(episode.records)
                                               if record.record_id == event.record_id)].policy_mask]
    assert target_events[0].instruction == algorithm_selection_instruction()


def test_native_serving_restore_rebuilds_context_without_dispatching(artifacts):
    export = artifacts["counter7"]
    context = export["contexts"][0]
    request_line = next(row["line"] for row in export["public"]["timeline"]
                        if row.get("kind") == "semantic")
    runtime = _FakeNativeRuntime([(request_line.encode(), True)])
    adapter = AlgorithmModelAdapterV7(runtime, verify_context=True)
    adapter.open(
        "serve-pending",
        export["public"]["timeline"][0]["program"],
        export["public"]["timeline"][1]["graph"],
        invocation_id=context["invocationId"],
    )
    selected = adapter.select("serve-pending")
    checkpoint = adapter.checkpoint("serve-pending")
    assert selected["dispatch"]["id"] == "r0"

    resumed_runtime = _FakeNativeRuntime()
    resumed = AlgorithmModelAdapterV7(resumed_runtime, verify_context=True)
    restored = resumed.restore("serve-restored", checkpoint, contexts=[context])
    assert restored["pending"]["id"] == "r0"
    assert resumed_runtime.calls == []
    assert not any(row["kind"] == "dispatch" for row in resumed.sessions["serve-restored"].timeline)

