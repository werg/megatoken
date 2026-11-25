from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import base64
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys

root=Path(__file__).parent
producer,consumer=root/'producer-dev',root/'consumer'
def read(name):
 return json.loads((root/name).read_bytes())
def git(repo,*args):
 return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
def clean_revision(repo):
 assert not git(repo,'status','--porcelain','--untracked-files=all')
 return git(repo,'rev-parse','HEAD')
p_revision,c_revision=clean_revision(producer),clean_revision(consumer)
canonical=read('finish-canonical-status.json')
canonical_recheck=read('finish-canonical-recheck-status.json')
cohort_export=read('finish-cohort-export-status.json')
producer_tests=read('finish-producer-status.json')
native=read('finish-native-status.json')
paired_native=read('finish-paired-native-status.json')
cohort_tests=read('finish-cohort-status.json')
preparation_status=read('finish-preparation-status.json')
support_status=read('finish-support-status.json')
python_style_status=read('finish-python-style-status.json')
sync=read('finish-canonical-sync.json')
cohort_sync=read('finish-cohort-sync.json')
assert len(canonical['steps'])==10 and all(step['exitCode']==0 for step in canonical['steps'])
canonical_commands=[
 ['node','tools/generate-semantic-region-contract.mjs','--definition-only'],
 ['node','tools/generate-protocol-machine-fixtures.mjs'],
 ['node','tools/generate-protocol-iterator-fixtures.mjs'],
 ['node','tools/generate-protocol-generator-fixtures.mjs'],
 ['node','tools/generate-protocol-iterator-source-fixtures.mjs'],
 ['node','tools/generate-protocol-host-fixtures.mjs'],
 ['node','tools/generate-primitive-conversion-fixtures.mjs'],
 ['node','tools/generate-protocol-source-fixtures.mjs'],
 ['node','tools/generate-semantic-region-contract.mjs'],
 ['npm','run','contracts:check'],
]
assert [step['command'] for step in canonical['steps']]==canonical_commands
assert git(producer,'merge-base','--is-ancestor',canonical['producerRevision'],p_revision)==''
assert canonical_recheck['producerRevision']==p_revision
assert canonical_recheck['generationSourceRevision']==canonical['implementationRevision']
assert canonical_recheck['generatedArtifactRevision']==canonical['producerRevision']
assert canonical_recheck['generatedDataUnchanged'] is True
if canonical_recheck['mode']=='verified-generation-artifact':
 assert p_revision==canonical['producerRevision']
 assert canonical_recheck['steps']==[]
 assert canonical_recheck['changesSinceGeneration']==[]
 generated_paths=git(producer,'diff','--name-only',canonical['implementationRevision'],canonical['producerRevision']).splitlines()
 generated_contracts={'contracts/semantic-execution-v8.json','contracts/semantic-native-policy-v8.json'}
 assert canonical_recheck['generatedPaths']==generated_paths
 assert all(path in generated_contracts or path.startswith('test/fixtures/semantic-') for path in generated_paths)
 evidence=canonical_recheck['evidenceLogs']
 assert [row['stepIndex'] for row in evidence]==[7,9]
 assert [row['command'] for row in evidence]==[
  canonical['steps'][7]['command'],canonical['steps'][9]['command']]
 assert all(row['valid'] is True for row in evidence)
 assert [row['mode'] for row in evidence]==['generated','checked']
 for row,step in zip(evidence,(canonical['steps'][7],canonical['steps'][9]),strict=True):
  path=Path(row['log'])
  assert path==Path(step['log']) and path.is_file()
  assert row['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
 source_evidence_log=Path(evidence[0]['log'])
 gate_evidence_log=Path(evidence[1]['log'])
elif canonical_recheck['mode']=='fresh-recheck':
 assert not git(producer,'diff','--name-only',canonical['producerRevision'],p_revision,'--','contracts','test/fixtures')
 assert len(canonical_recheck['steps'])==2
 assert [step['command'] for step in canonical_recheck['steps']]==[
  ['node','tools/generate-protocol-source-fixtures.mjs','--check'],
  ['npm','run','contracts:check'],
 ]
 assert all(step['exitCode']==0 and step['producerTreeUnchanged'] is True for step in canonical_recheck['steps'])
 source_evidence_log=root/'finish-canonical-recheck-00.log'
 gate_evidence_log=root/'finish-canonical-recheck-01.log'
else:
 raise AssertionError(f"unknown canonical recheck mode: {canonical_recheck.get('mode')}")
assert len(producer_tests['steps'])==2 and all(step['exitCode']==0 for step in producer_tests['steps'])
assert producer_tests['producerRevision']==p_revision
for status in (native,cohort_tests,preparation_status):
 assert status['exitCode']==0
 assert status['producerRevision']==p_revision and status['consumerRevision']==c_revision
native_files=[
 'tests/test_proxy_v8.py','tests/test_proxy_reflection_v8.py','tests/test_intrinsic_error_v8.py',
 'tests/test_collection_v8.py','tests/test_iterator_collections_v8.py','tests/test_iterator_spread_v8.py',
 'tests/test_primitive_conversion_v8.py','tests/test_primitive_conversion_source_v8.py',
 'tests/test_construction_v8.py','tests/test_protocol_v8.py','tests/test_protocol_properties.py',
 'tests/test_promise_scheduler_v8.py','tests/test_promise_executor_v8.py',
 'tests/test_promise_executor_boundaries_v8.py','tests/test_promise_join_v8.py',
 'tests/test_async_call_v8.py','tests/test_semantic_observation_unicode.py','tests/test_prepare_semantic_v8.py',
 'tests/test_native_v8.py','tests/test_native_generator_close_v8.py',
 'tests/test_prediction_prefix_v8.py','tests/test_enumeration_v8.py',
 'tests/test_lifecycle_recovery_v8.py','tests/test_native_v8_without_legacy.py',
 'tests/test_protocol_host_composition_v8.py','tests/test_protocol_host_v8.py',
]
native_module_count=len(native_files)
assert len(native_files)==26
assert native['command']==[str(root/'.venv/bin/python'),'-m','pytest','-q',*native_files]
paired_native_files=[
 'test/semantic-promise-adoption-getter-source-v8.test.js',
 'test/semantic-promise-mutation-profile-source-v8.test.js',
 'test/semantic-promise-object-profile-source-v8.test.js',
 'test/semantic-promise-object-review-source-v8.test.js',
 'test/semantic-promise-object-review-v8.test.js',
 'test/semantic-promise-object-source-v8.test.js',
 'test/semantic-promise-object-v8.test.js',
 'test/semantic-protocol-promise-adoption-source-v8.test.js',
 'test/semantic-shared-generator-source-v8.test.js',
]
paired_native_titles=[
 'native independently checks adoption constructor jobs, retained scopes and selected completions',
 'actual native independently replays every eligible mutation-boundary control',
 'native independently replays every eligible Promise object control',
 'native independently checks every Promise review source and retained continuation',
 'native independently replays retained invocations and captured adoption selection',
 'independent native checker replays every new source and retained shared state',
 'independent native transitions reproduce all Promise object cases and reject altered records',
 'actual native adoption consumer independently replays every source context and restored cut',
 'actual native shared replay checks every generated context and continuation',
]
assert paired_native['schema']=='issue51-paired-native-run/1'
assert paired_native['selectionMode']=='full-files'
assert paired_native['producerRevision']==p_revision and paired_native['consumerRevision']==c_revision
assert paired_native['producerRevisionAfter']==p_revision and paired_native['consumerRevisionAfter']==c_revision
assert paired_native['observedSelectedTests']=={title:1 for title in paired_native_titles}
assert git(producer,'rev-parse','HEAD')==p_revision and not git(producer,'status','--porcelain','--untracked-files=all')
assert git(consumer,'rev-parse','HEAD')==c_revision and not git(consumer,'status','--porcelain','--untracked-files=all')
assert all(paired_native[key] is True for key in (
 'producerCleanBefore','producerCleanAfter','consumerCleanBefore','consumerCleanAfter'))
assert paired_native['expectedTestCount']==9
assert paired_native['files']==paired_native_files
assert paired_native['testTitles']==paired_native_titles
assert paired_native['testCounts']=={'pass':9,'fail':0,'skip':0,'cancel':0}
file_counts=paired_native['fileTestCounts']
tap_summary=paired_native['tapSummary']
assert set(file_counts)=={'pass','fail','skip','cancel'}
assert file_counts['fail']==0 and file_counts['cancel']==0
assert tap_summary['tests']==file_counts['pass']+file_counts['fail']+file_counts['skip']+file_counts['cancel']
assert (tap_summary['pass'],tap_summary['fail'],tap_summary['skipped'],tap_summary['cancelled'])==(
 file_counts['pass'],file_counts['fail'],file_counts['skip'],file_counts['cancel'])
assert tap_summary['tests']>paired_native['expectedTestCount']
assert paired_native['exitCode']==0
assert paired_native['command'] and paired_native['durationSeconds']>=0
assert '--test-name-pattern' not in paired_native['command']
assert '--test-reporter=tap' in paired_native['command']
assert paired_native['command'][-len(paired_native_files):]==paired_native_files
paired_native_log=root/'finish-paired-native.log'
paired_native_runner=root/'run-finish-paired-native.py'
assert paired_native_log.is_file() and paired_native_runner.is_file()
assert paired_native['logPath']==str(paired_native_log)
paired_native_evidence={
 'status':paired_native,
 'log':'finish-paired-native.log',
 'logSha256':hashlib.sha256(paired_native_log.read_bytes()).hexdigest(),
 'runner':'run-finish-paired-native.py',
 'runnerSha256':hashlib.sha256(paired_native_runner.read_bytes()).hexdigest(),
}
cohort=read('cohort-finish.json')
assert cohort['producerRevision']==p_revision and cohort['producerTreeClean'] is True
assert cohort_export['producerRevision']==p_revision
assert cohort_export['exitCode']==0 and cohort_export['producerTreeUnchanged'] is True
assert cohort_export['reportSha256']==hashlib.sha256((root/'cohort-finish.json').read_bytes()).hexdigest()
assert cohort_export['logSha256']==hashlib.sha256((root/'finish-cohort-export.log').read_bytes()).hexdigest()
cases=cohort['cases']
supported=sum(case['artifact'] is not None for case in cases)
refused=len(cases)-supported
assert len(cases)==cohort_sync['declared'] and supported==cohort_sync['supported']
assert len(cohort['dispositions'])==len(cases)
assert all(row['status']==('passed' if row['expected']=='supported' else 'refused') for row in cohort['dispositions'])
supported_cases=[case for case in cases if case['artifact'] is not None]
cohort_export_path=root/'cohort-finish.json'
cohort_export_bytes=cohort_export_path.read_bytes()
consumer_cohort_archive=consumer/'tests/fixtures/semantic-issue51-objects-v8.json.gz.b64'
consumer_cohort_decoded=gzip.decompress(base64.b64decode(consumer_cohort_archive.read_bytes()))
assert consumer_cohort_decoded==cohort_export_bytes, 'consumer cohort archive differs from the exact final producer export bytes'
del consumer_cohort_decoded
artifact_jsonl_path=root/'artifacts-finish.jsonl'
artifact_jsonl_hasher=hashlib.sha256()
with artifact_jsonl_path.open('rb') as artifact_stream:
 for case in supported_cases:
  expected_line=(json.dumps(case['artifact'],separators=(',',':'))+'\n').encode()
  actual_line=artifact_stream.readline()
  assert actual_line==expected_line, f"preparation JSONL line differs from exported artifact {case['id']}"
  artifact_jsonl_hasher.update(actual_line)
 assert artifact_stream.read()==b'', 'preparation JSONL contains extra rows after the final supported artifact'
artifact_jsonl_sha256=artifact_jsonl_hasher.hexdigest()
supported_source_ids=[case['id'] for case in supported_cases]
assert len(supported_source_ids)==len(set(supported_source_ids))==supported
contract_bytes=(producer/'contracts/semantic-execution-v8.json').read_bytes()
contract=json.loads(contract_bytes)
assert contract_bytes==(consumer/'src/megatoken/data/contracts/semantic-execution-v8.json').read_bytes()
for sync_status in (sync,cohort_sync):
 assert sync_status['producerRevision']==p_revision
 assert sync_status['producerTreeClean'] is True
 for field,expected in {
  'contractDigest':contract['digest'],
  'fixtureDigest':contract['fixtureDigest'],
  'interoperabilityDigest':contract['interoperabilityDigest'],
 }.items():
  assert sync_status[field]==expected,(field,sync_status.get(field),expected)
support_files=[
 'semantic-generator-close-v8.json.gz.b64',
 'semantic-enumeration-v8.json.gz.b64',
 'semantic-host-recovery-v8.json.gz.b64',
]
def sha256_path(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()
support_log=root/'finish-support-fixtures.log'
assert support_status['schema']=='issue51-support-fixture-run/1'
assert support_status['producerRevision']==p_revision
assert support_status['producerRevisionAfter']==p_revision
assert support_status['producerTreeClean'] is True
assert support_status['consumerRevisionAfter']==support_status['consumerRevision']
assert git(consumer,'merge-base','--is-ancestor',support_status['consumerRevision'],c_revision)== ''
assert support_status['runnerScriptSha256']==sha256_path(root/'run-finish-support-fixtures.py')
assert support_status['generatorScriptSha256']==sha256_path(consumer/'scripts/generate_protocol_support_v8.mjs')
assert support_status['command']==['node','scripts/generate_protocol_support_v8.mjs','../producer-dev']
assert support_status['log']=='finish-support-fixtures.log'
assert support_status['logSha256']==sha256_path(support_log)
assert support_status['generatorExitCode']==0 and support_status['exitCode']==0
assert support_status['verificationPassed'] is True
assert support_status['allowedGeneratedFiles']==support_files
allowed_paths={f'tests/fixtures/{name}' for name in support_files}
assert set(support_status['changedFiles'])<=allowed_paths
assert set(support_status['changedTrackedFiles'])<=allowed_paths
assert support_status['unexpectedChanges']==[]
assert support_status['unexpectedUntrackedPathChanges']==[]
expected_support_hashes={name:sha256_path(consumer/'tests/fixtures'/name) for name in support_files}
assert support_status['generatedSupportFileSha256']==expected_support_hashes
assert python_style_status['consumerRevision']==c_revision
assert len(python_style_status['steps'])==2
style_commands=[step['command'] for step in python_style_status['steps']]
assert all(command[0]==str(root/'.venv/bin/ruff') for command in style_commands)
assert style_commands[0][1]=='check'
assert style_commands[1][1:3]==['format','--check']
assert all(step['exitCode']==0 for step in python_style_status['steps'])
combined_gate_path=producer/'tools/combined-v8-gate.json'
combined_current=combined_gate_path.read_bytes()
combined_baseline=subprocess.check_output([
 'git','-C',str(producer),'show',
 'f7696e63258913e7bb8a450c67a708b70c3232ee:tools/combined-v8-gate.json'
])
combined_definition=json.loads(combined_baseline)
combined_gate_evidence={
 'path':'tools/combined-v8-gate.json',
 'baselineCommit':'f7696e63258913e7bb8a450c67a708b70c3232ee',
 'run':False,
 'matchesHistoricalBaseline':combined_current==combined_baseline,
 'baselineSha256':hashlib.sha256(combined_baseline).hexdigest(),
 'currentSha256':hashlib.sha256(combined_current).hexdigest(),
 'historicalDefinition':{key:combined_definition[key] for key in (
  'schema','contractDigest','fixtureDigest','interoperabilityDigest','sourceExamples','payloads','lanes'
 )},
}
if not combined_gate_evidence['matchesHistoricalBaseline']:
 print(json.dumps({'combinedGateHistoricalBaselineMismatch':combined_gate_evidence},indent=2),file=sys.stderr)
 raise SystemExit('Combined gate changed from historical baseline; it was not rewritten or run.')
gate=json.loads(gate_evidence_log.read_text().splitlines()[-1])
source_recheck=json.loads(source_evidence_log.read_text().splitlines()[-1])
assert source_recheck['valid'] is True
assert source_recheck['mode']==('generated' if canonical_recheck['mode']=='verified-generation-artifact' else 'checked')
assert len(source_recheck['cases'])==sync['sourceCases']
assert gate['valid'] is True and gate['mode']=='checked'
assert len(gate['protocolSources'])==sync['sourceCases']
preparation_lines=[line for line in (root/'finish-preparation.log').read_text().splitlines() if line.startswith('{')]
assert len(preparation_lines)==1
preparation=json.loads(preparation_lines[0])
report=preparation['report']
assert report==read('prepared-finish/semantic-v8-report.json')
assert report['producerRevision']==p_revision and report['consumerRevision']==c_revision
assert report['counts']=={'attempted':supported,'admitted':supported} and report['refusals']==[]
preparation_input=preparation_status['command']
input_flag=preparation_input.index('--input')
assert preparation_input[input_flag+1]==str(artifact_jsonl_path)
preparation_input_binding={
 'path':str(artifact_jsonl_path),
 'sha256':artifact_jsonl_sha256,
 'attempted':report['counts']['attempted'],
 'admitted':report['counts']['admitted'],
 'refusals':report['refusals'],
 'supportedSourceIds':supported_source_ids,
}
assert report['optimizerSteps']==0 and report['learningClaim'] is None
assert report['maxPositions']==8192 and report['maxHistory']==1024
assert report['tokenizerSha256']=='40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60'
assert report['tokenizerIdentity']=='7f52faca2fe82273a889023defd2f1a6fd6080a21096ec77e97f4798a0a370b2'
assert report['contractDigest']==contract['digest'] and report['fixtureDigest']==contract['fixtureDigest']
for case in cases:
 if case['artifact'] is not None:
  assert case['artifact']['sidecar']['datasetBinding']['registryDigest']==report['registryDigest']
def node_counts(name):
 text=(root/name).read_text()
 result={key:int(value) for key,value in re.findall(r'(?m)^[ℹ#] (tests|pass|fail|skipped) (\d+)\s*$',text)}
 assert result['fail']==0 and result['pass']>0,result
 return result
def pytest_counts(name):
 text=(root/name).read_text()
 passed=re.findall(r'(\d+) passed',text)
 assert passed and not re.search(r'\d+ failed|\d+ errors?',text)
 skipped=re.findall(r'(\d+) skipped',text)
 return {'passed':int(passed[-1]),'skipped':int(skipped[-1]) if skipped else 0}
js=node_counts('finish-producer-all.log')
py=pytest_counts('finish-native-final.log')
cohort_checks=pytest_counts('finish-cohort-tests.log')
logs={
 'finish-contracts.log':str(gate_evidence_log.relative_to(root)),
 'finish-canonical-source-evidence.log':str(source_evidence_log.relative_to(root)),
 'finish-cohort-export.log':'finish-cohort-export.log',
 'finish-cohort-export-status.json':'finish-cohort-export-status.json',
 'finish-producer-tests.log':'finish-producer-all.log',
 'finish-reference.log':'finish-reference.log',
 'finish-native-tests.log':'finish-native-final.log',
 'finish-paired-native.log':'finish-paired-native.log',
 'finish-paired-native-status.json':'finish-paired-native-status.json',
 'finish-cohort-tests.log':'finish-cohort-tests.log',
 'finish-support-fixtures.log':'finish-support-fixtures.log',
 'finish-support-status.json':'finish-support-status.json',
 'finish-python-style.log':'finish-python-style.log',
 'finish-python-style-status.json':'finish-python-style-status.json',
 'finish-preparation.log':'finish-preparation.log',
}
driver_names=[
 'regenerate-finish-canonical.py','recheck-finish-canonical.py',
 'sync-object-canonical.py','run-finish-support-fixtures.py',
 'run-finish-producer-checks.py','run-finish-cohort-export.py',
 'sync-finish-cohort.py','run-finish-native-checks.py','run-finish-paired-native.py',
 'run-finish-cohort-checks.py','run-finish-preparation.py',
 'check-finish-python-style.py','record-finish-verification.py',
]
logs.update({'tools/'+name:name for name in driver_names})
historical_root=root/'finish-pre-correction-96e6d320'
historical_files=(
 'finish-producer-all.log','finish-producer-status.json','manifest.json',
 'finish-canonical-status.json','finish-canonical-generation-prefix.json',
 *tuple(f'finish-canonical-{index:02d}.log' for index in range(10)),
)
historical_pre_correction={
 'classification':'pre-correction historical evidence; the producer suite was interrupted and these files are not a final verification result',
 'files':{name:hashlib.sha256((historical_root/name).read_bytes()).hexdigest() for name in historical_files},
}
correction_root=root/'finish-pre-correction-5553de1a-6bb45749'
correction_manifest_path=correction_root/'manifest.json'
correction_manifest=json.loads(correction_manifest_path.read_text())
correction_files=correction_manifest['files']
for name,details in correction_files.items():
 path=correction_root/name
 assert path.is_file() and path.stat().st_size==details['bytes'],name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==details['sha256'],name
correction_summary={
 'classification':correction_manifest['classification'],
 'producerRevision':correction_manifest['producerRevision'],
 'consumerRevision':correction_manifest['consumerRevision'],
 'producer':correction_manifest['producer'],
 'nativeConsumer':correction_manifest['nativeConsumer'],
 'cohortConsumer':correction_manifest['cohortConsumer'],
 'preparation':correction_manifest['preparation'],
 'canonicalGeneration':correction_manifest['canonicalGeneration'],
 'manifestSha256':hashlib.sha256(correction_manifest_path.read_bytes()).hexdigest(),
 'files':correction_files,
 'largeArtifactsRetainedOutsideRepositoryEvidence':correction_manifest['largeArtifactsRetainedOutsideRepositoryEvidence'],
 'preparedOutputDirectory':correction_manifest['preparedOutputDirectory'],
}
selection_history_root=root/'finish-paired-setup-selection-20261124'
selection_manifest_path=selection_history_root/'manifest.json'
selection_manifest=json.loads(selection_manifest_path.read_text())
selection_files=selection_manifest['files']
for name,details in selection_files.items():
 path=selection_history_root/name
 assert path.is_file() and path.stat().st_size==details['bytes'],name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==details['sha256'],name
selection_history={
 'classification':selection_manifest['classification'],
 'producerRevision':selection_manifest['producerRevision'],
 'consumerRevision':selection_manifest['consumerRevision'],
 'exitCode':selection_manifest['exitCode'],
 'testCounts':selection_manifest['testCounts'],
 'tapSummary':selection_manifest['tapSummary'],
 'rawFailureDetailCount':selection_manifest['testCounts']['fail'],
 'distinctFailedTests':selection_manifest['tapSummary']['fail'],
 'countInterpretation':'testCounts.fail includes repeated failure-detail records; tapSummary.fail is the authoritative distinct TAP test count.',
 'manifestSha256':hashlib.sha256(selection_manifest_path.read_bytes()).hexdigest(),
 'files':selection_files,
}
paired_correction_root=root/'finish-paired-self-resolution-80280e4'
paired_correction_manifest_path=paired_correction_root/'manifest.json'
paired_correction_manifest=json.loads(paired_correction_manifest_path.read_text())
paired_correction_files=paired_correction_manifest['files']
expected_paired_correction_files={
 'run-finish-paired-native.py',
 'finish-paired-native.log',
 'finish-paired-native-status.json',
 'finish-paired-full-files-driver.log',
 'finish-paired-full-files-launch.json',
}
assert paired_correction_manifest['schema']=='issue51-paired-consumer-correction/1'
assert paired_correction_manifest['producerRevision']=='a72267e26ab7b7fc5f22134abee315fd7ed6e64b'
assert paired_correction_manifest['consumerRevision']=='80280e425b0402bb4ee5f18e42e27087fa91dea5'
assert paired_correction_manifest['exitCode']==1
assert paired_correction_manifest['selectedTestCounts']=={'cancel':0,'fail':1,'pass':8,'skip':0}
assert paired_correction_manifest['fileTestCounts']=={'cancel':0,'fail':1,'pass':129,'skip':0}
assert paired_correction_manifest['tapSummary']=={'cancelled':0,'fail':1,'pass':129,'skipped':0,'tests':130}
assert set(paired_correction_files)==expected_paired_correction_files
for name,details in paired_correction_files.items():
 assert Path(name).name==name,name
 path=paired_correction_root/name
 assert path.is_file() and path.stat().st_size==details['bytes'],name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==details['sha256'],name
paired_correction_summary={
 'classification':paired_correction_manifest['classification'],
 'producerRevision':paired_correction_manifest['producerRevision'],
 'consumerRevision':paired_correction_manifest['consumerRevision'],
 'exitCode':paired_correction_manifest['exitCode'],
 'selectedTestCounts':paired_correction_manifest['selectedTestCounts'],
 'fileTestCounts':paired_correction_manifest['fileTestCounts'],
 'tapSummary':paired_correction_manifest['tapSummary'],
 'failure':{
  'testTitle':'actual native adoption consumer independently replays every source context and restored cut',
  'cause':'Promise self-resolution reached intrinsic Error allocation without an active ordinary invocation; native consumer rejected it with active invocation required.',
 },
 'manifestSha256':hashlib.sha256(paired_correction_manifest_path.read_bytes()).hexdigest(),
 'files':paired_correction_files,
}
receipt={
 'schema':'issue51-object-verification/1',
 'historicalPreCorrection':historical_pre_correction,
 'historicalCorrectionRun':correction_summary,
 'historicalPairedSetupSelectionRun':selection_history,
 'historicalPairedConsumerCorrectionRun':paired_correction_summary,
 'date':datetime.now(ZoneInfo('Europe/Berlin')).date().isoformat(),
 'producerRevision':p_revision,'consumerRevision':c_revision,
 'producerTree':git(producer,'rev-parse','HEAD:'),'consumerTree':git(consumer,'rev-parse','HEAD:'),
 'generationImplementationRevision':canonical['implementationRevision'],
 'generatedArtifactRevision':canonical['producerRevision'],'nodeVersion':cohort['nodeVersion'],
 'pythonVersion':subprocess.check_output([str(root/'.venv/bin/python'),'--version'],text=True).strip(),
 'contractDigest':contract['digest'],'fixtureDigest':contract['fixtureDigest'],
 'interoperabilityDigest':contract['interoperabilityDigest'],
 'canonical':{'protocolSources':len(gate['protocolSources']),'regions':len(gate['examples']),
              'hostSources':len(gate['protocolHostSources']),'preservedSourceInputs':sync['preservedSources'],
              'synchronization':sync},
 'supportFixtureRun':support_status,
 'pythonStyleRun':python_style_status,
 'historicalCombinedGate':combined_gate_evidence,
 'cohort':{'declared':len(cases),'supported':supported,'refused':refused,
           'preservedOriginalInputs':cohort_sync['preservedOriginals'],'registryDigest':report['registryDigest'],
           'operationCoverage':cohort['operationCoverage'],'methodCoverage':cohort['methodCoverage'],
           'dispositions':cohort['dispositions'],
           'exportBinding':{
               'exportJsonSha256':hashlib.sha256(cohort_export_bytes).hexdigest(),
               'consumerArchiveSha256':sha256_path(consumer_cohort_archive),
               'artifactsJsonlSha256':artifact_jsonl_sha256,
               'supportedSourceIds':supported_source_ids}},
 'preparationInputBinding':preparation_input_binding,
 'tests':{'producer':js,'producerRun':producer_tests,'native':py,'nativeRun':native,
          'nativeModuleCount':native_module_count,
          'pairedNative':paired_native_evidence,
          'cohort':cohort_checks,'cohortRun':cohort_tests},
 'canonicalRun':canonical,'canonicalRecheckRun':canonical_recheck,'cohortExportRun':cohort_export,
 'preparationRun':preparation_status,'preparation':preparation,
 'evidenceSha256':{name:hashlib.sha256((root/source).read_bytes()).hexdigest() for name,source in logs.items()},
 'scope':'Completed bounded issue51 follow-up. Issue51 remains open; no full language-completeness, combined CI matrix, optimizer or learning claim.'
}
text=f"""
## Completed follow-up — retained object operations and recovery

The paired implementation verified here is producer `{p_revision}` and consumer
`{c_revision}`, both clean when the recorded checks began. Documentation commits
may advance the draft branches after these exact implementation and data pins.

The receipt records the canonical generation source revision
`{canonical['implementationRevision']}` and generated artifact revision
`{canonical['producerRevision']}` separately from the verified producer
revision above. The source fixture generator and unmodified standard contract
gate both have successful, hashed evidence logs. When the verified producer
revision exactly matches the generated artifact revision, the receipt uses
those recorded generation and gate results without claiming a second rerun.
For a later code-only producer revision, it records fresh source-check and
gate runs. Original generation status and all ten generation logs are preserved
separately as historical evidence.

### What this follow-up establishes

- Seven Proxy reflection operations now share the ordinary descriptor, prototype
  and extensibility rules: `getPrototypeOf`, `setPrototypeOf`, `isExtensible`,
  `preventExtensions`, `getOwnPropertyDescriptor`, `ownKeys` and
  `defineProperty`. Object and Reflect callers retain their different behavior.
  Callbacks use the selected handler and method; invariants inspect the target
  after callback effects. Definition callbacks receive a fresh descriptor object
  while the original descriptor snapshot remains authenticated through calls,
  returns and checkpoints.
- Transparent nested Proxy targets and `Proxy.revocable` retain distinct object,
  handler, target and revoker identities. Repeated revocation has no second
  effect. Revoked get, set, has and delete operations derive the correct
  TypeError, message and standard prototype from their source operands.
- A handler's own `has` getter is an explicit continuation. It runs once with
  the handler receiver. Its returned closure remains selected across handler
  mutation or revocation; a nullish result falls back to the captured target.
  Getter throws preserve identity. Every exposed selection/call/return boundary
  is checked on restoration.
- Standard Error objects use shared ordinary descriptors and prototype identity,
  including their inherited names, own messages and ordinary `instanceof`
  behavior. Invalid `in` checks retain operand effects and reject an invalid
  right-hand side before key conversion. Errors caught inside Promise reactions
  remain ordinary shared objects.
- Derived-constructor arrows retain the live receiver and `new.target` before
  and after `super()`, including pre-initialization ReferenceErrors and later
  scheduled calls. Promise reactions and adoption getters preserve nested call
  scopes, captured values and authentic parameter bindings across returns and
  suspended checkpoints.
- Ordinary `for…in` checks captured candidates against live property state,
  including prototype and enumerability changes under the pinned engine.
  Stable transparent Proxy enumeration retains Proxy identity and target state.
- Map and Set operations retain SameValueZero equality and ordered tombstones.
  Live collection iterators observe supported mutations without restarting;
  completed iterators stay complete. Array/string spread retains evaluated
  values before later effects. Nested ordinary array binding, defaults, rest
  and abrupt outer-iterator cleanup retain their order and identities.

### Verification on the recorded revisions

| Check | Recorded result |
| --- | --- |
| Standard unmodified `npm run contracts:check` | Passed; {len(gate['protocolSources'])} protocol source cases, {len(gate['examples'])} region examples, {len(gate['protocolHostSources'])} host source cases |
| Complete producer test selection | {js['pass']} passed, {js.get('skipped',0)} skipped, zero failures; {producer_tests['testFiles']} test modules |
| Producer reference fixture | Passed |
| Paired native-source checks | 9 selected titles passed with 0 selected failures/skips/cancellations; all {tap_summary['tests']} tests in 9 files ran ({file_counts['pass']} passed, {file_counts['skip']} skipped, 0 failed/cancelled) at the recorded producer/consumer pins |
| Independent Python regression selection | {py['passed']} passed, {py['skipped']} skipped across {native_module_count} test modules |
| Python Ruff lint and format checks | {len(python_style_status['steps'])} steps passed |
| Complete frozen issue51 cohort checks | {cohort_checks['passed']} passed, {cohort_checks['skipped']} skipped |
| Actual pinned-tokenizer preparation | {supported} attempted, {supported} admitted, zero refusals and zero optimizer steps |

The producer command uses Node {cohort['nodeVersion']} with experimental VM
modules and {producer_tests['testConcurrency']} concurrent test modules. The full command and individual test
results are retained in the receipt and logs. The Python selection and its
exact command are recorded separately; these counts are tests, not percentages
of JavaScript or independent source families.

The canonical population preserves all **{sync['preservedSources']} previous
source inputs** byte-for-byte for source, entry, arguments, family and ancestry.
The issue51 cohort contains **{len(cases)} declared originals: {supported}
supported and {refused} refused**, preserving all **{cohort_sync['preservedOriginals']}
previous cohort inputs** in the same fields. Earlier negative controls were
promoted only after their unchanged original source passed native execution,
source observation checks and fresh replay. The cohort remains one declared
ancestry group; additional examples do not become independent families.

A separate historical correction run on producer `5553de1a10761626c5604a57453291039d73683d`
and consumer `6bb4574931c4fdf19588a86aef96577b22ff6541` recorded 3,558 producer tests
(3,429 passed, 31 failed, 98 skipped), an interrupted native run with 465 passed
and 16 failed among 481 completed of 546 collected, 125 passed and 9 failed
cohort tests, and preparation that admitted 111 of 121 attempts with 10 refusals.
These results are retained as historical correction evidence and are not final
passing verification. The receipt records the archive manifest hash and hashes
for its small evidence files; the large cohort export, preparation JSONL and
prepared output remain outside repository evidence. The failed initial paired
test-title filtering attempt is also preserved as historical runner-selection
evidence and is not counted as a semantic result. Its raw failure-detail counter
is 18, while the TAP summary records the authoritative 9 distinct failed tests.

A separate full-files paired native run on producer
`a72267e26ab7b7fc5f22134abee315fd7ed6e64b` and consumer
`80280e425b0402bb4ee5f18e42e27087fa91dea5` completed 130 tests: 129 passed,
one failed, zero skipped or cancelled. The selected nine native checks had
eight passes and one failure in the Promise self-resolution adoption replay:
intrinsic Error allocation occurred without an active ordinary invocation,
and the consumer rejected it with `active invocation required`. This is
historical correction evidence, not a final passing result. Its manifest and
five hash-verified files are preserved under
`historical/finish-paired-self-resolution-80280e4/`.

The receipt records operation and method counts for every accepted source.
Method counts distinguish the operation and Object/Reflect caller where the
record provides them. They describe exercised transitions, not general method
coverage percentages.

Preparation used the pinned Ling tokenizer with 8192 positions and 1024 history.
The preparation report confirms zero optimizer steps and no learning claim.
The full manifest/report fingerprint and tokenizer identity are retained.

- Contract digest: `{contract['digest']}`.
- Fixture digest: `{contract['fixtureDigest']}`.
- Interoperability digest: `{contract['interoperabilityDigest']}`.
- Frozen registry digest: `{report['registryDigest']}`.
- Preparation manifest: `{preparation['manifestFingerprint']}`.

See [the exact verification receipt](evidence/issue51-20261124/finish-verification.json)
and [the preparation result](evidence/issue51-20261124/finish-preparation.json).
Successful logs and their SHA256 hashes accompany them. Earlier checkpoint
receipts below remain historical evidence for their recorded revisions.

### Explicit remaining boundaries

Issue #51 is an ongoing umbrella and remains open. This follow-up does not
claim every Proxy/constructor/iterator combination. Remaining bounded cases
include general callable Proxy apply/construct, nested target invariant
dispatch, effectful Proxy enumeration, trap getters beyond the own `has`
path, non-callable trap-getter results, effectful descriptor or own-key list
conversion, broader exotic inheritance and asynchronous callbacks, general
nested custom iterables, assignment destructuring, and implicit captured
builtin iterator bindings. Native engine-specific Error stacks remain outside
the standard Error property profile.

Only the recorded Node 24 environment was checked here. The separate combined CI gate file was compared byte-for-byte with
historical baseline `f7696e63258913e7bb8a450c67a708b70c3232ee`; the baseline and
current hashes and original definition values are in the receipt. The gate has
`run: false`: this is a pin-preservation check, not a passing combined CI
execution. Neither draft PR is merged.

### Reproduction

Check out the producer and consumer revisions above. Regenerate the paired
fixtures with the repository generators only when intentionally changing their
inputs; the saved receipts apply to the committed artifacts at these pins.

```sh
# In the producer:
npm run contracts:check
node --experimental-vm-modules --test --test-concurrency=4 test/*.test.js
node fixtures/reference/reference-fixture.mjs
node tools/export-issue51-evidence.mjs --out /tmp/issue51-finish.json

# Create the preparation input from the same frozen export:
python - /tmp/issue51-finish.json /tmp/issue51-artifacts.jsonl <<'PYINPUT'
import json, sys
with open(sys.argv[2], "w") as output:
    for row in json.load(open(sys.argv[1]))["cases"]:
        if row["artifact"] is not None:
            print(json.dumps(row["artifact"]), file=output)
PYINPUT

# In the consumer:
PYTHONPATH=src python -m pytest -q tests/test_issue51_object_source_v8.py
PYTHONPATH=src python scripts/prepare_semantic_v8.py \\
  --input /tmp/issue51-artifacts.jsonl --output /tmp/issue51-prepared \\
  --producer-root /path/to/tracing-js-self \\
  --producer-revision {p_revision} \\
  --consumer-revision {c_revision} \\
  --tokenizer-json /path/to/pinned/tokenizer.json \\
  --max-positions 8192 --max-history 1024 --source-recheck-timeout-seconds 90
```
"""
historical_root=root/'finish-pre-correction-96e6d320'
historical_files=(
 'finish-producer-all.log','finish-producer-status.json','manifest.json',
 'finish-canonical-status.json','finish-canonical-generation-prefix.json',
 *tuple(f'finish-canonical-{index:02d}.log' for index in range(10)),
)
for name in historical_files:
 assert (historical_root/name).is_file(), name
for repo in (producer,consumer):
 directory=repo/'docs/evidence/issue51-20261124'
 directory.mkdir(parents=True,exist_ok=True)
 historical_directory=directory/'historical'
 historical_directory.mkdir(parents=True,exist_ok=True)
 for name in historical_files:
  shutil.copyfile(historical_root/name,historical_directory/name)
 correction_directory=historical_directory/'finish-pre-correction-5553de1a-6bb45749'
 correction_directory.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(correction_manifest_path,correction_directory/'manifest.json')
 for name in correction_files:
  shutil.copyfile(correction_root/name,correction_directory/name)
 selection_directory=historical_directory/'finish-paired-setup-selection-20261124'
 selection_directory.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(selection_manifest_path,selection_directory/'manifest.json')
 for name in selection_files:
  shutil.copyfile(selection_history_root/name,selection_directory/name)
 paired_correction_directory=historical_directory/'finish-paired-self-resolution-80280e4'
 paired_correction_directory.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(paired_correction_manifest_path,paired_correction_directory/'manifest.json')
 for name in paired_correction_files:
  shutil.copyfile(paired_correction_root/name,paired_correction_directory/name)
 for name,source in logs.items():
  (directory/name).parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(root/source,directory/name)
 (directory/'finish-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (directory/'finish-preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
 (directory/'finish-canonical-sync.json').write_text(json.dumps(sync,indent=2)+'\n')
 (directory/'finish-cohort-sync.json').write_text(json.dumps(cohort_sync,indent=2)+'\n')
 doc=repo/'docs/issue51-object-semantics.md'
 old=doc.read_text()
 assert '## Completed follow-up —' not in old
 old=re.sub(r'\nCurrent completed scope and receipts are in \*\*Checkpoint 6\*\* below\. Earlier\ncheckpoint descriptions apply only to their recorded revisions\.\n','\n',old,count=1)
 heading,history=old.split('\n',1)
 banner='\nCurrent verified scope is in **Completed follow-up** immediately below.\nEarlier checkpoints describe only their recorded code and inputs.\n'
 history=history.replace('## Starting point','## Historical starting point',1).replace('## Implementation checklist','## Historical initial implementation checklist',1)
 doc.write_text(heading+'\n'+banner+text+'\n## Historical work record\n'+history)
(root/'finish-final-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(root/'finish-current-summary.md').write_text(text)
print(json.dumps({'producerRevision':p_revision,'consumerRevision':c_revision,'producerTests':js,
                 'nativeTests':py,'cohortTests':cohort_checks,'supported':supported,'refused':refused,
                 'manifestFingerprint':preparation['manifestFingerprint']},indent=2))
