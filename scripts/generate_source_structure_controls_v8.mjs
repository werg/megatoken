// Rebuild local admission-shape and recovery-census controls from their original
// source and inputs. Neither fixture attests a producer revision or tokenizer.
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {gzipSync} from 'node:zlib';

const args = process.argv.slice(2), check = args.includes('--check');
const paths = args.filter(arg => arg !== '--check');
assert(paths.length <= 1, 'Usage: generate_source_structure_controls_v8.mjs [producer] [--check]');
const producer = resolve(paths[0] ?? '../tracing-js-self');
const directory = resolve('tests/fixtures');
const {semanticRecipeFamilies} = await import(pathToFileURL(resolve(producer, 'src/training/recipe-fixtures.js')));
const {verifyRegionTrainingCase, recheckRegionEvidence} = await import(pathToFileURL(resolve(producer, 'src/semantic/compiler-v8.js')));
const family = semanticRecipeFamilies().find(row => row.template === 'rectangle-perimeter');
assert(family, 'the original rectangle-perimeter source family must remain declared');
const result = await verifyRegionTrainingCase({...family.repeated, sessionCases: family.repeated.cases, seed: 17});
assert.equal(result.status, 'locally_verified', result.reason);
assert.equal(result.declared, 2);
assert(result.components.every(row => row.sourceQualified));
const artifact = JSON.parse(JSON.stringify(result.exported));
const checked = await recheckRegionEvidence(artifact);
for (const field of ['valid', 'sourceValidated', 'projectionValidated', 'replayValidated']) {
  assert.equal(checked[field], true, `fresh session ${field} must pass`);
}
assert.equal(checked.completed, 2);
assert.equal(artifact.sidecar.nativeAdmission, null);
assert.equal(artifact.sidecar.optimizerSteps, 0);
assert.equal(artifact.sidecar.learningClaim, null);
assert.equal(artifact.sidecar.sourceArtifact.sidecar.producerRevision, null);
const fixtures = new Map([['semantic-source-session-v8.json.gz.b64', {
  fixtureAuthority: 'producer-generated local source-session structure control; its source and tokenizer receipts are not production admission evidence',
  artifact,
}]]);

const recovery = await verifyRegionTrainingCase({...family.callbacks, seed: 17});
assert.equal(recovery.status, 'locally_verified', recovery.reason);
const recoveryArtifact = JSON.parse(JSON.stringify(recovery.exported));
const recoveryChecked = await recheckRegionEvidence(recoveryArtifact);
for (const field of ['valid', 'sourceValidated', 'projectionValidated', 'replayValidated']) {
  assert.equal(recoveryChecked[field], true, `fresh recovery ${field} must pass`);
}
assert(recoveryArtifact.public.timeline.some(row => row.kind === 'reconcile'));
assert(recoveryArtifact.public.timeline.some(row => row.kind === 'source-evict'));
// The census fixture retains only the exact public trace and replay contexts;
// it is a classifier control, with no production source-admission claim.
fixtures.set('semantic-source-recovery-census-v8.json.gz.b64', {
  public: recoveryArtifact.public, contexts: recoveryArtifact.contexts,
});
for (const [name, fixture] of fixtures) {
  const output = resolve(directory, name);
  const bytes = gzipSync(JSON.stringify(fixture), {level: 9}).toString('base64') + '\n';
  if (check) assert.equal(readFileSync(output, 'utf8'), bytes, `Regenerate ${name}`);
  else writeFileSync(output, bytes);
}
console.log(JSON.stringify({mode: check ? 'checked' : 'generated', fixtures: [...fixtures.keys()],
  components: checked.completed, sessionContexts: artifact.contexts.length,
  recoveryContexts: recoveryArtifact.contexts.length, sourceChecks: 'passed',
  publicDigest: artifact.sidecar.publicDigest, contextDigest: artifact.sidecar.contextDigest}));
