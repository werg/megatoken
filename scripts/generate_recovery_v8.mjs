// Selected public replay controls: these records do not certify source execution.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {gzipSync} from 'node:zlib';

const producer = resolve(process.argv[2] ?? '../tracing-js-self');
const output = resolve(process.argv[3] ?? 'tests/fixtures/semantic-recovery-v8.json.gz.b64');
const producerRevision = execFileSync('git', ['rev-parse', 'HEAD'], {cwd: producer, encoding: 'utf8'}).trim();
assert.equal(execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {cwd: producer, encoding: 'utf8'}).trim(), '');
const {RegionExecutionHarness} = await import(pathToFileURL(resolve(producer, 'src/semantic/region-harness.js')));
const {analyzeSemanticProgram} = await import(pathToFileURL(resolve(producer, 'src/training/semantic-program.js')));
const examples = [];

function machine(id, source, capabilities, services = {}) {
  const m = {h: new RegionExecutionHarness({sessionId: id, services}), contexts: []};
  m.h.publishProgram({programId: 'program', source, entry: 'run', capabilities});
  m.h.admitInvocation({programId: 'program', invocationId: 'root', realmId: 'realm',
    trackId: 'worker', triggerId: 'outside', realmPolicy: 'retained'});
  m.select = (invocation, kind, fields) => {
    m.contexts.push(m.h.modelContext(invocation));
    m.h.select(invocation, {kind, fields});
  };
  m.request = (effect, operation, args, payload, extra = {}) => {
    const {start, end} = analyzeSemanticProgram(source).sites.find(site => site.operation === operation);
    m.select('root', 'request', {effect, operation, invocation: 'root',
      site: {start, end}, arguments: args, payload, ...extra});
  };
  return m;
}
function finish(m, invocation, value, kind = 'return') {
  m.select(invocation, kind, {value});
  m.select(invocation, 'end', {kind});
}
function retain(id, m, recovery, requireTerminal = true) {
  const publicEpisode = m.h.exportEpisode();
  assert.deepEqual(RegionExecutionHarness.replay(publicEpisode, {requireTerminal}).timeline, m.h.timeline);
  assert.deepEqual(RegionExecutionHarness.restore(JSON.parse(JSON.stringify(m.h.checkpoint()))).timeline, m.h.timeline);
  examples.push({id, public: publicEpisode, contexts: m.contexts, recovery});
}

const numberSource = '"use strict";function run(){return host.readNumber("key");}';
for (const disposition of ['fulfilled', 'rejected', 'lost']) {
  let calls = 0, release;
  const m = machine('recovery-' + disposition, numberSource, ['demo.readNumber'], {
    'demo.readNumber': () => {calls++; return new Promise(resolve => {release = resolve;});},
  });
  m.request('effect', 'demo.readNumber', [['str', 'key']], '["key"]');
  const pending = m.h.dispatch('root', 'effect');
  for (let turn = 0; turn < 20 && !release; turn++) await Promise.resolve();
  assert.equal(calls, 1);
  m.h = RegionExecutionHarness.restore(JSON.parse(JSON.stringify(m.h.checkpoint())), {
    services: {'demo.readNumber': () => {calls++; return 99;}},
  });
  assert.equal(m.h.inner.intents.get('root/effect').status, 'uncertain');
  await assert.rejects(m.h.dispatch('root', 'effect'), /uncertain|committed/);
  m.h.reconcile('root', 'effect', {disposition, evidenceId: 'external-receipt',
    ...(disposition === 'fulfilled' ? {value: 7} : disposition === 'rejected' ? {value: 'failed'} : {})});
  if (disposition !== 'lost') finish(m, 'root', disposition === 'fulfilled' ? ['num', 7] : ['str', 'failed'],
    disposition === 'fulfilled' ? 'return' : 'throw');
  else await assert.rejects(m.h.dispatch('root', 'effect'), /uncertain|committed/);
  assert.equal(calls, 1, 'recovery must not repeat the native adapter');
  retain('reconcile-' + disposition, m, {disposition, adapterCalls: calls,
    outcome: disposition === 'lost' ? 'unresolved-lost-effect' : 'completed'}, disposition !== 'lost');
  release(7); await pending;
}

// This observed receipt represents the boundary after durable dispatch intent
// publication and before calling the adapter. External evidence, rather than
// checkpoint age, is what makes the second attempt eligible.
const retry = machine('recovery-not-dispatched', numberSource, ['demo.readNumber']);
retry.request('effect', 'demo.readNumber', [['str', 'key']], '["key"]');
retry.h.importObserved({mode: 'observed', kind: 'dispatch', fields: {
  session: 'recovery-not-dispatched', track: 'worker', invocationId: 'root', effectId: 'effect',
  key: 'root/effect', attempt: 1,
}});
let retryCalls = 0;
retry.h = RegionExecutionHarness.restore(JSON.parse(JSON.stringify(retry.h.checkpoint())), {
  services: {'demo.readNumber': () => {retryCalls++; return 7;}},
});
await assert.rejects(retry.h.dispatch('root', 'effect'), /uncertain|committed/);
retry.h.reconcile('root', 'effect', {disposition: 'not-dispatched', evidenceId: 'before-adapter-receipt'});
await retry.h.dispatch('root', 'effect');
assert.equal(retryCalls, 1);
assert.deepEqual(retry.h.timeline.filter(row => row.kind === 'dispatch').map(row => row.fields.attempt), [1, 2]);
finish(retry, 'root', ['num', 7]);
retain('reconcile-not-dispatched', retry, {disposition: 'not-dispatched', adapterCalls: retryCalls,
  firstAttempt: 'observed-before-adapter-control', outcome: 'completed'});

const listenerSource = '"use strict";function run(){return host.listen("tick",handler);}function handler(value){return value;}';
let registrations = 0, reattachments = 0, received;
const listener = machine('callback-reattachment', listenerSource, ['demo.listen'], {
  'demo.listen': (_, meta) => {registrations++; return meta.subscriptionId(1);},
});
listener.select('root', 'allocate', {name: 'fn', kind: 'closure', code: 'handler', captures: [], properties: []});
listener.request('registration', 'demo.listen', [['str', 'tick'], ['ref', 'fn']],
  '["tick",{"$ownedCallback":"fn"}]', {promise: 'registration'});
await listener.h.dispatch('root', 'registration');
finish(listener, 'root', ['undefined']);
listener.h = RegionExecutionHarness.restore(JSON.parse(JSON.stringify(listener.h.checkpoint())), {
  services: {'demo.listen': () => {registrations++; throw new Error('registration was repeated');}},
});
assert.equal(listener.h.inner.handlers.get('callback1').state, 'lost');
listener.h.inner.reattachSubscription('callback1', {evidenceId: 'existing-native-registration',
  attach: receive => {reattachments++; received = receive;}});
received('first', [2]);
finish(listener, 'event1', ['num', 2]);
received('second', [3]);
finish(listener, 'event2', ['num', 3]);
listener.h.closeEventHandler('callback1');
assert.equal(registrations, 1);
assert.equal(reattachments, 1);
retain('reattach-existing-subscription', listener, {registrations, reattachments, outcome: 'completed'});

writeFileSync(output, gzipSync(JSON.stringify({schema: 'semantic-recovery-controls/1', producerRevision,
  evidenceScope: 'selected-public-replay-controls', examples})).toString('base64') + '\n');
console.log(JSON.stringify({producerRevision, examples: examples.length,
  generatedRecords: examples.reduce((total, example) => total + example.contexts.length, 0)}));
