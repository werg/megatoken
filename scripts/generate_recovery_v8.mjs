// Selected public replay controls: these records do not certify source execution.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {gzipSync} from 'node:zlib';

const producer = resolve(process.argv[2] ?? '../tracing-js-self');
const output = resolve(process.argv[3] ?? 'tests/fixtures/semantic-recovery-v8.json.gz.b64');
const producerRevision = execFileSync('git', ['rev-parse', 'HEAD'], {cwd: producer, encoding: 'utf8'}).trim();
assert.equal(execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {cwd: producer, encoding: 'utf8'}).trim(), '');
const {RegionExecutionHarness} = await import(pathToFileURL(resolve(producer, 'src/semantic/region-harness.js')));
const {analyzeSemanticProgram} = await import(pathToFileURL(resolve(producer, 'src/training/semantic-program.js')));
const {DurableHostIngressBroker} = await import(pathToFileURL(resolve(producer, 'src/semantic/host-ingress-journal-v8.js')));
const {DurableHostIngressJournal} = await import(pathToFileURL(resolve(producer, 'src/storage/host-ingress-journal.js')));
const examples = [];

function machine(id, source, capabilities, services = {}, {harness = {}, invocation = {}} = {}) {
  const m = {h: new RegionExecutionHarness({sessionId: id, services, ...harness}), contexts: []};
  const root = invocation.invocationId ?? 'root';
  m.h.publishProgram({programId: 'program', source, entry: 'run', capabilities});
  m.h.admitInvocation({programId: 'program', invocationId: 'root', realmId: 'realm',
    trackId: 'worker', triggerId: 'outside', realmPolicy: 'retained', ...invocation});
  m.select = (invocation, kind, fields) => {
    m.contexts.push(m.h.modelContext(invocation));
    m.h.select(invocation, {kind, fields});
  };
  m.request = (effect, operation, args, payload, extra = {}) => {
    const {start, end} = analyzeSemanticProgram(source).sites.find(site => site.operation === operation);
    m.select(root, 'request', {effect, operation, invocation: root,
      site: {start, end}, arguments: args, payload, ...extra});
  };
  return m;
}
function finish(m, invocation, value, kind = 'return') {
  m.select(invocation, kind, {value});
  m.select(invocation, 'end', {kind});
}
function retain(id, m, recovery, requireTerminal = true, restored = null) {
  const publicEpisode = m.h.exportEpisode();
  assert.deepEqual(RegionExecutionHarness.replay(publicEpisode, {requireTerminal}).timeline, m.h.timeline);
  assert.deepEqual((restored ?? RegionExecutionHarness.restore(JSON.parse(JSON.stringify(m.h.checkpoint())))).timeline, m.h.timeline);
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

const directory = mkdtempSync(resolve('.recovery-v8-'));
try {
  const source = '"use strict";function run(){let saved={value:9};return host.natlangCall("x",saved,function handler(value,alias){if(saved.value)return value;return alias;});}';
  let calls = 0, metadata;
  const owned = machine('owned-callback-recovery', source, ['natlang.call'], {
    'natlang.call': (_args, meta) => {calls++; metadata = meta; return new Promise(() => {});},
  }, {harness: {automaticDelivery: false}, invocation: {completionPolicy: 'await-returned-promise'}});
  owned.select('root', 'allocate', {name: 'retained', kind: 'object', properties: [{key: 'value', value: ['num', 9]}]});
  owned.select('root', 'bind', {name: 'saved', value: ['ref', 'retained']});
  owned.select('root', 'allocate', {name: 'fn', kind: 'closure', code: 'handler',
    captures: [{binding: 'saved', cell: 'saved'}], properties: []});
  owned.request('effect', 'natlang.call', [['str', 'x'], ['ref', 'retained'], ['ref', 'fn']],
    JSON.stringify({schema: 'owned-graph/1', roots: [['str', 'x'], ['ref', 'retained'], ['callback', 'fn']],
      objects: [{id: 'retained', kind: 'object', properties: [{key: 'value', value: ['num', 9]}]}]}),
    {promise: 'external'});
  void owned.h.dispatch('root', 'effect');
  for (let turn = 0; turn < 20 && !metadata; turn++) await Promise.resolve();
  assert.equal(calls, 1);
  assert(metadata);
  const journal = new DurableHostIngressJournal({directory: join(directory, 'journal'), streamId: 'owned-inputs'});
  const broker = new DurableHostIngressBroker({harness: owned.h, journal});
  const input = {value: 4}; input.self = input;
  const beforeArrival = owned.h.exportEpisode();
  const receipt = await broker.receiveCallback({ownerInvocationId: 'root', parentEffectId: 'effect',
    closureId: 'fn', invocationId: 'child', args: [input, input], receiver: input});
  await broker.flush();
  assert.equal(receipt.status, 'buffered');
  assert.deepEqual(owned.h.exportEpisode(), beforeArrival, 'durable arrival is private until activation');
  input.value = 99;
  const restore = async () => {
    const packet = JSON.parse(JSON.stringify(owned.h.checkpoint()));
    const independentReader = new DurableHostIngressJournal({directory: join(directory, 'journal'), streamId: 'owned-inputs'});
    owned.h = await RegionExecutionHarness.restoreDurable(packet, {ingressJournal: independentReader,
      ingressLogReceipt: broker.receipt, services: {'natlang.call': () => {calls++; throw new Error('parent adapter repeated');}}});
    assert.equal(owned.h.inner.intents.get('root/effect').status, 'uncertain');
    await assert.rejects(owned.h.dispatch('root', 'effect'), /uncertain|committed/);
    assert.equal(calls, 1);
  };
  await restore();
  assert.equal(owned.h.inner.closureQueue.length, 1);
  owned.select('root', 'return', {value: ['ref', 'external']});
  owned.h.inner.activateQueuedClosures('realm'); owned.h.sync();
  const child = owned.h.invocations.get('child').controller;
  const value = child.receiver;
  assert.deepEqual(child.input.graph.roots, [value, value]);
  assert.deepEqual(child.heap.get(value[1]).properties.get('value'), ['num', 4]);
  assert.deepEqual(child.heap.get(value[1]).properties.get('self'), value);
  assert.deepEqual(child.locals.get('saved'), ['ref', 'retained']);
  owned.select('child', 'return', {value});
  await restore();
  assert.equal(owned.h.inner.closureQueue.length, 0);
  assert.equal(owned.h.inner.completionWaiters.size, 0, 'restore cannot revive the old adapter continuation');
  owned.select('child', 'end', {kind: 'return'});
  assert.deepEqual(owned.h.inner.callbackOutcome('child').completion.value, value);
  assert.equal(owned.h.inner.intents.get('root/effect').status, 'uncertain', 'child completion does not invent the adapter result');
  owned.h.reconcile('root', 'effect', {disposition: 'fulfilled', result: value, evidenceId: 'independent-parent-result'});
  owned.select('root', 'end', {kind: 'return'});
  assert.deepEqual(owned.h.invocations.get('root').controller.completion.value, value);
  assert.equal(calls, 1);
  const completed = await RegionExecutionHarness.restoreDurable(JSON.parse(JSON.stringify(owned.h.checkpoint())), {
    ingressJournal: new DurableHostIngressJournal({directory: join(directory, 'journal'), streamId: 'owned-inputs'}),
    ingressLogReceipt: broker.receipt,
  });
  assert.deepEqual(completed.invocations.get('root').controller.completion.value, value);
  retain('owned-return-callback-reconciliation', owned, {adapterCalls: calls, durableArrivals: 1,
    independentReaderRestores: 3, callbackResult: value, outcome: 'completed'}, true, completed);
} finally {
  rmSync(directory, {recursive: true, force: true});
}

for (const invocationId of ['callbackinvoke9', 'callbackinvokezero']) {
  let calls = 0;
  const named = machine('ordinary-' + invocationId, listenerSource, ['demo.listen'], {
    'demo.listen': (_, meta) => {calls++; return meta.subscriptionId(1);},
  }, {invocation: {invocationId}});
  named.select(invocationId, 'allocate', {name: 'fn', kind: 'closure', code: 'handler', captures: [], properties: []});
  named.request('registration', 'demo.listen', [['str', 'tick'], ['ref', 'fn']],
    '["tick",{"$ownedCallback":"fn"}]', {promise: 'registration'});
  await named.h.dispatch(invocationId, 'registration');
  assert.equal(named.h.inner.callbacks.get(invocationId + '/registration/1').subscriptionId, 'callback1');
  finish(named, invocationId, ['undefined']);
  named.h.closeEventHandler('callback1');
  retain('notification-after-ordinary-' + invocationId, named, {invocationId, adapterCalls: calls,
    subscriptionId: 'callback1', outcome: 'completed'});
}

let lateCalls = 0, releaseRegistration;
const retired = machine('late-retired-registration', listenerSource, ['demo.listen'], {
  'demo.listen': (_, meta) => {lateCalls++; return new Promise(resolve => {
    releaseRegistration = () => resolve(meta.subscriptionId(1));
  });},
});
retired.select('root', 'allocate', {name: 'fn', kind: 'closure', code: 'handler', captures: [], properties: []});
retired.request('registration', 'demo.listen', [['str', 'tick'], ['ref', 'fn']],
  '["tick",{"$ownedCallback":"fn"}]', {promise: 'registration'});
const lateResult = retired.h.dispatch('root', 'registration');
for (let turn = 0; turn < 20 && !releaseRegistration; turn++) await Promise.resolve();
assert.equal(lateCalls, 1);
retired.h.closeEventHandler('callback1');
releaseRegistration(); await lateResult;
assert.equal(retired.h.inner.handlers.get('callback1').state, 'closed');
assert.equal(retired.h.inner.callbacks.get('root/registration/1').state, 'revoked');
finish(retired, 'root', ['undefined']);
retain('late-result-keeps-registration-retired', retired, {adapterCalls: lateCalls,
  registrationRetiredBeforeResult: true, outcome: 'completed'});

const mixedSource = '"use strict";function run(){return host.natlangCall("x",null,handler);}function handler(value){return host.listen("tick",listener);}function listener(value){return value;}';
let returningCalls = 0, notificationCalls = 0;
const mixed = machine('returning-then-notification-callback', mixedSource, ['natlang.call', 'demo.listen'], {
  'natlang.call': (_args, meta) => {returningCalls++; return meta.invokeClosure('fn', [4]);},
  'demo.listen': (_, meta) => {notificationCalls++; return meta.subscriptionId(1);},
}, {harness: {automaticDelivery: false}, invocation: {completionPolicy: 'await-returned-promise'}});
mixed.select('root', 'allocate', {name: 'fn', kind: 'closure', code: 'handler', captures: [], properties: []});
mixed.request('effect', 'natlang.call', [['str', 'x'], ['null'], ['ref', 'fn']],
  JSON.stringify({schema: 'owned-graph/1', roots: [['str', 'x'], ['null'], ['callback', 'fn']], objects: []}),
  {promise: 'external'});
const returningResult = mixed.h.dispatch('root', 'effect');
for (let turn = 0; turn < 20 && !mixed.h.inner.closureQueue.length; turn++) await Promise.resolve();
assert.equal(returningCalls, 1);
mixed.select('root', 'return', {value: ['ref', 'external']});
mixed.h.inner.activateQueuedClosures('realm'); mixed.h.sync();
const childId = 'callbackinvoke1';
assert(mixed.h.invocations.has(childId));
mixed.select(childId, 'allocate', {name: 'notification', kind: 'closure', code: 'listener', captures: [], properties: []});
const notificationSite = analyzeSemanticProgram(mixedSource).sites.find(site => site.operation === 'demo.listen');
mixed.select(childId, 'request', {effect: 'registration', operation: 'demo.listen', invocation: childId,
  site: {start: notificationSite.start, end: notificationSite.end},
  arguments: [['str', 'tick'], ['ref', 'notification']], payload: '["tick",{"$ownedCallback":"notification"}]',
  promise: 'registered'});
await mixed.h.dispatch(childId, 'registration');
assert.equal(notificationCalls, 1);
assert.equal(mixed.h.inner.callbacks.get(childId + '/registration/1').subscriptionId, 'callback1');
finish(mixed, childId, ['ref', 'registered']);
await returningResult;
mixed.select('root', 'end', {kind: 'return'});
mixed.h.closeEventHandler('callback1');
retain('returning-callback-does-not-allocate-notification-identities', mixed, {
  adapterCalls: returningCalls + notificationCalls, returningAdapterCalls: returningCalls,
  registrationCalls: notificationCalls, returningInvocation: childId, subscriptionId: 'callback1', outcome: 'completed',
});

const sharedSource = '"use strict";function run(){host.listen("first",handler);return host.listen("second",handler);}function handler(value){return value;}';
const completions = [];
let sharedCalls = 0;
const shared = machine('two-registrations-one-closure', sharedSource, ['demo.listen'], {
  'demo.listen': (_, meta) => {sharedCalls++; return new Promise(resolve => {
    completions.push(() => resolve(meta.subscriptionId(1)));
  });},
});
shared.select('root', 'allocate', {name: 'fn', kind: 'closure', code: 'handler', captures: [], properties: []});
const sharedSites = analyzeSemanticProgram(sharedSource).sites.filter(site => site.operation === 'demo.listen');
const pendingRegistrations = [];
for (const [index, effect] of ['first', 'second'].entries()) {
  shared.request(effect, 'demo.listen', [['str', effect], ['ref', 'fn']],
    JSON.stringify([effect, {$ownedCallback: 'fn'}]), {promise: effect,
      site: {start: sharedSites[index].start, end: sharedSites[index].end}});
  pendingRegistrations.push(shared.h.dispatch('root', effect));
  for (let turn = 0; turn < 20 && completions.length <= index; turn++) await Promise.resolve();
}
assert.equal(sharedCalls, 2);
assert.deepEqual([...shared.h.inner.callbacks.values()].map(callback => callback.subscriptionId), ['callback1', 'callback2']);
for (const [index, complete] of completions.entries()) {complete(); await pendingRegistrations[index];}
finish(shared, 'root', ['undefined']);
shared.h.closeEventHandler('callback1'); shared.h.closeEventHandler('callback2');
retain('concurrent-registrations-share-one-closure', shared, {adapterCalls: sharedCalls,
  registrations: 2, sharedClosure: 'fn', outcome: 'completed'});

let actionCalls = 0;
const action = machine('bound-native-action', numberSource, ['demo.readNumber'], {
  'demo.readNumber': (_args, meta) => {
    actionCalls++; assert.equal(meta.idempotencyKey, 'selected-action/1'); return 7;
  },
});
action.request('effect', 'demo.readNumber', [['str', 'key']], '["key"]');
const actionIntent = action.h.inner.intents.get('root/effect');
const nativeAction = {schema: 'semantic-native-action/4', actionKey: 'selected-action/1',
  invocationId: 'root', effectId: 'effect', operation: actionIntent.operation,
  payload: actionIntent.payload, ownerTrack: 'worker',
  recordDigest: createHash('sha256').update(actionIntent.witness).digest('hex'),
  modelTrack: 'controlled-model-track', authority: {issuer: 'selected-test', scopes: ['dispatch'], evidence: null},
  signature: {algorithm: 'fixture-only', value: 'not-authenticated'}};
await action.h.dispatch('root', 'effect', {beforeDispatch: async () => structuredClone(nativeAction)});
assert.equal(actionCalls, 1);
assert.deepEqual(actionIntent.nativeAction, nativeAction);
finish(action, 'root', ['num', 7]);
retain('native-action-binds-the-exact-selected-request', action, {adapterCalls: actionCalls,
  nativeAction, nativeActionAuthentication: 'not-claimed', outcome: 'completed'});

writeFileSync(output, gzipSync(JSON.stringify({schema: 'semantic-recovery-controls/1', producerRevision,
  evidenceScope: 'selected-public-replay-controls', examples})).toString('base64') + '\n');
console.log(JSON.stringify({producerRevision, examples: examples.length,
  generatedRecords: examples.reduce((total, example) => total + example.contexts.length, 0)}));
