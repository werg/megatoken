// Public replay controls only: these selected records are not source certification.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {gzipSync} from 'node:zlib';

const producer = resolve(process.argv[2] ?? '../tracing-js-self');
const output = resolve(process.argv[3] ?? 'tests/fixtures/semantic-retained-invocations-v8.json.gz.b64');
const producerRevision = execFileSync('git', ['rev-parse', 'HEAD'], {cwd: producer, encoding: 'utf8'}).trim();
assert.equal(execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {cwd: producer, encoding: 'utf8'}).trim(), '');
const {RegionExecutionHarness} = await import(pathToFileURL(resolve(producer, 'src/semantic/region-harness.js')));
const {analyzeSemanticProgram} = await import(pathToFileURL(resolve(producer, 'src/training/semantic-program.js')));
const source = '"use strict";function run(initial){let count=initial;return function add(value){count=count+value;return count;};}';
const addSite = 'function:' + analyzeSemanticProgram(source).functions.find(fn => fn.name === 'add').start;
const runSite = 'function:' + analyzeSemanticProgram(source).functions.find(fn => fn.name === 'run').start;
const examples = [];

function machine(id) {
  const h = new RegionExecutionHarness({sessionId: id});
  h.publishProgram({programId: 'program', source, entry: 'run', randomSeed: 17});
  const contexts = [];
  return {h, contexts, select(invocationId, kind, fields) {
    contexts.push(h.modelContext(invocationId));
    h.select(invocationId, {kind, fields});
  }};
}
function enter(h, invocationId, options = {}) {
  h.admitInvocation({programId: 'program', invocationId, realmId: 'realm', trackId: 'worker',
    triggerId: invocationId, realmPolicy: 'retained', args: [], ...options});
}
function finish(m, invocation, value) {
  m.select(invocation, 'return', {value});
  m.select(invocation, 'end', {kind: 'return'});
}
function initialize(m) {
  enter(m.h, 'root', {args: [5]});
  m.select('root', 'bind', {name: 'count', value: ['num', 5]});
  m.select('root', 'allocate', {name: 'fn', kind: 'closure', code: addSite,
    captures: [{binding: 'count', cell: 'count'}], properties: []});
  finish(m, 'root', ['ref', 'fn']);
}
function increment(m, invocation, amount, total, options = {}) {
  enter(m.h, invocation, {closureId: 'fn', entry: 'add', args: [amount], ...options});
  advance(m, invocation, total);
}
function advance(m, invocation, total) {
  m.select(invocation, 'compute', {expr: {op: 'binary', operator: '+',
    left: {op: 'local', name: 'count'}, right: {op: 'local', name: 'value'}}, result: ['num', total]});
  m.select(invocation, 'bind', {name: 'count', value: ['num', total]});
  finish(m, invocation, ['num', total]);
}
function retain(id, m) {
  const publicEpisode = m.h.exportEpisode();
  assert.deepEqual(RegionExecutionHarness.replay(publicEpisode).timeline, m.h.timeline);
  assert.deepEqual(RegionExecutionHarness.restore(JSON.parse(JSON.stringify(m.h.checkpoint()))).timeline, m.h.timeline);
  examples.push({id, public: publicEpisode, contexts: m.contexts});
}

const calls = machine('retained-calls');
initialize(calls);
increment(calls, 'next', 2, 7);
increment(calls, 'again123', 3, 10);
enter(calls.h, 'ordinary', {entry: runSite, args: [9]});
finish(calls, 'ordinary', ['num', 9]);
retain('retained-closure-and-ordinary-calls', calls);

const callback = machine('retained-callback');
initialize(callback);
increment(callback, 'child', 4, 9, {activation: 'callback', parentInvocationId: 'root'});
retain('nonnumeric-callback-identity', callback);

const separate = machine('fresh-realms');
enter(separate.h, 'first', {realmId: 'one', realmPolicy: 'fresh', args: [2]});
finish(separate, 'first', ['num', 2]);
enter(separate.h, 'second', {realmId: 'two', realmPolicy: 'fresh', args: [8]});
finish(separate, 'second', ['num', 8]);
retain('fresh-realms-use-global-invocation-order', separate);

const host = machine('host-retained-handler');
initialize(host);
host.h.registerEventHandler({subscriptionId: 'outside', programId: 'program', realmId: 'realm',
  trackId: 'worker', closureId: 'fn', queueLimit: 4});
host.h.postEvent({subscriptionId: 'outside', eventId: 'first', args: [2]});
advance(host, 'event1', 7);
host.h.postEvent({subscriptionId: 'outside', eventId: 'second', args: [4]});
advance(host, 'event2', 11);
host.h.closeEventHandler('outside');
retain('host-retained-closure-handler', host);

for (const realmPolicy of ['fresh', 'retained']) {
  const host = machine('host-' + realmPolicy + '-handler');
  host.h.registerEventHandler({subscriptionId: 'outside', programId: 'program', realmId: 'realm',
    trackId: 'worker', realmPolicy, entry: 'run', queueLimit: 2});
  host.h.postEvent({subscriptionId: 'outside', eventId: 'first', args: [2]});
  finish(host, 'event1', ['num', 2]);
  host.h.postEvent({subscriptionId: 'outside', eventId: 'second', args: [8]});
  finish(host, 'event2', ['num', 8]);
  host.h.closeEventHandler('outside');
  retain('host-' + realmPolicy + '-static-handler', host);
}

const discarded = machine('host-discarded-arrivals');
initialize(discarded);
discarded.h.registerEventHandler({subscriptionId: 'outside', programId: 'program', realmId: 'realm',
  trackId: 'worker', closureId: 'fn', queueLimit: 2});
discarded.h.postEvent({subscriptionId: 'outside', eventId: 'queued-first', args: [2], deliver: false});
discarded.h.postEvent({subscriptionId: 'outside', eventId: 'queued-second', args: [4], deliver: false});
discarded.h.closeEventHandler('outside');
enter(discarded.h, 'after', {entry: 'run', args: [9]});
finish(discarded, 'after', ['num', 9]);
retain('host-discards-private-arrivals', discarded);

const ownedFresh = machine('host-owned-fresh-handler');
enter(ownedFresh.h, 'owner', {args: [5]});
finish(ownedFresh, 'owner', ['num', 5]);
ownedFresh.h.registerEventHandler({subscriptionId: 'outside', programId: 'program', realmId: 'realm',
  trackId: 'worker', realmPolicy: 'fresh', entry: 'run', ownerInvocationId: 'owner', queueLimit: 2});
ownedFresh.h.postEvent({subscriptionId: 'outside', eventId: 'first', args: [2]});
finish(ownedFresh, 'event1', ['num', 2]);
ownedFresh.h.postEvent({subscriptionId: 'outside', eventId: 'second', args: [8]});
finish(ownedFresh, 'event2', ['num', 8]);
ownedFresh.h.closeEventHandler('outside');
retain('host-owned-fresh-static-handler', ownedFresh);

writeFileSync(output, gzipSync(JSON.stringify({schema: 'semantic-retained-invocation-controls/1',
  producerRevision, evidenceScope: 'selected-public-replay-controls', examples})).toString('base64') + '\n');
console.log(JSON.stringify({producerRevision, examples: examples.length,
  generatedRecords: examples.reduce((total, example) => total + example.contexts.length, 0)}));
