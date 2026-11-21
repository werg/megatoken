// Rebuild supporting native fixtures from their original source/controlled
// calls. Historical source receipts are never repaired by editing their state.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve,join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {gzipSync,gunzipSync} from 'node:zlib';

const args=process.argv.slice(2),check=args.includes('--check'),paths=args.filter(arg=>arg!=='--check');
if(paths.length>1)throw new Error('Usage: generate_protocol_support_v8.mjs [producer] [--check]');
const producer=resolve(paths[0]??'../tracing-js-self'),directory=resolve('tests/fixtures');
const producerRevision=execFileSync('git',['rev-parse','HEAD'],{cwd:producer,encoding:'utf8'}).trim();
assert.equal(execFileSync('git',['status','--porcelain','--untracked-files=no'],{cwd:producer,encoding:'utf8'}).trim(),'','Clean producer required');
const fromProducer=path=>import(pathToFileURL(join(producer,path)));
const {verifyProtocolRegionTrainingCase,recheckProtocolRegionEvidence}=await fromProducer('src/semantic/protocol-source-v8.js');
const {LANGUAGE_EXPANSION_INTERACTIONS}=await fromProducer('fixtures/language-expansion/benchmark.js');
const {hashAlgorithmData}=await fromProducer('src/semantic/program.js');
const {RegionExecutionHarness}=await fromProducer('src/semantic/region-harness.js');
const {analyzeSemanticProgram}=await fromProducer('src/training/semantic-program.js');
const unpack=name=>JSON.parse(gunzipSync(Buffer.from(readFileSync(join(directory,name),'utf8'),'base64')));
const packets=new Map();

const generator=LANGUAGE_EXPANSION_INTERACTIONS.find(row=>row.id==='generator-delegation-close');
assert(generator,'Original generator-close input required');
const originalGenerator=unpack('semantic-generator-close-v8.json.gz.b64');
assert.equal(originalGenerator.public.timeline.find(row=>row.kind==='program').fields.source,generator.source);
assert.deepEqual(originalGenerator.public.timeline.find(row=>row.kind==='invocation').fields.inputGraph.roots,[['num',1]]);
const generated=await verifyProtocolRegionTrainingCase({...generator,entry:'run',family:'ordinary-array-mutation'});
assert(generated.exported,generated.reason);
const sourceChecks=await recheckProtocolRegionEvidence(generated.exported);
assert(sourceChecks.sourceValidated&&sourceChecks.projectionValidated&&sourceChecks.replayValidated);
packets.set('semantic-generator-close-v8.json.gz.b64',{
  ...generated.exported,provenance:{case:generator.id,producerRevision,
    sourceInputDigest:hashAlgorithmData({...generator,entry:'run',family:'ordinary-array-mutation'}),
    source:'independently-rechecked',native:'untested',model:'unmeasured'}
});

const sourcePacket=JSON.parse(readFileSync(join(producer,'test/fixtures/semantic-protocol-region-v8.json'),'utf8'));
const enumeration=sourcePacket.cases.find(row=>row.id==='source-for-in-own-keys');
assert(enumeration,'Original enumeration input required');
const contract=JSON.parse(readFileSync(join(producer,'contracts/semantic-execution-v8.json'),'utf8'));
const declaration=contract.interoperability.find(row=>row.id==='protocol-region:'+enumeration.id);
assert.equal(hashAlgorithmData(enumeration.artifact.public),declaration.publicDigest);
assert.equal(hashAlgorithmData(enumeration.artifact.contexts),declaration.contextDigest);
const enumerationChecks=await recheckProtocolRegionEvidence(enumeration.artifact);
assert(enumerationChecks.sourceValidated&&enumerationChecks.projectionValidated&&enumerationChecks.replayValidated);
packets.set('semantic-enumeration-v8.json.gz.b64',enumeration.artifact);

// Re-execute the original selected-call recovery control against an actual
// supplied service. These authored selections remain non-source-certified.
const source='"use strict";function run(){return host.listen("tick",handler);}function handler(value){return value;}';
const previous=unpack('semantic-host-recovery-v8.json.gz.b64');
assert.equal(previous.qualification,'controlled-machine-regression-not-source-certified');
assert.equal(previous.public.timeline.find(row=>row.kind==='program').fields.source,source);
let fire,dispatches=0;
const host=new RegionExecutionHarness({sessionId:'recovery',automaticDelivery:false,services:{
  'demo.listen':(_,metadata)=>{dispatches++;fire=metadata.callback(1);return metadata.subscriptionId(1);}
}});
const contexts=[];
const select=(id,record)=>{contexts.push(host.modelContext(id));host.select(id,record);};
const finish=(id,value=['undefined'])=>{
  select(id,{kind:'return',fields:{value}});select(id,{kind:'end',fields:{kind:'return'}});
};
host.publishProgram({programId:'program',source,entry:'run',capabilities:['demo.listen']});
host.admitInvocation({programId:'program',invocationId:'root',realmId:'realm',trackId:'worker',
  triggerId:'start',realmPolicy:'retained',args:[]});
select('root',{kind:'allocate',fields:{name:'anchor',kind:'object',properties:[]}});
select('root',{kind:'property.define',object:'anchor',key:['str','marker'],success:true,
  descriptor:{kind:'data',value:['num',1],writable:true,enumerable:true,configurable:true}});
select('root',{kind:'allocate',fields:{name:'fn',kind:'closure',code:'handler',captures:[],properties:[]}});
const {start,end}=analyzeSemanticProgram(source).sites.find(row=>row.operation==='demo.listen');
select('root',{kind:'request',fields:{effect:'effect',operation:'demo.listen',invocation:'root',site:{start,end},
  arguments:[['str','tick'],['ref','fn']],payload:'["tick",{"$ownedCallback":"fn"}]',promise:'external'}});
await host.dispatch('root','effect');
fire(11);fire(22);finish('root');host.deliverReadyEvents('callback1');
finish('event1',['num',11]);host.deliverReadyEvents('callback1');
const midpoint={cut:host.timeline.length,context:host.modelContext('event2')};
assert.equal(midpoint.cut,previous.midpoint.cut);
assert.equal(host.inner.callbacks.get('root/effect/1').sequence,2);
const restored=RegionExecutionHarness.restore(JSON.parse(JSON.stringify(host.checkpoint())),{
  services:{'demo.listen':()=>{throw new Error('Completed native registration was redispatched');}}
});
assert.equal(restored.inner.callbacks.get('root/effect/1').sequence,2);
assert.deepEqual([...restored.handlers.get('callback1').seen],['callbackevent1','callbackevent2']);
assert.deepEqual(RegionExecutionHarness.replay(host.exportEpisode(),{requireTerminal:false}).modelContext('event2'),midpoint.context);
finish('event2',['num',22]);host.closeRealm('realm',{reason:'controlled recovery qualification'});
assert.equal(dispatches,1);assert.equal(contexts.length,10);
assert.deepEqual(RegionExecutionHarness.replay(host.exportEpisode()).timeline,host.timeline);
assert.deepEqual(RegionExecutionHarness.restore(JSON.parse(JSON.stringify(host.checkpoint()))).timeline,host.timeline);
packets.set('semantic-host-recovery-v8.json.gz.b64',{
  schema:previous.schema,qualification:previous.qualification,public:host.exportEpisode(),contexts,midpoint,
  producerRevision,sourceBinding:'untested',nativeAdapterCalls:dispatches
});

assert.equal(execFileSync('git',['rev-parse','HEAD'],{cwd:producer,encoding:'utf8'}).trim(),
  producerRevision,'Producer revision changed during generation');
assert.equal(execFileSync('git',['status','--porcelain','--untracked-files=no'],{cwd:producer,encoding:'utf8'}).trim(),
  '','Producer changed during generation');
for(const [name,packet]of packets){
  const bytes=gzipSync(Buffer.from(JSON.stringify(packet)+'\n')).toString('base64')+'\n',path=join(directory,name);
  if(check)assert.equal(readFileSync(path,'utf8'),bytes,'Regenerate '+name);
  else writeFileSync(path,bytes);
}
console.log(JSON.stringify({producerRevision,mode:check?'checked':'generated',
  fixtures:[...packets.keys()],sourceQualified:2,controlledHostRecovery:1,hostAdapterCalls:dispatches}));
