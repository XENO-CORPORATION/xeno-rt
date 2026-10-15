/** Exercise the real installer with an ephemeral TEST signing identity.
 * No production key, publication, model download or model-weight copy occurs.
 */
import { generateKeyPairSync, createHash, createPublicKey } from 'node:crypto'
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync, statSync } from 'node:fs'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
const args=process.argv.slice(2)
const option=name=>{const i=args.indexOf('--'+name);if(i<0||!args[i+1])throw Error('--'+name+' required');return resolve(args[i+1])}
const cli=option('cli'),server=option('server'),out=option('out')
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..')
mkdirSync(out)
const payload=join(out,'payload');mkdirSync(payload)
for(const [name,source] of [['xrt-cli.exe',cli],['xrt-server.exe',server],['LICENSE',join(root,'LICENSE')],['NOTICE',join(root,'NOTICE')],['FIRST-PARTY-PROVENANCE.json',join(root,'vendor/xenosystem-primitives.source.json')]])copyFileSync(source,join(payload,name))
const sha=bytes=>createHash('sha256').update(bytes).digest('hex')
const files=['xrt-cli.exe','xrt-server.exe','LICENSE','NOTICE','FIRST-PARTY-PROVENANCE.json'].map(path=>({path,size_bytes:statSync(join(payload,path)).size,sha256:sha(readFileSync(join(payload,path))),source:'https://example.invalid/qualification/'+path}))
const manifest={schema_version:2,id:'xrt-signed-candidate-proof',revision:'local-qualification-only',kind:'native-runtime',family:'xeno-rt',tasks:['native-audio-qualification'],minimum_runtime:'0.4.0',platforms:['windows-x86_64'],backends:['cpu','cuda'],entrypoints:{cli:'xrt-cli.exe',server:'xrt-server.exe'},dependencies:[],license:{spdx:'Apache-2.0 AND LicenseRef-Proprietary',evidence:'local qualification fixture; first-party primitives retain their proprietary declaration; not a distribution authorization',files:['LICENSE','NOTICE','FIRST-PARTY-PROVENANCE.json']},files}
const manifestPath=join(out,'bundle.json');writeFileSync(manifestPath,JSON.stringify(manifest,null,2)+'\n')
const run=(name,command,argv,env={})=>{
 const result=spawnSync(command,argv,{encoding:'utf8',windowsHide:true,env:{...process.env,...env}})
 writeFileSync(join(out,name+'.log'),(result.stdout||'')+(result.stderr||''))
 return result
}
const parsed=run('digest',cli,['bundle','digest','--manifest',manifestPath]);if(parsed.status!==0)throw Error('manifest rejected')
const digest=JSON.parse(parsed.stdout).digest
const {privateKey,publicKey}=generateKeyPairSync('ed25519')
const publicRaw=publicKey.export({format:'der',type:'spki'}).subarray(12)
const keyId=sha(publicRaw),keyPath=join(out,'TEST-public-key.json'),signaturePath=join(out,'TEST-signature.json')
writeFileSync(keyPath,JSON.stringify({schema_version:1,algorithm:'ed25519',key_id:keyId,public_key:publicRaw.toString('hex')},null,2)+'\n')
const signed=run('sign',process.execPath,[join(root,'scripts/sign-artifact-manifest.mjs'),'--cli',cli,'--manifest',manifestPath,'--out',signaturePath,'--expected-key-id',keyId],
 {XRT_MANIFEST_SIGNING_KEY:privateKey.export({format:'der',type:'pkcs8'}).toString('base64')})
if(signed.status!==0)throw Error('test signer failed')
const common=['--manifest',manifestPath,'--digest',digest,'--signature',signaturePath,'--trusted-key',keyPath,'--max-bytes','200000000']
const cache=join(out,'cache')
if(run('install',cli,['bundle','--cache',cache,'import',...common,'--directory',payload,'--confirm']).status!==0)throw Error('signed import failed')
if(run('verify',cli,['bundle','--cache',cache,'verify',manifest.id,'--digest',digest]).status!==0)throw Error('installed candidate failed checksum verification')
const corrupted=JSON.parse(readFileSync(signaturePath));corrupted.signature=(corrupted.signature.startsWith('00')?'01':'00')+corrupted.signature.slice(2)
const badSignature=join(out,'bad-signature.json');writeFileSync(badSignature,JSON.stringify(corrupted))
const rejectedCache=join(out,'rejected-cache')
const bad=run('reject-signature',cli,['bundle','--cache',rejectedCache,'import',...common.map(v=>v===signaturePath?badSignature:v),'--directory',payload,'--confirm'])
if(bad.status===0||existsSync(rejectedCache))throw Error('bad signature was accepted or created cache state')
const damaged=join(out,'damaged');mkdirSync(damaged)
for(const file of files)copyFileSync(join(payload,file.path),join(damaged,file.path))
const bytes=readFileSync(join(damaged,'xrt-server.exe'));bytes[0]^=1;writeFileSync(join(damaged,'xrt-server.exe'),bytes)
if(run('reject-payload',cli,['bundle','--cache',join(out,'damaged-cache'),'import',...common,'--directory',damaged,'--confirm']).status===0)throw Error('modified executable was accepted')
if(run('verify-after-rejection',cli,['bundle','--cache',cache,'verify',manifest.id,'--digest',digest]).status!==0)throw Error('previous verified candidate was damaged')
writeFileSync(join(out,'summary.json'),JSON.stringify({passed:true,qualification_only:true,production_publisher_authorized:false,published:false,model_weights_bundled:false,manifest_digest:digest,test_key_id:keyId,files,
 checks:['Node signer and Rust verifier interoperate','signed native binary import and offline rehash','tampered signature refused before cache creation','tampered executable refused','previous verified candidate remains intact']},null,2)+'\n')
console.log(JSON.stringify({passed:true,qualification_only:true,published:false}))
