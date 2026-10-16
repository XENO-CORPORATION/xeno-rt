/** Sign a canonical manifest digest. No download, publication or key generation.
 * Run the local signer from the xeno-platform release checkout. Private material
 * must be supplied by its protected secret store as XRT_MANIFEST_SIGNING_KEY
 * (base64 PKCS#8), never as a command-line argument or a repository file.
 */
import { createHash, createPrivateKey, createPublicKey, sign } from 'node:crypto'
import { writeFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'

const args = process.argv.slice(2)
const value = name => { const index=args.indexOf('--'+name); if(index<0 || !args[index+1] || args[index+1].startsWith('--')) throw new Error('--'+name+' is required'); return args[index+1] }
const cli=value('cli'), manifest=value('manifest'), output=value('out'), expectedKey=value('expected-key-id')
const material=process.env.XRT_MANIFEST_SIGNING_KEY
if(!material) throw new Error('XRT_MANIFEST_SIGNING_KEY is absent; load the authorized key from the protected operator secret store')
const key=createPrivateKey({key:Buffer.from(material,'base64'),format:'der',type:'pkcs8'})
if(key.asymmetricKeyType!=='ed25519') throw new Error('the authorized key must be Ed25519')
const der=createPublicKey(key).export({format:'der',type:'spki'})
if(der.length!==44 || der.subarray(0,12).toString('hex')!=='302a300506032b6570032100') throw new Error('unexpected Ed25519 public-key encoding')
const keyId=createHash('sha256').update(der.subarray(12)).digest('hex')
if(keyId!==expectedKey) throw new Error('provided key does not match the independently pinned publisher identity')
const result=spawnSync(cli,['bundle','digest','--manifest',manifest],{encoding:'utf8',windowsHide:true,env:{...process.env,XRT_MANIFEST_SIGNING_KEY:''}})
if(result.status!==0) throw new Error('the runtime refused the manifest; no signature was produced')
const {digest,id}=JSON.parse(result.stdout)
if(!/^[a-f0-9]{64}$/.test(digest)) throw new Error('runtime returned an invalid digest')
const message=Buffer.concat([Buffer.from('xeno-rt.artifact-manifest.v1\0'),Buffer.from(digest,'ascii')])
const signature=sign(null,message,key).toString('hex')
writeFileSync(output,JSON.stringify({schema_version:1,algorithm:'ed25519',key_id:keyId,manifest_digest:digest,signature},null,2)+'\n',{flag:'wx'})
console.log(JSON.stringify({id,manifest_digest:digest,key_id:keyId,signature_written:true,published:false}))
