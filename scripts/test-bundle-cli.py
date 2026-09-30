"""Real CLI import/verify/corruption proof; no networking and no tree deletion."""
import argparse, hashlib, json, os, pathlib, subprocess, tempfile

def main():
 p=argparse.ArgumentParser();p.add_argument('--cli',required=True);a=p.parse_args()
 root=pathlib.Path(tempfile.mkdtemp(prefix='xrt-bundle-cli-'));source=root/'source';source.mkdir()
 files=[]
 for name,data in [('model.onnx',b'test bytes not inference weights'),('LICENSE',b'fixture license')]:
  with (source/name).open('xb') as f:f.write(data)
  files.append({'path':name,'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source':'https://example.com/v1/'+name})
 manifest={'schema_version':2,'id':'cli-fixture','revision':'fixed','kind':'model','family':'fixture','tasks':['test'],'minimum_runtime':'0.3.0','platforms':['any'],'backends':['cpu'],'entrypoints':{'model':'model.onnx'},'dependencies':[],'license':{'spdx':'MIT','evidence':'fixture','files':['LICENSE']},'files':sorted(files,key=lambda f:f['path'])}
 encoded=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode();digest=hashlib.sha256(encoded).hexdigest()
 path=root/'manifest.json'
 with path.open('xb') as f:f.write(encoded)
 def run(*args,ok=True):
  r=subprocess.run([a.cli,'bundle','--cache',str(root/'cache'),*args],capture_output=True,text=True)
  if (r.returncode==0)!=ok:raise AssertionError(r.stdout+r.stderr)
  return json.loads(r.stdout) if r.returncode==0 else None
 run('import','--manifest',str(path),'--digest',digest,'--directory',str(source))
 assert not (root/'cache'/'bundles'/'cli-fixture').exists()
 installed=run('import','--manifest',str(path),'--digest',digest,'--directory',str(source),'--confirm')
 assert installed['digest']==digest
 assert run('verify','cli-fixture','--digest',digest)['verified']
 assert run('path','cli-fixture')['path']==installed['path']
 run('import','--manifest',str(path),'--digest','a'*64,'--directory',str(source),'--confirm',ok=False)
 target=pathlib.Path(installed['path'])/'model.onnx';before=target.read_bytes();tmp=target.with_suffix('.tmp')
 with tmp.open('xb') as f:f.write(bytes([before[0]^1])+before[1:])
 os.replace(tmp,target)
 run('verify','cli-fixture','--digest',digest,ok=False)
 run('remove','cli-fixture','--digest',digest,'--confirm',ok=False)
 with tmp.open('xb') as f:f.write(before)
 os.replace(tmp,target)
 assert run('remove','cli-fixture','--digest',digest)['removed'] is False
 assert target.exists()
 extra=pathlib.Path(installed['path'])/'user-note.txt';extra.write_bytes(b'keep this')
 run('remove','cli-fixture','--digest',digest,'--confirm',ok=False)
 assert target.read_bytes()==before and extra.read_bytes()==b'keep this'
 assert run('verify','cli-fixture','--digest',digest)['verified']
 extra.unlink()
 empty=pathlib.Path(installed['path'])/'extra-directory';empty.mkdir()
 run('remove','cli-fixture','--digest',digest,'--confirm',ok=False)
 assert target.read_bytes()==before
 empty.rmdir()
 partial=root/'cache'/'.partial-bundles'/('cli-fixture-'+digest);partial.mkdir(parents=True)
 (partial/'bad.onnx').write_bytes(b'corrupt partial')
 assert not run('discard-partial','cli-fixture','--digest',digest)['partial_discarded']
 assert partial.exists()
 assert run('discard-partial','cli-fixture','--digest',digest,'--confirm')['partial_discarded']
 assert not partial.exists() and target.exists()
 assert run('remove','cli-fixture','--digest',digest,'--confirm')['removed'] is True
 assert not target.exists()
 run('path','cli-fixture',ok=False)
 print('PASS dry-run, import, pinned identity, offline verify, resolve, wrong manifest refusal, corrupted artifact refusal, safe removal')
 print('Evidence retained:',root)

if __name__=='__main__':main()
