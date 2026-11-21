"""Stage a Windows audio runtime from pinned vendor assets; dry-run by default.

No model weights or reference voices are bundled. The target must not exist;
all sources are verified before any payload is activated. Not a publisher.
"""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,uuid
ROOT=pathlib.Path(__file__).resolve().parent.parent

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binaries',type=pathlib.Path,required=True)
    p.add_argument('--onnx',type=pathlib.Path,required=True)
    p.add_argument('--cuda',type=pathlib.Path)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--stage',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('output exists; choose a new versioned directory')
    manifest=json.loads((ROOT/'reference/runtime'/('onnxruntime-gpu-1.23.0-windows-x64.json' if a.cuda else 'onnxruntime-1.23.0-windows-x64.json')).read_text())
    sources=[]
    for name in ['xrt-cli.exe','xrt-server.exe']:
        source=a.binaries/name
        if not source.is_file():raise RuntimeError('missing binary '+str(source))
        sources.append((name,source,source.stat().st_size,digest(source)))
    for f in [manifest['dll'],*manifest.get('companion_dlls',[])]:
        sources.append((f['file_name'],a.onnx/f['file_name'],f['size_bytes'],f['sha256']))
    for name in ['onnxruntime.LICENSE','onnxruntime.ThirdPartyNotices.txt']:
        source=a.onnx/name;sources.append((name,source,source.stat().st_size,digest(source)))
    if a.cuda:
        cuda=json.loads((ROOT/'reference/runtime/cuda12-payload-xrt-audio-windows-x64.json').read_text())
        for f in cuda['files']:sources.append((f['file_name'],a.cuda/f['file_name'],f['size_bytes'],f['sha256']))
    for name in ['LICENSE','NOTICE','README.md','docs/AUDIO-PIPELINE.md']:
        source=ROOT/name;sources.append((name,source,source.stat().st_size,digest(source)))
    # Source identity and exact bytes are recorded even for local qualification.
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT))
    inventory=[]
    for name,source,size,want in sources:
        if source.is_symlink() or (hasattr(source,'is_junction') and source.is_junction()):raise RuntimeError('linked payload source')
        if source.stat().st_size!=size or digest(source)!=want:raise RuntimeError('source identity mismatch '+str(source))
        inventory.append({'path':name,'size_bytes':size,'sha256':want})
    metadata={'schema_version':1,'source_commit':commit,'dirty_source':dirty,'qualification_only':True,
              'signing':'unsigned','platform':'windows-x86_64','backend':'cuda' if a.cuda else 'cpu','files':inventory}
    print(json.dumps({'action':'stage' if a.stage else 'plan','bytes':sum(f['size_bytes'] for f in inventory),
                      'output':str(a.output),'source_commit':commit,'dirty_source':dirty}),flush=True)
    if not a.stage:return
    if shutil.disk_usage(a.output.parent).free < sum(f['size_bytes'] for f in inventory)+64*1024*1024:raise RuntimeError('insufficient staging disk space')
    staging=a.output.with_name(a.output.name+'.staging-'+uuid.uuid4().hex);staging.mkdir()
    for name,source,size,want in sources:
        target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,target.open('xb') as dst:
            shutil.copyfileobj(src,dst,8*1024*1024);dst.flush();os.fsync(dst.fileno())
        if target.stat().st_size!=size or digest(target)!=want:raise RuntimeError('staged file differs '+name)
    with (staging/'artifact.json').open('x',encoding='utf-8') as f:json.dump(metadata,f,indent=2)
    with (staging/'EXPERIMENTAL-UNSIGNED.txt').open('x',encoding='utf-8') as f:
        f.write('Experimental unsigned local qualification build. Not a published release. CUDA payload requires its vendor redistribution notices before publication.\n')
    # rename, not replace: a concurrently created destination is never overwritten.
    os.rename(staging,a.output)
    print('STAGED',a.output,flush=True)

if __name__=='__main__':main()
