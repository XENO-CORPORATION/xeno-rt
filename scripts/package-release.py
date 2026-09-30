"""Stage inspectable local RC archives from separately qualified binaries."""
import argparse, hashlib, json, pathlib, shutil, subprocess, tarfile, tomllib, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-bin', type=pathlib.Path, required=True)
    p.add_argument('--cuda-bin', type=pathlib.Path, required=True)
    p.add_argument('--linux-bin', type=pathlib.Path, required=True)
    p.add_argument('--metadata', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    version = tomllib.loads((ROOT/'Cargo.toml').read_text())['workspace']['package']['version']
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('release source must be clean')
    notices = ['Conservative locked workspace dependency notices (including build/test dependencies).\n']
    for package in sorted(json.loads(a.metadata.read_text(encoding='utf-8-sig'))['packages'], key=lambda x:x['name']):
        if not package.get('source'): continue
        base = pathlib.Path(package['manifest_path']).parent
        files = [f for f in base.iterdir() if f.is_file() and f.name.upper().startswith(('LICENSE','LICENCE','COPYING','NOTICE'))]
        if package.get('license_file'):
            file = base/package['license_file']
            if file not in files and file.is_file(): files.append(file)
        notices.append(f"\n{'='*72}\n{package['name']} {package['version']} | {package.get('license')}\n{package.get('repository') or ''}\n")
        for file in sorted(files):
            if file.stat().st_size > 256*1024: raise RuntimeError(f'oversized legal text: {file.name}')
            notices.append(f'\n--- {file.name} ---\n'+file.read_text(encoding='utf-8', errors='replace'))
    sums=[]
    for platform, binaries, extension, features in [
        ('windows-x86_64',a.cpu_bin,'.exe','transcription'),
        ('windows-x86_64-cuda',a.cuda_bin,'.exe','transcription,cuda'),
        ('linux-x86_64',a.linux_bin,'','transcription')]:
        name=f'xeno-rt-{version}-{platform}'; stage=a.out/name; stage.mkdir()
        for binary in ['xrt-cli','xrt-server']:
            shutil.copy2(binaries/(binary+extension),stage/(binary+extension))
            if not extension: (stage/binary).chmod(0o755)
        for file in ['README.md','LICENSE','NOTICE','CHANGELOG.md','RELEASE.md']:
            shutil.copy2(ROOT/file,stage/file)
        shutil.copy2(ROOT/f'docs/releases/{version}.md',stage/'RELEASE_NOTES.md')
        shutil.copy2(ROOT/'docs/AUDIO-PIPELINE.md',stage/'AUDIO-PIPELINE.md')
        (stage/'examples/audio').mkdir(parents=True)
        shutil.copy2(ROOT/'examples/audio/narrate_job.py',stage/'examples/audio/narrate_job.py')
        (stage/'THIRD-PARTY-LICENSES.txt').write_text(''.join(notices),encoding='utf-8')
        info={'version':version,'commit':commit,'platform':platform,'features':features,
              'build':'local clean locked build','hosted_attestation':False,
              'native_payload_bundled':False,'model_weights_bundled':False,
              'binaries':{binary+extension:hashlib.sha256((stage/(binary+extension)).read_bytes()).hexdigest() for binary in ['xrt-cli','xrt-server']}}
        (stage/'BUILD_INFO.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
        archive=a.out/(name+('.zip' if extension else '.tar.gz'))
        if extension:
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
                for file in sorted(stage.rglob('*')):
                    if file.is_file(): z.write(file,file.relative_to(a.out).as_posix())
        else:
            with tarfile.open(archive,'x:gz') as t: t.add(stage,arcname=name)
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        row=f'{digest}  {archive.name}\n';sums.append(row)
        archive.with_name(archive.name+'.sha256').write_text(row,encoding='ascii')
        print(row.strip())
    (a.out/'SHA256SUMS').write_text(''.join(sums),encoding='ascii')

if __name__ == '__main__': main()
