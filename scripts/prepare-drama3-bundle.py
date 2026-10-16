"""Prepare a private, revision-bound install manifest; do not advertise support.

Large-file SHA-256 values come from immutable upstream LFS metadata. Small
files are read with bounds and hashed. The existing Rust bundle installer owns
the actual large downloads, resume, verification and activation.
"""
import argparse
import hashlib
import json
import pathlib
import urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=pathlib.Path,required=True)
args=parser.parse_args()
pin=json.loads((ROOT/'reference/audio/drama3/provenance.json').read_text(encoding='utf-8-sig'))
assert pin['revision']=='e0a9218aa48888270f06ef1ed9ea67622c0e17d2'
files=[]
for record in pin['files']:
    if record['path'] in ['.gitattributes','README.md']:
        continue
    size=record['sizeBytes']; checksum=record['sha256']
    if not checksum:
        if size>32*1024*1024: raise ValueError('unhashed file exceeds metadata bound')
        with urllib.request.urlopen(record['source'],timeout=90) as response:
            data=response.read(size+1)
        if len(data)!=size: raise ValueError('immutable file size mismatch')
        checksum=hashlib.sha256(data).hexdigest()
    files.append({'path':record['path'],'size_bytes':size,'sha256':checksum,'source':record['source']})
manifest={'schema_version':2,'id':'fish-drama3-native-dev','revision':pin['revision'],'kind':'model',
 'family':'fish-omni-drama3','tasks':['speech.experimental-native-development'],
 'minimum_runtime':'0.4.0','platforms':['any'],'backends':['cpu','cuda'],
 'entrypoints':{'config':'config.json','tokenizer':'tokenizer.json','weights':'model.safetensors.index.json','codec':'codec.pth'},
 'dependencies':[],'license':{'spdx':'LicenseRef-Fish-Audio-Community-1.0','evidence':next(f['source'] for f in files if f['path']=='LICENSE'),'files':['LICENSE']},'files':files}
args.out.parent.mkdir(parents=True,exist_ok=True)
manifest['files'].sort(key=lambda f:f['path'])
canonical=json.dumps(manifest,separators=(',',':'),sort_keys=True).encode()
args.out.write_bytes(canonical+b'\n')
print(json.dumps({'manifest':str(args.out),'bytes':sum(f['size_bytes'] for f in files),'files':len(files),
                  'digest':hashlib.sha256(canonical).hexdigest(),
                  'admitted':False,'license':'Fish Audio Community License v1.0'}))
