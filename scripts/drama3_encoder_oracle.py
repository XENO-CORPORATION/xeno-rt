"""Compare native trained encoder codes with independent PyTorch operators."""
import argparse
import hashlib
import json
import pathlib
import struct
import torch
from drama3_codec_reference import TensorGraph

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--codec', type=pathlib.Path, required=True)
p.add_argument('--wav', type=pathlib.Path, required=True)
p.add_argument('--native', type=pathlib.Path, required=True)
p.add_argument('--out', type=pathlib.Path, required=True)
a = p.parse_args()
torch.set_num_threads(8)
torch.backends.mkldnn.enabled = False
raw = a.wav.read_bytes()
assert raw[:4] == b'RIFF' and raw[8:12] == b'WAVE'
position = 12
fmt = data = None
while position + 8 <= len(raw):
    size = struct.unpack_from('<I', raw, position + 4)[0]
    chunk = raw[position + 8:position + 8 + size]
    if raw[position:position + 4] == b'fmt ': fmt = chunk
    if raw[position:position + 4] == b'data': data = chunk
    position += 8 + size + (size & 1)
assert fmt is not None and data is not None
encoding, channels, rate = struct.unpack_from('<HHI', fmt)
assert encoding == 3 and channels == 1 and rate == 44100 and struct.unpack_from('<H', fmt, 14)[0] == 32
samples = torch.frombuffer(bytearray(data), dtype=torch.float32)
with torch.inference_mode():
    expected = TensorGraph(a.codec).encode(samples)
actual = torch.tensor(json.loads(a.native.read_text()), dtype=torch.long)
assert actual.shape == expected.shape
matches = actual == expected
report = {'scope': 'native reference encoder vs independent PyTorch tensor graph', 'admitted': False,
          'frames': actual.shape[0], 'codes': actual.numel(), 'mismatches': int((~matches).sum()),
          'input_sha256': hashlib.sha256(raw).hexdigest(), 'torch': torch.__version__, 'passed': bool(matches.all())}
a.out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
assert report['passed'], 'native encoder acoustic code mismatch'
