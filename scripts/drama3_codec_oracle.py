"""Independent PyTorch operator reference for the tensor-only DAC decode graph.

Uses standard PyTorch kernels, not the Rust implementation. This is a decoded
tensor-graph comparison, not a claim of executing Fish's packaged serving stack.
"""
import argparse,json,pathlib,hashlib
import torch
from drama3_codec_reference import TensorGraph

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--codec',type=pathlib.Path,required=True);p.add_argument('--codes',type=pathlib.Path,required=True);p.add_argument('--native',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args();torch.set_num_threads(8);torch.backends.mkldnn.enabled=False
codes=torch.tensor(json.loads(a.codes.read_text()),dtype=torch.long)
with torch.inference_mode():reference=TensorGraph(a.codec).decode(codes)
native=torch.frombuffer(bytearray(a.native.read_bytes()),dtype=torch.float32)
assert native.shape==reference.shape
error=(native-reference).abs();snr=10*torch.log10(reference.square().sum()/((native-reference).square().sum().clamp(min=1e-20)))
report={'scope':'native codec vs independent PyTorch operator reference','admitted':False,
 'samples':reference.numel(),'max_absolute_error':float(error.max()),'rms_error':float((native-reference).square().mean().sqrt()),
 'snr_db':float(snr),'torch':torch.__version__,'codec_sha256':hashlib.file_digest((a.codec/'model.safetensors').open('rb'),'sha256').hexdigest(),
 'passed':bool(error.max()<1e-3 and snr>60)}
a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));assert report['passed'], 'native codec waveform comparison failed'
