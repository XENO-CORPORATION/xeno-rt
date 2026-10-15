"""Load only hash-pinned upstream modules for independent local qualification."""
import hashlib
import sys
import types
import json
import struct
import math

HASHES = {
    'config': '1e885113d269d7f7c9cd6648f393eec7bbc6974be6974719e8493f012ed0bfd7',
    'layers': 'e4704b4fde7906bf021d6696c4132458b4bc52c836d22da702275252683540bc',
    'backbone': 'dd969c85df49c0ec6293b726cbc226e55ca9d2b85cce234a2acf1df340237517',
    'moe': '3b3981c3fa6849942db1a58c9aa9f5085148b3d98427ca7ef1f8cd81e84e2d2c',
    'fp8_moe': 'c30de427f2ec31beaff7f95e8e15cdd004fcdde005f26d23e1a04593c228b40f',
    'tower': '71f47e85bcc51c1b644fd29d863598bf87208dbd86d479791d699629b7b92d8c',
}


def load(source):
    for name, digest in HASHES.items():
        assert hashlib.sha256((source / f'{name}.py').read_bytes()).hexdigest() == digest
    for name in ['fish_speech', 'fish_speech.models', 'fish_speech.models.text2semantic', 'fish_speech.models.text2semantic.fish_omni']:
        module = types.ModuleType(name)
        module.__path__ = [str(source)] if name.endswith('fish_omni') else []
        sys.modules[name] = module


class TensorReader:
    """Read tensor bytes without torch.Storage mmap slicing on Windows.

    The local safetensors 0.5.3/Torch 2.14 reader access-violated in
    Storage.__getitem__, and another mapping failed Windows commit admission.
    Explicit bounded reads avoid large extra mappings; operators, tensor dtype
    and trained bytes are unchanged. This does not diagnose the binding's bug.
    The Rust runtime does not use this qualification-only reader.
    """

    def __init__(self, path):
        self.file = path.open('rb')
        size = path.stat().st_size
        length, = struct.unpack('<Q', self.file.read(8))
        assert length <= 32 * 1024 * 1024 and 8 + length <= size
        self.header = json.loads(self.file.read(length))
        self.start = 8 + length
        self.end = size
        edges = []
        for name, tensor in self.header.items():
            if name == '__metadata__':
                continue
            begin, end = tensor['data_offsets']
            assert 0 <= begin <= end <= size - self.start
            width = {'BF16': 2, 'F32': 4, 'F8_E4M3': 1}[tensor['dtype']]
            assert all(isinstance(v, int) and v >= 0 for v in tensor['shape'])
            assert math.prod(tensor['shape']) * width == end - begin
            edges.append((begin, end))
        cursor = 0
        for begin, end in sorted(edges):
            assert begin == cursor
            cursor = end
        assert cursor == size - self.start

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def get_tensor(self, name):
        import torch
        tensor = self.header[name]
        begin, end = tensor['data_offsets']
        assert 0 <= begin <= end <= self.end - self.start
        self.file.seek(self.start + begin)
        data = bytearray(self.file.read(end - begin))
        assert len(data) == end - begin
        dtype = {'BF16': torch.bfloat16, 'F32': torch.float32,
                 'F8_E4M3': torch.float8_e4m3fn}[tensor['dtype']]
        return torch.frombuffer(data, dtype=dtype).reshape(tensor['shape'])
