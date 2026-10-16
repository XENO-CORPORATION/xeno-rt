"""Refuse changes, omissions or additions in the pinned first-party crate."""
import hashlib
import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
source = json.loads((root / 'vendor/xenosystem-primitives.source.json').read_text())
crate = root / 'vendor/xenosystem-primitives'
actual = {path.relative_to(crate).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in crate.rglob('*') if path.is_file()}
assert actual == source['files'], 'shared primitives snapshot differs from its pinned source inventory'
assert source['commit'] == '9e7f7070608f42e7010fbd0fee3e38b60fd5a3f7'
print(f"Verified {len(actual)} original shared-crate files at {source['commit']}")
