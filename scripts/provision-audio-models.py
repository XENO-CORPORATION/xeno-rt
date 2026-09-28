"""Pinned audio model installer. Dry-run by default; --install downloads.

Uses stdlib only. Verifies bytes+SHA256, resumes with HTTP Range, never replaces
an existing mismatched model. Retains partials for retry; no recursive cleanup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def checked_entries(manifest):
    if manifest.get('schema_version') != 1 or not re.fullmatch(r'[0-9a-f]{40}', manifest.get('revision', '')):
        raise ValueError('invalid manifest schema or revision')
    seen = set()
    for entry in manifest['files']:
        path = PurePosixPath(entry['path'])
        if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in entry['path'] or ':' in entry['path']:
            raise ValueError('unsafe model path')
        if entry['path'].casefold() in seen:
            raise ValueError('duplicate model path')
        seen.add(entry['path'].casefold())
        if type(entry['bytes']) is not int or entry['bytes'] <= 0 or not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
            raise ValueError('invalid model size/hash')
        url = urllib.parse.urlsplit(entry['url'])
        expected = '/' + manifest['repository'] + '/resolve/' + manifest['revision'] + '/' + entry['path']
        if url.scheme != 'https' or url.hostname != 'huggingface.co' or url.path != expected or url.query or url.fragment or url.username:
            raise ValueError('model URL must use the pinned upstream revision')
        yield entry


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified(path, entry):
    return path.is_file() and path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']


def no_links(path, root):
    current = path
    while True:
        if current.exists() or current.is_symlink():
            if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
                raise ValueError('refusing link/junction in model destination: ' + str(current))
        if current == root:
            break
        if current.parent == current:
            raise ValueError('destination escapes root')
        current = current.parent


def install(root, entry):
    target = root.joinpath(*PurePosixPath(entry['path']).parts)
    no_links(target, root)
    if target.exists():
        if not verified(target, entry):
            raise ValueError('existing model differs; choose another destination, not overwrite: ' + str(target))
        print('VERIFIED', entry['path'], flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial-' + entry['sha256'][:16])
    lock = target.with_name(target.name + '.download-lock')
    no_links(partial, root)
    # Cross-process exclusion; stale locks require operator inspection, never
    # guess that a slow multipart transfer or checksum is dead.
    with lock.open('x', encoding='ascii') as lease:
        lease.write(str(os.getpid()))
    try:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > entry['bytes']:
            raise ValueError('oversized partial: ' + str(partial))
        if offset < entry['bytes']:
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            request = urllib.request.Request(entry['url'], headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                # A server ignoring Range must not append a second full file.
                if offset and response.status != 206:
                    raise ValueError('server refused resume; preserve partial and use a fresh destination')
                if response.status not in (200, 206):
                    raise ValueError('unexpected download status')
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset or int(match[3]) != entry['bytes'] or int(match[2]) != entry['bytes'] - 1:
                        raise ValueError('invalid resume range')
                with partial.open('ab' if offset else 'xb') as output:
                    written = offset
                    while block := response.read(8 * 1024 * 1024):
                        written += len(block)
                        if written > entry['bytes']:
                            raise ValueError('download exceeds manifest size')
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
        if not verified(partial, entry):
            raise ValueError('partial checksum mismatch: ' + str(partial))
        if target.exists():
            raise ValueError('destination appeared during download; refusing overwrite')
        # Hard-link publication is atomic and fails if the target exists;
        # unlike os.replace it cannot overwrite a concurrent install.
        os.link(partial, target)
        partial.unlink()
        print('INSTALLED', entry['path'], flush=True)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['chatterbox-multilingual-v3', 'whisper-small-timestamped'], required=True)
    parser.add_argument('--destination', type=Path, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--install', action='store_true')
    actions.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'reference/audio' / (args.model + '.json')).read_text(encoding='utf-8'))
    entries = list(checked_entries(manifest))
    root = args.destination.absolute()
    for entry in entries:
        path = root.joinpath(*PurePosixPath(entry['path']).parts)
        no_links(path, root)
        if args.verify:
            if not verified(path, entry):
                raise ValueError('verification failed: ' + str(path))
            print('VERIFIED', entry['path'])
        elif args.install:
            install(root, entry)
        else:
            print('PLAN', entry['bytes'], entry['path'], entry['sha256'])
    print('TOTAL', sum(entry['bytes'] for entry in entries), 'bytes', flush=True)


if __name__ == '__main__':
    main()
