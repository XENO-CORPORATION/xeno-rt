"""Format workspace members while leaving pinned vendored dependencies intact."""
import argparse
import json
import pathlib
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
metadata = json.loads(subprocess.run(
    ['cargo', 'metadata', '--no-deps', '--locked', '--format-version', '1'],
    cwd=root, check=True, capture_output=True).stdout)
members = set(metadata['workspace_members'])
command = ['cargo', 'fmt']
for package in metadata['packages']:
    if package['id'] in members:
        command.extend(['--package', package['name']])
if args.check:
    command.extend(['--', '--check'])
subprocess.run(command, cwd=root, check=True)
