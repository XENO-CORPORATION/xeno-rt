"""Exercise an owned native server over HTTP; no model/network provisioning.

Only the child created here is stopped. Receipts distinguish development
service behavior from model quality, installed packaging and public admission.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
for name in ['server', 'config', 'out']:
    p.add_argument('--' + name, type=pathlib.Path, required=True)
p.add_argument('--expect-host-refusal', action='store_true')
p.add_argument('--cpu-only', action='store_true', help='build without the CUDA feature; GPU budgets must be zero')
p.add_argument('--artifact-sha256', help='test an installed artifact instead of building; require its independently supplied digest')
a = p.parse_args()
a.out.mkdir(exist_ok=False)
env = os.environ.copy()
env.update(HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9')
env.update(XRT_BACKEND='auto', XRT_AUDIO_JOBS_DIR=str((a.out / 'jobs').resolve()),
           XRT_AUDIO_VOICES_DIR=str((a.out / 'voices').resolve()))
for name in ['XRT_EXTERNAL_BASE_URL', 'XRT_EXTERNAL_API_KEY', 'XRT_EXTERNAL_MODEL']:
    env.pop(name, None)
source_root = pathlib.Path(__file__).resolve().parent.parent


def source_digest():
    files = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                           cwd=source_root, check=True, capture_output=True).stdout.decode().split('\0')
    digest = hashlib.sha256()
    for name in sorted(set(files) - {''}):
        path = source_root / name
        if path.suffix in ['.rs', '.cu', '.toml', '.lock', '.py'] and path.is_file():
            digest.update(name.encode() + b'\0' + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


if a.artifact_sha256:
    with a.server.open('rb') as artifact:
        actual = hashlib.file_digest(artifact, 'sha256').hexdigest()
    assert actual == a.artifact_sha256.lower(), 'installed artifact checksum differs from the expected digest'
    (a.out / 'build.json').write_text(json.dumps({'source_built':False, 'expected_artifact_sha256':actual}, indent=2) + '\n')
else:
    assert a.server.parent.name == 'release' and a.server.name in ['xrt-server', 'xrt-server.exe'], \
        'source builds require <target-dir>/release/xrt-server; use --artifact-sha256 for an installed artifact'
    before = source_digest()
    command = ['cargo', 'build', '--release', '--locked', '-p', 'xrt-server', '--features',
               'drama3-native' if a.cpu_only else 'drama3-native,cuda', '--target-dir', str(a.server.resolve().parent.parent)]
    with (a.out / 'build.log').open('wb') as output:
        subprocess.run(command, cwd=source_root, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    assert source_digest() == before, 'source changed while building; refuse a stale-artifact proof'
    (a.out / 'build.json').write_text(json.dumps({'source_built':True, 'source_sha256':before, 'command':command}, indent=2) + '\n')
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
receipts = []
process = None
log = None
base = ''


def save(name, value):
    (a.out / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')


def call(path, payload=None):
    start = time.monotonic()
    request = urllib.request.Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with opener.open(request, timeout=600) as response:
            status, headers, body = response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as response:
        status, headers, body = response.code, dict(response.headers), response.read()
    return status, {k.lower(): v for k, v in headers.items()}, body, time.monotonic() - start


def read_json(path):
    status, _, body, _ = call(path)
    assert status == 200, (path, status, body[:512])
    return json.loads(body)


def until(predicate, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.1)
    raise TimeoutError('owned server state did not settle')


def stop():
    global process, log
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        process = None
    if log is not None:
        log.close()
        log = None


def start(name):
    global process, log, base
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    log = (a.out / (name + '.server.log')).open('wb')
    process = subprocess.Popen([str(a.server.resolve()), '--port', str(port), '--drama3-config', str(a.config.resolve())],
                               env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    def ready():
        assert process.poll() is None, 'owned server exited during startup'
        try:
            return read_json('/v1/xrt/audio/native/status')['configured']
        except OSError:
            return False
    until(ready, 30)
    save(name + '.start', {'pid': process.pid, 'server_sha256': hashlib.file_digest(a.server.open('rb'), 'sha256').hexdigest(),
                         'config_sha256': hashlib.sha256(a.config.read_bytes()).hexdigest(),
                         'native': read_json('/v1/xrt/audio/native/status'), 'runtime': read_json('/v1/runtime/status')})


def speech(name, payload, expected=200):
    status, headers, body, seconds = call('/v1/xrt/audio/native/speech', payload)
    record = {'case': name, 'http_status': status, 'seconds': seconds, 'headers': headers, 'input': payload,
              'admitted': False}
    if status == 200:
        assert headers['content-type'] == 'audio/wav' and headers['x-xrt-admitted'] == 'false'
        assert body[:4] == b'RIFF' and body[8:12] == b'WAVE'
        (a.out / (name + '.wav')).write_bytes(body)
        record['wav_sha256'] = hashlib.sha256(body).hexdigest()
    else:
        record['error'] = json.loads(body)
    save(name, record)
    assert status == expected, record
    receipts.append(record)
    return record


plain = {'input': 'The evidence remains clear.', 'seed': 42, 'max_frames': 128}
try:
    start('first')
    assert read_json('/v1/audio/status')['model'] == 'chatterbox-multilingual-v3'
    assert all('drama' not in str(v).lower() for v in read_json('/v1/models')['data'])
    if a.expect_host_refusal:
        rejected = speech('host-headroom-refusal', plain, 503)
        assert rejected['error']['error']['code'] == 'host_headroom'
        assert not read_json('/v1/xrt/audio/native/status')['loaded']
        runtime = read_json('/v1/runtime/status')
        assert runtime['gpu_resource']['arena_allocated_bytes'] == 0
        assert call('/v1/xrt/audio/native/unload', {})[0] == 200
        save('summary', {'passed':True, 'admitted':False, 'scope':'live Windows commit refusal and release of native resources; not full service qualification', 'runtime':runtime})
        raise SystemExit(0)
    first = speech('plain', plain)
    if os.name == 'nt':
        memory = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            f'Get-Process -Id {process.pid} | Select-Object Id,WorkingSet64,PrivateMemorySize64 | ConvertTo-Json'],
            check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        save('resident-process-snapshot', {'admitted':False, 'scope':'one post-generation sample, not peak memory', 'process':json.loads(memory.stdout)})
        modules = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            f'(Get-Process -Id {process.pid}).Modules | Select-Object -ExpandProperty ModuleName | ConvertTo-Json'],
            check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        names = json.loads(modules.stdout)
        forbidden = [name for name in names if any(tag in name.lower() for tag in ['onnxruntime', 'python', 'torch'])]
        save('native-module-audit', {'admitted':False, 'pid':process.pid, 'forbidden_modules':forbidden,
                                    'cuda_modules':[name for name in names if any(tag in name.lower() for tag in ['nvcuda', 'nvrtc', 'cublas'])]})
        assert not forbidden, 'native speech loaded an external inference engine'
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(speech, 'cancelled', dict(plain, input='The runtime must finish every sentence and preserve every spoken word.', max_frames=256), 408)
        until(lambda: read_json('/v1/xrt/audio/native/status')['active'])
        assert call('/v1/xrt/audio/native/unload', {})[0] == 409
        assert call('/v1/audio/unload', {})[0] == 409
        assert call('/v1/xrt/audio/native/speech', plain)[0] == 429
        assert json.loads(call('/v1/xrt/audio/native/cancel', {})[2])['cancel_requested']
        pending.result(timeout=60)
    speech('frame-limit', dict(plain, max_frames=1), 422)
    speech('deadline', dict(plain, timeout_seconds=1), 408)
    with socket.create_connection(('127.0.0.1', int(base.rsplit(':', 1)[1])), timeout=5) as client:
        data = json.dumps(plain).encode()
        client.sendall(b'POST /v1/xrt/audio/native/speech HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: '
                       + str(len(data)).encode() + b'\r\n\r\n' + data)
        until(lambda: read_json('/v1/xrt/audio/native/status')['active'])
        disconnect_status = read_json('/v1/xrt/audio/native/status')
    until(lambda: not read_json('/v1/xrt/audio/native/status')['active'])
    save('disconnect', {'admitted': False, 'active_before_disconnect': disconnect_status, 'settled': read_json('/v1/xrt/audio/native/status')})
    after = speech('plain-after-cancel-and-limit', plain)
    assert first['wav_sha256'] == after['wav_sha256'], 'request state leaked across requests'
    speech('narration', {'input': 'The runtime works offline. Keep the door closed, and do not open gate seventeen until the test is complete.', 'seed':42, 'max_frames':256})
    assert call('/v1/xrt/audio/native/unload', {})[0] == 200
    status = read_json('/v1/xrt/audio/native/status')
    runtime = read_json('/v1/runtime/status')
    save('unloaded', {'native': status, 'runtime': runtime})
    assert not status['loaded'] and runtime['gpu_resource']['arena_allocated_bytes'] == 0
    stop()
    start('restart')
    restarted = speech('plain-after-restart', plain)
    assert first['wav_sha256'] == restarted['wav_sha256'], 'fresh-process reproducibility failed'
    assert call('/v1/audio/drain', {})[0] == 200
    speech('drained', plain, 503)
    assert call('/v1/xrt/audio/native/unload', {})[0] == 200
    save('summary', {'passed': True, 'admitted': False, 'scope': 'owned development server HTTP lifecycle, not installed package or model admission',
                     'cases': [v['case'] for v in receipts], 'waveform_sha256': first['wav_sha256']})
finally:
    stop()
