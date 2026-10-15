"""Exercise word-verified narration through an owned installed server artifact.

The expected executable digest must come from the inspected candidate inventory.
No build, provisioning or publication occurs here. Only owned children stop.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
for name in ['server', 'config', 'checker-cache', 'corpus', 'out']:
    p.add_argument('--' + name, type=pathlib.Path, required=True)
p.add_argument('--artifact-sha256', required=True)
a = p.parse_args()
with a.server.open('rb') as artifact:
    digest = hashlib.file_digest(artifact, 'sha256').hexdigest()
assert digest == a.artifact_sha256.lower(), 'installed executable differs from inspected inventory'
assert a.config.stat().st_size <= 32768 and a.corpus.stat().st_size <= 1024 * 1024
config = json.loads(a.config.read_text())
assert config['word_check_enabled'] and config['numerics'] == 'bf16_w8a8'
corpus = json.loads(a.corpus.read_text())
assert 1 <= len(corpus['cases']) * len(corpus['seeds']) <= 32
assert all(re.fullmatch(r'[A-Za-z0-9-]{1,64}', case['id']) for case in corpus['cases'])
a.out.mkdir(exist_ok=False)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
env = os.environ.copy()
env.update(HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9',
           RAYON_NUM_THREADS='8', XRT_AUDIO_JOBS_DIR=str((a.out / 'jobs').resolve()),
           XRT_AUDIO_VOICES_DIR=str((a.out / 'voices').resolve()), XENO_HOME=str((a.out / 'xeno-home').resolve()))
for name in ['XRT_EXTERNAL_BASE_URL', 'XRT_EXTERNAL_API_KEY', 'XRT_EXTERNAL_MODEL', 'XENO_RT_WHISPER_DIR']:
    env.pop(name, None)
process, log, base, auth = None, None, '', {}
records = []


def save(name, value):
    (a.out / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')


def call(path, payload=None, headers=None):
    request = urllib.request.Request(base + path, None if payload is None else json.dumps(payload).encode(),
                                     {'Content-Type':'application/json', **(headers or {})})
    try:
        with opener.open(request, timeout=180) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as response:
        return response.code, dict(response.headers), response.read()


def cap(name, args):
    status, _, body = call('/capabilities/' + name, {'args':args}, auth)
    result = json.loads(body)
    assert status == 200 and result['ok'], (status, result)
    return result['data']


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


def start(name, checker_cache):
    global process, log, base, auth
    child_env = dict(env, XRT_CACHE_DIR=str(checker_cache.resolve()))
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    log = (a.out / (name + '.server.log')).open('wb')
    process = subprocess.Popen([str(a.server.resolve()), '--port', str(port), '--drama3-config', str(a.config.resolve()), '--agent-capabilities'],
                               env=child_env, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    deadline = time.monotonic() + 30
    descriptor_path = a.out / 'xeno-home/run/platform/capability-servers/rt.json'
    while time.monotonic() < deadline:
        assert process.poll() is None, 'owned server exited during startup'
        try:
            descriptor = json.loads(descriptor_path.read_text())
            assert descriptor['pid'] == process.pid and descriptor['baseUrl'] == base
            if json.loads(call('/health')[2])['ready']:
                auth = {'Authorization':'Bearer ' + descriptor['token']}
                save(name + '.start', {'pid':process.pid, 'server_sha256':digest,
                                      'native':cap('rt.audio.status',{})})
                return
        except (OSError, ValueError, AssertionError):
            time.sleep(.1)
    raise TimeoutError('owned capability binding did not become ready')


def submit(key, request):
    return cap('rt.audio.start', {'idempotency_key':key, 'speech':request})


def wait_job(job_id, seconds=1800):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        job = cap('rt.audio.job_get', {'id':job_id})
        if job['status'] not in ['queued','running','cancelling']:
            return job
        time.sleep(.5)
    raise TimeoutError('owned narration job exceeded its bounded wait')


try:
    start('missing-checker', a.out / 'empty-checker-cache')
    request = {'input':'The evidence remains clear.', 'word_check':{'spoken_text':'The evidence remains clear.','max_attempts':1}}
    status, _, body = call('/v1/xrt/audio/native/speech', request)
    missing = json.loads(body)
    assert status == 503 and missing['error']['code'] == 'word_checker_unavailable', (status, missing)
    assert not cap('rt.audio.status',{})['native']['loaded']
    save('missing-checker-refusal', {'http_status':status,'response':missing,'model_loaded':False})
    stop()
    start('verified', a.checker_cache)
    for case in corpus['cases']:
        for seed in corpus['seeds']:
            name = case['id'] + '-' + str(seed)
            request = {'input':case['text'],'seed':seed,'max_frames':256,
                       'word_check':{'spoken_text':case['spoken'],'max_attempts':2}}
            started = time.monotonic()
            job = submit(name, request)
            assert submit(name, request)['id'] == job['id'], 'duplicate submission created another job'
            terminal = wait_job(job['id'])
            record = {'id':name,'admitted':False,'job':terminal,'elapsed_seconds':time.monotonic()-started,
                      'spoken_text':case['spoken']}
            if terminal['status'] == 'succeeded':
                metadata = terminal['result_metadata']
                assert metadata['x-xrt-word-check'] == 'verified'
                assert 1 <= int(metadata['x-xrt-attempts']) <= 2
                assert re.fullmatch(r'[a-f0-9]{64}', metadata['x-xrt-word-check-digest'])
                status, headers, wav = call(terminal['result_url'])
                assert status == 200 and wav[:4] == b'RIFF'
                (a.out / (name + '.wav')).write_bytes(wav)
                record['wav_sha256'] = hashlib.sha256(wav).hexdigest()
            record['passed'] = terminal['status'] == 'succeeded'
            save(name, record)
            records.append(record)
            print(json.dumps({'id':name,'passed':record['passed'],'status':terminal['status'],
                              'attempts':terminal.get('result_metadata',{}).get('x-xrt-attempts') if terminal.get('result_metadata') else None}), flush=True)
    wrong = submit('wrong-spoken-script', {'input':'The evidence remains clear.','seed':42,'max_frames':128,
                   'word_check':{'spoken_text':'Do not change the spoken meaning.','max_attempts':1}})
    refused = wait_job(wrong['id'])
    assert refused['status'] == 'failed' and 'quality_rejected' in refused['error'] and refused['result_url'] is None
    assert call('/v1/audio/jobs/' + wrong['id'] + '/result')[0] == 409
    save('wrong-script-refusal', refused)
    repeated = submit('cancel-second-attempt', {'input':'The evidence remains clear.','max_frames':1,
                      'word_check':{'spoken_text':'The evidence remains clear.','max_attempts':2}})
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = cap('rt.audio.job_get',{'id':repeated['id']})
        if state['progress_attempt'] == 2:
            break
        assert state['status'] in ['queued','running'], state
        time.sleep(.1)
    else:
        raise TimeoutError('second attempt was not observed')
    cap('rt.audio.job_cancel',{'id':repeated['id']})
    cancelled = wait_job(repeated['id'])
    assert cancelled['status'] == 'cancelled' and cancelled['result_url'] is None
    save('second-attempt-cancellation',cancelled)
    assert cap('rt.audio.unload',{})['unloaded']
    save('summary',{'passed':all(v['passed'] for v in records),'admitted':False,
         'scope':'installed development artifact, English word-verified narration and bounded retries; not perceptual quality or production admission',
         'server_sha256':digest,'cases':len(records),'passed_cases':sum(v['passed'] for v in records),
         'negative_controls':['missing checker refused before native load','wrong spoken script publishes no result','cancellation during second attempt'],
         'weights_published':False})
    assert all(v['passed'] for v in records), 'narration corpus has failed jobs; retained failures are not admission'
finally:
    stop()
