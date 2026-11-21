"""Real-model lifecycle gate for a DEDICATED loopback audio test server.

Mutates model lifetime (unload/drain), never a user's voices. Run last: drain
intentionally refuses new work until restart. Every native prerequisite must
exist; missing models never skip this gate.
"""
import argparse
import base64
import concurrent.futures
import http.client
import json
from pathlib import Path
import socket
import time
import urllib.error
import urllib.parse
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--reference', type=Path, required=True)
    a = p.parse_args()
    url = urllib.parse.urlsplit(a.base)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        p.error('a dedicated loopback server is required')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = {'input': 'The speaker describes a life of truth and justice. The heart is weighed against a feather, and the scales reveal what those claims are worth.',
               'voice_b64': base64.b64encode(a.reference.read_bytes()).decode(), 'device': 'cuda', 'response_format': 'json'}
    def call(path, body=None):
        request = urllib.request.Request(a.base + path, method='GET' if body is None else 'POST',
            data=None if body is None else json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        try:
            with opener.open(request, timeout=600) as r:return r.status, json.load(r)
        except urllib.error.HTTPError as e:return e.code, json.load(e)
    def check(name, ok):
        print(('PASS ' if ok else 'FAIL ') + name, flush=True)
        if not ok:raise AssertionError(name)
    def idle(max_seconds=60):
        deadline=time.monotonic()+max_seconds
        while time.monotonic()<deadline:
            code,state=call('/v1/audio/status')
            if code==200 and not state['busy']:return state
            time.sleep(.1)
        raise AssertionError('native worker failed to release admission')

    check('unload before loading is idempotent', call('/v1/audio/unload', {})[0]==200)
    code,result=call('/v1/audio/speech', payload)
    if code!=200:print(result,flush=True)
    check('warm synthesis succeeds',code==200)
    check('real loaded status',call('/v1/audio/status')[1]['load_verified'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        timed=pool.submit(call,'/v1/audio/speech',{**payload,'timeout_seconds':1})
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and not call('/v1/audio/status')[1]['busy']:time.sleep(.05)
        check('unload refuses active native work',call('/v1/audio/unload',{})[0]==409)
        code,result=timed.result(timeout=60)
        check('deadline returns cancellation not partial audio',code==408 and 'cancel' in result['error']['message'])
    check('cancel clears cached model allocations',not idle()['load_verified'])

    code,result=call('/v1/audio/speech',payload)
    check('fresh synthesis after cancellation succeeds',code==200)
    connection=http.client.HTTPConnection(url.hostname,url.port,timeout=10)
    connection.request('POST','/v1/audio/speech',json.dumps(payload),{'Content-Type':'application/json'})
    deadline=time.monotonic()+10
    while time.monotonic()<deadline and not call('/v1/audio/status')[1]['busy']:time.sleep(.05)
    check('disconnect test entered inference',call('/v1/audio/status')[1]['busy'])
    connection.sock.shutdown(socket.SHUT_RDWR)
    connection.close()
    check('disconnect cancels worker and unloads sessions',not idle()['load_verified'])
    check('unload succeeds after cancellation',call('/v1/audio/unload',{})[0]==200)
    check('drain acknowledged',call('/v1/audio/drain',{})[1]['draining'])
    check('drained server rejects new work',call('/v1/audio/speech',payload)[0]==503)
    print('all lifecycle gates passed',flush=True)


if __name__=='__main__':main()
