"""Own a dedicated server and retain real-model regression/performance evidence."""
import argparse, base64, hashlib, http.client, json, os, pathlib, socket
import statistics, subprocess, time, urllib.error, urllib.request, uuid

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',type=pathlib.Path,required=True)
    p.add_argument('--native',type=pathlib.Path,required=True)
    p.add_argument('--cache',type=pathlib.Path,required=True)
    p.add_argument('--reference',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--benchmark-only',action='store_true')
    p.add_argument('--client',type=pathlib.Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    for key in ['XRT_AUDIO_MODEL_DIR','XRT_AUDIO_ASR_DIR','XENO_RT_WHISPER_DIR','ORT_DYLIB_PATH']:
        env.pop(key,None)
    env.update(XRT_CACHE_DIR=str(a.cache),ORT_DYLIB_PATH=str(a.native/'onnxruntime.dll'),
        XRT_AUDIO_VOICES_DIR=str(a.output/'voices'),XRT_AUDIO_JOBS_DIR=str(a.output/'jobs'))
    # This is qualification of a separate native payload, not a packaged claim.
    env['PATH']=str(a.native)+os.pathsep+env['PATH']
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    checks=[];times=[];outputs=[]
    def call(path,body=None,method=None,key=None):
        headers={'Content-Type':'application/json'}
        if key:headers['Idempotency-Key']=key
        req=urllib.request.Request(base+path,method=method or ('GET' if body is None else 'POST'),
            headers=headers,data=None if body is None else json.dumps(body).encode())
        try:
            with op.open(req,timeout=900) as r:return r.status,json.load(r)
        except urllib.error.HTTPError as e:return e.code,json.load(e)
    def check(name,ok):
        checks.append({'name':name,'passed':bool(ok)});print(('PASS ' if ok else 'FAIL ')+name,flush=True)
        if not ok:raise AssertionError(name)
    log=(a.output/'server.log').open('wb')
    process=subprocess.Popen([str(a.server.resolve()),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        deadline=time.monotonic()+30
        while True:
            if process.poll() is not None:raise RuntimeError('server exited; inspect server.log')
            try:
                if call('/v1/audio/status')[0]==200:break
            except OSError:pass
            if time.monotonic()>deadline:raise TimeoutError('startup')
            time.sleep(.1)
        text='The speaker describes a life of truth and justice. The heart is weighed against a feather, and the scales reveal what those claims are worth.'
        payload={'input':text,'voice_b64':base64.b64encode(a.reference.read_bytes()).decode(),
            'device':a.device,'response_format':'json','word_check':False}
        for i in range(4):
            started=time.monotonic();code,out=call('/v1/audio/speech',payload);elapsed=time.monotonic()-started
            if code!=200:print(out,flush=True)
            check(f'synthesis {i}',code==200 and out['provider'].startswith(a.device))
            times.append(elapsed);outputs.append(hashlib.sha256(base64.b64decode(out['audio_b64'])).hexdigest())
            print(f'BENCH {i} {elapsed:.3f}s',flush=True)
        if a.benchmark_only:return
        check('word-check-off does not load Whisper',call('/v1/audio/status')[1]['asr_provider'] is None)
        # Valid JSON reaches native synthesis but fails reference validation.
        short=base64.b64encode(__import__('struct').pack('<4sI4s4sIHHIIHH4sI',b'RIFF',38,b'WAVE',b'fmt ',16,1,1,24000,48000,2,16,b'data',2)+b'\0\0').decode()
        code,_=call('/v1/audio/speech',{**payload,'voice_b64':short})
        check('invalid reference fails',code>=400)
        check('request error retains healthy model',call('/v1/audio/status')[1]['load_verified'])
        conn=http.client.HTTPConnection('127.0.0.1',port,timeout=40)
        conn.putrequest('POST','/v1/audio/transcriptions');conn.putheader('Content-Type','multipart/form-data; boundary=slow')
        conn.putheader('Content-Length','100000');conn.endheaders()
        conn.send(b'--slow\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\nRIFF')
        try:
            time.sleep(.25)
            check('stalled upload does not own inference admission',not call('/v1/audio/status')[1]['busy'])
            check('unload succeeds while upload stalls',call('/v1/audio/unload',{})[0]==200)
            check('stalled upload has bounded timeout',conn.getresponse().status==408)
        finally:conn.close()
        # Exercise durable jobs with the recognizer path deliberately absent.
        key='qualification-'+uuid.uuid4().hex
        code,job=call('/v1/audio/jobs',payload,key=key)
        check('word-check-off job admitted',code==202)
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            _,job=call('/v1/audio/jobs/'+job['id'])
            if job['status'] in ['succeeded','failed','cancelled','interrupted']:break
            time.sleep(.2)
        check('word-check-off durable job succeeds',job['status']=='succeeded')
        if a.client:
            script=a.output/'script.txt';script.write_text(text,encoding='utf-8')
            result=subprocess.run([__import__('sys').executable,str(a.client),str(a.reference),str(script),str(a.output/'client.wav'),
                '--base',base,'--device',a.device],capture_output=True,text=True,timeout=900)
            (a.output/'client.log').write_text(result.stdout+result.stderr,encoding='utf-8')
            check('maintained client produces a WAV',result.returncode==0 and (a.output/'client.wav').read_bytes()[:4]==b'RIFF')
    finally:
        process.terminate()
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        log.close()
        (a.output/'evidence.json').write_text(json.dumps({'server':str(a.server),
            'binary_sha256':hashlib.sha256(a.server.read_bytes()).hexdigest(),'device':a.device,'checks':checks,
            'seconds':times,'warm_median_seconds':statistics.median(times[1:]) if len(times)>1 else None,
            'output_sha256':outputs},indent=2),encoding='utf-8')

if __name__=='__main__':main()
