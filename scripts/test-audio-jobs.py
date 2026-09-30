"""Real durable-job proof against a dedicated loopback server.
Creates only unique job records, tests cancellation and idempotency, and retains
one completed job so the restart invocation can verify its persisted result.
"""
import argparse,base64,json,pathlib,time,urllib.request,urllib.error,urllib.parse,uuid

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--reference',type=pathlib.Path);p.add_argument('--resume-job');a=p.parse_args()
 if urllib.parse.urlsplit(a.base).hostname not in ('localhost','127.0.0.1','::1'):p.error('loopback test server required')
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 def call(method,path,body=None,key=None):
  headers={'Content-Type':'application/json'}
  if key:headers['Idempotency-Key']=key
  req=urllib.request.Request(a.base+path,method=method,headers=headers,data=None if body is None else json.dumps(body).encode())
  try:
   with op.open(req,timeout=180) as r:return r.status,json.load(r)
  except urllib.error.HTTPError as e:return e.code,json.load(e)
 def check(name,ok):
  print(('PASS ' if ok else 'FAIL ')+name,flush=True)
  if not ok:raise AssertionError(name)
 def wait(id):
  end=time.monotonic()+600
  while time.monotonic()<end:
   _,j=call('GET','/v1/audio/jobs/'+id)
   if j['status'] in ('succeeded','failed','cancelled','interrupted'):return j
   time.sleep(.5)
  raise AssertionError('job never reached terminal state')
 if a.resume_job:
  code,job=call('GET','/v1/audio/jobs/'+a.resume_job);check('completed job survives restart',code==200 and job['status']=='succeeded')
  code,result=call('GET',job['result_url']);check('result survives restart',code==200 and base64.b64decode(result['audio_b64'])[:4]==b'RIFF')
  check('explicit retention cleanup',call('DELETE','/v1/audio/jobs/'+a.resume_job)[0]==200);return
 if not a.reference:p.error('--reference required on first run')
 payload={'input':'The speaker describes a life of truth and justice. The heart is weighed against a feather, and the scales reveal what those claims are worth.','voice_b64':base64.b64encode(a.reference.read_bytes()).decode(),'device':'cuda','response_format':'json'}
 key='job-proof-'+uuid.uuid4().hex
 check('idempotency key required',call('POST','/v1/audio/jobs',payload)[0]==400)
 code,job=call('POST','/v1/audio/jobs',payload,key);check('submission returns job identity',code==202 and len(job['id'])==32);id=job['id']
 code,same=call('POST','/v1/audio/jobs',payload,key);check('same key returns same job',code==200 and same['id']==id)
 check('different request cannot reuse key',call('POST','/v1/audio/jobs',{**payload,'seed':21},key)[0]==409)
 code,canceljob=call('POST','/v1/audio/jobs',{**payload,'seed':22},key+'-cancel');check('second job queued',code==202)
 code,_=call('POST','/v1/audio/jobs/'+canceljob['id']+'/cancel',{});check('cancel accepted',code==200)
 check('cancelled job terminal',wait(canceljob['id'])['status']=='cancelled')
 check('cancelled job has no audio result',call('GET','/v1/audio/jobs/'+canceljob['id']+'/result')[0]==409)
 check('cancelled job can be deleted',call('DELETE','/v1/audio/jobs/'+canceljob['id'])[0]==200)
 done=wait(id)
 if done['status']!='succeeded':print(done,flush=True)
 check('durable speech job succeeds',done['status']=='succeeded')
 code,result=call('GET',done['result_url']);check('result URL returns real WAV and timings',code==200 and base64.b64decode(result['audio_b64'])[:4]==b'RIFF' and bool(result['words']))
 print('RESTART_JOB='+id,flush=True)

if __name__=='__main__':main()
