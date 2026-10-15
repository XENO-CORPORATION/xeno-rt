"""Transcribe a retained native WAV with an owned offline Rust Whisper server."""
import argparse,json,os,pathlib,socket,subprocess,time,urllib.request,urllib.error
p=argparse.ArgumentParser();p.add_argument('--wav',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
env=os.environ.copy();env.update(XRT_CACHE_DIR='X:/tmp/audio-base-registry-proof',ORT_DYLIB_PATH='X:/tmp/audio-native-package-proof/onnxruntime.dll',HTTP_PROXY='http://127.0.0.1:9',HTTPS_PROXY='http://127.0.0.1:9',ALL_PROXY='http://127.0.0.1:9')
for name in ['XRT_AUDIO_MODEL_DIR','XRT_AUDIO_ASR_DIR','XENO_RT_WHISPER_DIR']:env.pop(name,None)
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with socket.socket()as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
log=a.out.with_suffix('.server.log').open('wb')
server=subprocess.Popen(['X:/tmp/audio-rc1-public-extracted-20260930/xeno-rt-0.4.0-rc.1-windows-x86_64/xrt-server.exe','--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
try:
 for _ in range(100):
  if server.poll() is not None:raise RuntimeError('owned server exited')
  try:
   with op.open(f'http://127.0.0.1:{port}/v1/audio/status',timeout=3)as response:break
  except OSError:time.sleep(.1)
 boundary='drama3-native-proof';body=b'--'+boundary.encode()+b'\r\nContent-Disposition: form-data; name="file"; filename="native.wav"\r\nContent-Type: audio/wav\r\n\r\n'+a.wav.read_bytes()+b'\r\n--'+boundary.encode()+b'--\r\n'
 request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/audio/transcriptions',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary})
 with op.open(request,timeout=180)as response:report=json.load(response)
 a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
finally:
 server.terminate()
 try:server.wait(timeout=20)
 except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
 log.close()
