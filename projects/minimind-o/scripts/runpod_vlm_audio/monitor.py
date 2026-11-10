"""Host-only capacity watcher: acquire -> train -> verify/upload -> clean up.

Never copy this process's environment or credentials to a Pod. Start from the
repository root using the host Python with huggingface_hub installed.
"""
import argparse,concurrent.futures,fcntl,hashlib,json,os,shlex,subprocess,sys,tarfile,time,traceback,urllib.error,urllib.parse,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
RUN_ID=os.environ.get('OMNI_RUN_ID','20261109-vlm-audio-auto')
OUT=ROOT/'outputs/minimind-o'/RUN_ID
REMOTE=os.environ.get('OMNI_REMOTE_ROOT','/workspace/vlm-audio-20261109')
REPO_ID=os.environ.get('OMNI_HF_REPO','caspar/MiniMind-O-VLM-Audio-Mini-5090-20261109')
OUT.mkdir(parents=True,exist_ok=True)

def write(name,value):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def status(phase,**kw):
 value=dict(phase=phase,time=time.time(),**kw);write('status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
def api_key():
 value=os.environ.get('RUNPOD_API_KEY')
 for line in (ROOT/'.env').read_text().splitlines():
  if '=' in line and not line.lstrip().startswith('#'):
   k,v=line.strip().removeprefix('export ').split('=',1)
   if k.strip() in ['RUNPOD_API_KEY','RUNPOD-API-KEY','RUNPOD_APIKEY']:value=v.strip().strip('\"\'')
 if not value:raise RuntimeError('Runpod API key missing')
 return value
def api(path,body=None,method=None):
 req=urllib.request.Request('https://api.runpod.io/v2/'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+api_key(),'Content-Type':'application/json'},method=method)
 try:
  with urllib.request.urlopen(req,timeout=60) as r:
   data=r.read();return json.loads(data) if data else None
 except urllib.error.HTTPError as e:
  error=e.read().decode();raise ApiError(e.code,error) from None
class ApiError(Exception):
 def __init__(self,code,detail):self.code=code;self.detail=detail;super().__init__(f'HTTP {code}: {detail}')
def cli(*args):
 env=dict(os.environ,RUNPOD_API_KEY=api_key())
 p=subprocess.run(['runpodctl',*args],env=env,capture_output=True,text=True,timeout=90)
 if p.returncode:raise RuntimeError(p.stderr)
 return json.loads(p.stdout)
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def delete_volume(vid):
 cli('network-volume','delete',vid)
 try:api('network-volumes/'+vid)
 except ApiError as e:
  if e.code==404:return
  raise
 raise RuntimeError('Deleted volume still exists: '+vid)
def delete_pod(pid):
 cli('pod','delete',pid)
 try:api('pods/'+pid)
 except ApiError as e:
  if e.code==404:return
  raise
 raise RuntimeError('Deleted Pod still exists: '+pid)
def prepare_bundle():
 project=ROOT/'projects/minimind-o';bundle=OUT/'source.tar.gz'
 with tarfile.open(bundle,'w:gz') as tar:
  tar.add(project/'LICENSE',arcname='repo/LICENSE')
  for folder in ['model','trainer','dataset']:
   for p in (project/folder).glob('*.py'):tar.add(p,arcname=f'repo/{folder}/{p.name}')
  for p in (project/'scripts/runpod_vlm_audio').iterdir():
   if p.is_file() and p.name!='monitor.py':tar.add(p,arcname=f'repo/scripts/runpod_vlm_audio/{p.name}')
  for p in (ROOT/'outputs/minimind-o/omni-mini-20261025/artifacts/samples').glob('image-*-input.jpg'):
   tar.add(p,arcname='image_inputs/'+p.name)
 with tarfile.open(bundle) as tar:write('source.json',dict(sha256=digest(bundle),files=tar.getnames()))
 return bundle

def acquire(wait_hours,interval):
 deadline=time.time()+wait_hours*3600;attempt=0
 while time.time()<deadline:
  attempt+=1;status('checking_capacity',attempt=attempt,deadline=deadline)
  gpu=api('catalog/gpus/'+urllib.parse.quote('NVIDIA GeForce RTX 5090')+'?include=AVAILABILITY&product=POD&cloud=SECURE&minCudaVersion=12.8')
  write('latest-capacity.json',gpu)
  price=gpu.get('price',{}).get('secure')
  if price is None or price>1.19:raise RuntimeError(f'RTX 5090 price exceeds configured $1.19/hour ceiling: {price}')
  for dc in gpu.get('dataCenters',[]):
   if str(dc.get('availability','')).upper() not in ['LOW','MEDIUM','HIGH']:continue
   dcinfo=api('catalog/datacenters/'+dc['id'])
   if 'STANDARD' not in dcinfo.get('networkVolumeTypes',[]):continue
   # Never touch account resources that were not created by this invocation.
   vol=cli('network-volume','create','--name','omni-vlm-audio-auto','--size','50','--data-center-id',dc['id'])
   vid=vol['id'];write('resources.json',dict(volume=vid,pod=None,dc=dc['id']))
   checked=cli('network-volume','get',vid);assert checked['size']==50 and checked['dataCenterId']==dc['id']
   body=dict(name='omni-vlm-audio-auto',cloud='SECURE',image='runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404',disk=30,gpu=dict(id='NVIDIA GeForce RTX 5090',count=1,minCudaVersion='12.8'),dataCenterIds=[dc['id']],mounts=dict(network=[dict(volumeId=vid,path='/workspace')]),ports=['22/tcp'],startSsh=True)
   try:pod=api('pods',body)
   except ApiError as e:
    delete_volume(vid);write('resources.json',dict(volume=None,pod=None))
    if e.code==400 and ('no longer any instances' in e.detail.lower() or 'no instances available' in e.detail.lower()):
     status('placement_unavailable',attempt=attempt,dc=dc['id']);continue
    raise
   pid=pod['id'];write('resources.json',dict(volume=vid,pod=pid,dc=dc['id'],created=time.time()))
   # Save only non-secret metadata: Pod reads include injected environment values.
   checked=api('pods/'+pid);assert checked['id']==pid
   status('pod_allocated',pod=pid,volume=vid,dc=dc['id'],gpu_price_per_hour=price)
   return pid,vid
  status('waiting_for_capacity',attempt=attempt,next_check=time.time()+interval,deadline=deadline)
  time.sleep(min(interval,max(0,deadline-time.time())))
 status('capacity_wait_expired',wait_hours=wait_hours)
 return None,None

def connection(pid):
 end=time.time()+5*60
 while time.time()<end:
  try:
   # v3.1.0 can report "no container" even when the v2 runtime and SSH are ready.
   direct=api('pods/'+pid).get('ssh',{}).get('direct')
   d=(dict(ip=direct['host'],port=direct['port']) if direct else cli('ssh','info',pid))
   if d.get('ip') and d.get('port'):
    key=d.get('ssh_key',{}).get('path') or str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')
    ssh=['ssh','-i',key,'-p',str(d['port']),'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(OUT/'known_hosts'),'-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','root@'+d['ip']]
    probe=subprocess.run(ssh+['true'],capture_output=True,text=True,timeout=30)
    if probe.returncode==0:return ssh,d,key
    write('ssh-diagnostic.json',dict(ip=d['ip'],port=d['port'],key_path=key,returncode=probe.returncode,stderr=probe.stderr[-3000:]))
   elif d.get('error'):write('ssh-diagnostic.json',d)
  except Exception as e:write('ssh-diagnostic.json',dict(error=str(e)))
  status('waiting_for_ssh',pod=pid);time.sleep(30)
 raise TimeoutError('SSH not ready after 5 minutes; see ssh-diagnostic.json')
def scp_args(d,key):
 return ['scp','-i',key,'-P',str(d['port']),'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(OUT/'known_hosts'),'-o','ConnectTimeout=20']
def ssh_run(ssh,command,timeout=90,check=True):
 result=subprocess.run(ssh+[command],capture_output=True,text=True,timeout=timeout)
 if check and result.returncode:
  raise RuntimeError(f'Remote command failed ({result.returncode}): {command}\n{result.stderr[-6000:]}')
 return result
def verify_and_publish(bundle,success):
 assert digest(bundle)==success['bundle_sha256'],'Bundle hash mismatch'
 with tarfile.open(bundle) as tar:tar.extractall(OUT,filter='data')
 folder=OUT/'artifacts';hashes=json.loads((folder/'SHA256SUMS.json').read_text())
 for name,expected in hashes.items():assert digest(folder/name)==expected,name
 assert digest(folder/'sft_omni_768.pth')==success['checkpoint_sha256']
 assert json.loads((folder/'vlm_preservation.json').read_text())['exact_match']
 from huggingface_hub import HfApi
 hf=HfApi();assert hf.whoami()['name']=='caspar'
 # The name belongs to this run; never silently replace an existing different model.
 hf.create_repo(REPO_ID,private=True,exist_ok=False)
 commit=hf.upload_folder(repo_id=REPO_ID,folder_path=folder,commit_message='Upload frozen VLM backbone with aligned Omni audio modules, validation and samples')
 info=hf.model_info(REPO_ID,revision=commit.oid,files_metadata=True);remote={s.rfilename:s for s in info.siblings}
 for p in folder.rglob('*'):
  if not p.is_file():continue
  name=p.relative_to(folder).as_posix();f=remote[name];assert f.size==p.stat().st_size,name
  if f.lfs:assert f.lfs.sha256==digest(p),name
  else:
   raw=p.read_bytes();assert f.blob_id==hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),name
 result=dict(repo=REPO_ID,url='https://huggingface.co/'+REPO_ID,private=True,commit=commit.oid,checkpoint_sha256=success['checkpoint_sha256'],verified_files=len(remote)-1,samples=success['samples'])
 write('upload-verified.json',result)
 report=ROOT/'reports/minimind-o'/f'{RUN_ID}.md';report.parent.mkdir(parents=True,exist_ok=True)
 report.write_text((folder/'REPORT_ko.md').read_text()+'\n## 업로드 및 실행 기록\n\n'+f"모델: {result['url']} (비공개)\n\n커밋: `{commit.oid}`\n\n산출물: `outputs/minimind-o/{RUN_ID}/`\n\n체크포인트 SHA256: `{result['checkpoint_sha256']}`\n\n원본 리비전·환경·명령은 artifacts/revisions.json, run_config.json, requirements-resolved.txt와 source/를 참조하세요.\n")
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--interval',type=int,default=600);parser.add_argument('--wait-hours',type=float,default=24);parser.add_argument('--resume',action='store_true',help='Continue this run from its recorded Pod and volume');args=parser.parse_args()
 lock=(OUT/'monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (OUT/'resources.json').exists():
  old=json.loads((OUT/'resources.json').read_text())
  if (old.get('pod') or old.get('volume')) and not args.resume:raise RuntimeError('Previous resource state requires recovery before a new monitor')
 write('monitor.json',dict(pid=os.getpid(),started=time.time(),interval_seconds=args.interval,wait_hours=args.wait_hours,repo=REPO_ID,training_deadline_hours=12,python=sys.version))
 from huggingface_hub import HfApi
 assert HfApi().whoami()['name']=='caspar'
 bundle=prepare_bundle();pid=vid=None;ssh=d=key=None
 try:
  if args.resume:
   old=json.loads((OUT/'resources.json').read_text());pid=old['pod'];vid=old['volume']
   assert pid and api('pods/'+pid)['id']==pid
  else:pid,vid=acquire(args.wait_hours,args.interval)
  if pid is None:return
  ssh,d,key=connection(pid);status('deploying_sources',pod=pid)
  ssh_run(ssh,'mkdir -p '+shlex.quote(REMOTE))
  subprocess.run(scp_args(d,key)+[str(bundle),'root@'+d['ip']+':'+REMOTE+'/source.tar.gz'],check=True,timeout=300)
  ssh_run(ssh,'tar --no-same-owner --no-same-permissions --touch -xzf '+REMOTE+'/source.tar.gz -C '+REMOTE)
  command=f'export TRAINING_POD_ID={shlex.quote(pid)} OMNI_RUN_ROOT={shlex.quote(REMOTE)} TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4; setsid timeout 12h bash {REMOTE}/repo/scripts/runpod_vlm_audio/setup.sh >{REMOTE}/run.log 2>&1 </dev/null &'
  ssh_run(ssh,command);started=time.time();errors=0
  while time.time()-started<12*3600:
   try:
    check=ssh_run(ssh,'cat '+REMOTE+'/SUCCESS.json 2>/dev/null',check=False)
    if check.returncode==0:
     success=json.loads(check.stdout);status('downloading_results',pod=pid)
     local=OUT/'artifacts.tar.gz'
     subprocess.run(scp_args(d,key)+['root@'+d['ip']+':'+REMOTE+'/artifacts.tar.gz',str(local)],check=True,timeout=1800)
     ssh_run(ssh,'tail -n 500 '+REMOTE+'/run.log',check=False)
     status('verifying_and_uploading',pod=pid);result=verify_and_publish(local,success)
     delete_pod(pid);pid=None;delete_volume(vid);vid=None
     write('resources.json',dict(pod=None,volume=None));status('complete',**result)
     with (ROOT/'reports/minimind-o'/f'{RUN_ID}.md').open('a') as f:f.write('\nPod 및 이번 작업의 네트워크 볼륨 삭제를 API 읽기로 확인했습니다.\n')
     return
    snapshot=ssh_run(ssh,'cat '+REMOTE+'/status.json 2>/dev/null; tail -n 8 '+REMOTE+'/run.log',check=False)
    (OUT/'remote-latest.txt').write_text(snapshot.stdout);status('training_or_evaluating',pod=pid,elapsed_seconds=time.time()-started)
    failure=ssh_run(ssh,'cat '+REMOTE+'/failure.json 2>/dev/null',check=False)
    if failure.returncode==0:(OUT/'remote-failure.json').write_text(failure.stdout);raise RuntimeError('Remote training reported failure')
    alive=ssh_run(ssh,"pgrep -f "+shlex.quote('^timeout 12h bash '+REMOTE+'/repo/scripts/runpod_vlm_audio/setup.sh$'),check=False)
    if alive.returncode!=0:raise RuntimeError('Remote pipeline exited before SUCCESS.json')
    errors=0
   except (subprocess.SubprocessError,OSError):
    errors+=1
    if errors>=10:raise
   time.sleep(45)
  raise TimeoutError('12-hour training deadline reached')
 except BaseException:
  error=traceback.format_exc();(OUT/'error.txt').write_text(error)
  if pid is None and (OUT/'resources.json').exists():
   recorded=json.loads((OUT/'resources.json').read_text());pid=recorded.get('pod');vid=recorded.get('volume')
  if pid:
   # Preserve diagnostics and durable network checkpoints, release billed GPU.
   if ssh:
    try:
     r=ssh_run(ssh,'tail -n 1000 '+REMOTE+'/run.log',check=False);(OUT/'remote-error.log').write_text(r.stdout)
    except Exception:pass
   try:cli('pod','stop',pid)
   except Exception as e:error+='\nSTOP FAILED: '+str(e)
   status('failed_pod_stopped_volume_preserved',pod=pid,volume=vid,error=error)
  else:
   if vid:
    try:delete_volume(vid);vid=None;write('resources.json',dict(pod=None,volume=None))
    except Exception as e:error+='\nVOLUME CLEANUP FAILED: '+str(e)
   status('failed',volume=vid,error=error)
  raise
if __name__=='__main__':main()
