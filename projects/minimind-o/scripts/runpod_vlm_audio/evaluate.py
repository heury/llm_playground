"""Generate heldout examples and package a reloadable native checkpoint."""
import os,sys,json,time,shutil,io,zipfile,hashlib,tarfile
from pathlib import Path
ROOT=Path(os.environ.get('OMNI_RUN_ROOT','/workspace/vlm-audio-20261109'));REPO=ROOT/'repo'
sys.path.insert(0,str(REPO))
import torch,numpy as np,soundfile as sf,pyarrow.parquet as pq
from PIL import Image,ImageOps
from transformers import AutoTokenizer,MimiModel
from model.model_omni import MiniMindOmni,OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.trainer_utils import setup_seed
from train import write,sha,phase,ART

def main():
 torch.set_num_threads(4);setup_seed(42)
 assert (ROOT/'TRAINED').exists()
 model=MiniMindOmni(OmniConfig.from_pretrained(ART),audio_encoder_path=str(ROOT/'models/audio'),vision_model_path=str(ROOT/'vlm/vision_encoder'))
 model.load_state_dict(torch.load(ART/'sft_omni_768.pth',map_location='cpu',weights_only=True),strict=True)
 model.eval().cuda();model.audio_encoder.cuda().eval();model.vision_encoder.cuda().eval()
 tok=AutoTokenizer.from_pretrained(ART/'tokenizer');mimi=MimiModel.from_pretrained(ROOT/'models/mimi').eval().cuda()
 samples=ART/'samples';samples.mkdir(exist_ok=True);tests=[]
 splits=json.loads((ART/'audio_splits.json').read_text())
 for kind in ['t2a','a2a']:
  table=pq.read_table(ROOT/f'data/audio/sft_{kind}_mini.parquet')
  for j,idx in enumerate(splits[kind]['test'][:3]):
   row=table.slice(idx,1).to_pylist()[0];convs=json.loads(row['conversations'])
   # First complete turn avoids accidentally passing a reference assistant answer.
   user=next(t['content'] for t in convs if t['role']=='user')
   expected=next(t['content'] for t in convs if t['role']=='assistant')
   name=f'{kind}-{j+1}';kwargs={};prompt=user;input_file=None
   if kind=='a2a':
    raw=row['question_audios'][0];wave,sr=sf.read(io.BytesIO(raw));input_file=f'{name}-input.wav';sf.write(samples/input_file,wave,sr)
    mel,length=OmniDataset.process_audio(str(samples/input_file),model.audio_processor)
    kwargs=dict(audio_inputs=mel[None].cuda(),audio_lens=torch.tensor([length],device='cuda'))
    prompt=model.config.audio_special_token*length
   tests.append(dict(id=name,kind=kind,prompt=prompt,kwargs=kwargs,source_row=idx,input_text=user,reference_text=expected,input_file=input_file))
  del table
 for j,path in enumerate(sorted((ROOT/'image_inputs').glob('*.jpg'))):
  im=ImageOps.exif_transpose(Image.open(path)).convert('RGB');name=f'image-{j+1}';input_file=f'{name}-input.jpg';im.save(samples/input_file)
  pixels={k:v.cuda() for k,v in model.vision_processor(images=im,return_tensors='pt').items()}
  user='Describe the image briefly.'
  prompt=user+'\n'+model.config.image_special_token*model.config.image_token_len
  tests.append(dict(id=name,kind='image',prompt=prompt,kwargs=dict(pixel_values=pixels),input_text=user,reference_text='Manual image inspection required',input_file=input_file))
 results=[]
 for test in tests:
  phase('generating',sample=test['id']);setup_seed(42)
  prompt=tok.apply_chat_template([{'role':'user','content':test['prompt']}],tokenize=False,add_generation_prompt=True,open_thinking=False)
  ids=torch.tensor([tok.encode(prompt,add_special_tokens=False)],device='cuda')
  frames=[];answer='';steps=0;start=time.time()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   delta=None
   if test['kwargs']:
    actual=model(ids,**test['kwargs']).logits[0,-1].float()
    zero={k:({a:torch.zeros_like(b) for a,b in v.items()} if isinstance(v,dict) else torch.zeros_like(v)) for k,v in test['kwargs'].items()}
    null=model(ids,**zero).logits[0,-1].float();delta=float((actual-null).abs().max());assert delta>1e-5
   for y,frame in model.generate(ids,eos_token_id=tok.eos_token_id,max_new_tokens=512,temperature=.3,top_p=.9,stream=True,return_audio_codes=True,open_thinking=False,enter_token_id=tok.encode("\n",add_special_tokens=False)[0],pad_token_id=tok.pad_token_id,**test['kwargs']):
    steps+=1
    if y is not None:answer=tok.decode(y[0],skip_special_tokens=True)
    if frame is not None:frames.append(frame)
  result={k:v for k,v in test.items() if k not in ['prompt','kwargs']}
  result.update(response_text=answer,frames=len(frames),generation_steps=steps,max_new_tokens=512,possibly_truncated=steps>=512,conditioning_logit_difference=delta,elapsed_seconds=time.time()-start)
  if frames:
   codes=torch.tensor(frames,device='cuda').T[None];assert codes.min()>=0 and codes.max()<2048
   with torch.no_grad():wav=mimi.decode(codes).audio_values.squeeze().float().cpu().numpy()
   assert np.isfinite(wav).all();path=samples/f"{test['id']}-output.wav";sf.write(path,wav,24000)
   result.update(output_audio=path.name,seconds=len(wav)/24000,rms=float(np.sqrt(np.mean(wav**2))))
  else:result['output_audio']=None
  results.append(result);write(samples/'results.json',results);print(json.dumps(result,ensure_ascii=False),flush=True)
 # Poor generations are evaluation results, and must accompany the trained model.
 # Do not discard a completed training run merely because its speech is poor.
 checks={f'{kind}_has_non_silent_audio':any(r['kind']==kind and r.get('output_audio') and r.get('rms',0)>1e-6 for r in results) for kind in ['t2a','a2a']}
 checks['all_responses_nonempty']=all(r['response_text'].strip() for r in results)
 checks['all_generations_terminated']=all(not r['possibly_truncated'] for r in results)
 write(ART/'sample_checks.json',checks)
 # Automatic ASR is an auxiliary transcription check, not human listening.
 phase('sample_asr')
 from funasr import AutoModel
 from funasr.utils.postprocess_utils import rich_transcription_postprocess
 asr=AutoModel(model=str(ROOT/'models/audio'),trust_remote_code=True,disable_update=True,device='cpu')
 for result in results:
  if result.get('output_audio'):
   try:
    transcription=asr.generate(input=str(samples/result['output_audio']),cache={},language='en',use_itn=True,batch_size_s=60)
    result['output_audio_asr']=rich_transcription_postprocess(transcription[0]['text'])
   except Exception as e:result['output_audio_asr_error']=str(e)
 write(samples/'results.json',results)
 for folder in ['model','dataset','trainer']:
  dest=ART/'source'/folder;dest.mkdir(parents=True,exist_ok=True)
  for p in (REPO/folder).glob('*.py'):shutil.copyfile(p,dest/p.name)
 shutil.copytree(REPO/'scripts/runpod_vlm_audio',ART/'source/scripts/runpod_vlm_audio',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 import subprocess
 (ART/'requirements-resolved.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 (ART/'README.md').write_text('''---
language:
- en
license: other
license_name: upstream-model-and-dataset-terms
license_link: https://huggingface.co/caspar/MiniMind-O-VLM-Audio-Mini-5090-20261110/blob/main/provenance/LICENSE_NOTES.md
base_model:
- caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0
- jingyaogong/minimind-3o-pytorch
datasets:
- jingyaogong/minimind-o_dataset
pipeline_tag: any-to-any
tags:
- minimind
- audio
- experimental
---
# MiniMind-O VLM + Audio Mini — RTX 5090

Omni model using the exact frozen Thinker, language head, vision projector and tokenizer of
`caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0`. Talker and audio projector are initialized
from `jingyaogong/minimind-3o-pytorch/sft_omni_768.pth` and aligned to the VLM hidden states.
The original Omni text embeddings are replaced, never paired with the VLM tokenizer.

Training uses all rows of `sft_t2a_mini` and `sft_a2a_mini`, apart from 128 validation and 128 disjoint test rows each.
One epoch T2A Talker, one epoch A2A audio projector alignment, one epoch A2A Talker+audio projector.
VLM weights, SenseVoice, SigLIP and Mimi stay frozen. Exact VLM preservation is checked.
No visual dataset is trained in this run. Speech quality and factual accuracy are experimental.
Audio losses include 10x stop-token weights; lower teacher-forced loss is not a speech-quality score.

## Inference

This is a native PyTorch checkpoint, not an AutoModel export. Download a complete snapshot,
install the recorded dependencies in a CUDA-compatible environment, then run from the snapshot:

```bash
python source/scripts/runpod_vlm_audio/infer.py --model-dir . --prompt "Hello!" --output-wav reply.wav
python source/scripts/runpod_vlm_audio/infer.py --model-dir . --audio question.wav
python source/scripts/runpod_vlm_audio/infer.py --model-dir . --image photo.jpg --prompt "Describe the image."
```

Frozen encoders and Mimi are downloaded separately at pinned revisions in `revisions.json`.
See `training_summary.json`, `metrics/`, `samples/results.json`, `initialization.json`,
`vlm_preservation.json` and `REPORT_ko.md` for provenance and actual measurements.
The included source is Apache-2.0. Upstream weights and datasets retain their original terms.
''')
 card=ART/'README.md'
 card.write_text(card.read_text().replace('caspar/MiniMind-O-VLM-Audio-Mini-5090-20261110',os.environ.get('OMNI_HF_REPO','caspar/MiniMind-O-VLM-Audio-Mini-5090-20261114')))
 prov=ART/'provenance';prov.mkdir(exist_ok=True)
 shutil.copyfile(REPO/'LICENSE',prov/'MINIMIND_O_CODE_LICENSE')
 shutil.copyfile(ROOT/'vlm/LICENSE-CODE',prov/'MINIMIND_V_CODE_LICENSE')
 shutil.copyfile(ROOT/'data/audio/README.md',prov/'AUDIO_DATASET_CARD.md')
 shutil.copyfile(ROOT/'vlm/README.md',prov/'VLM_MODEL_CARD.md')
 (prov/'LICENSE_NOTES.md').write_text('Code: Apache-2.0; included code licenses apply. Weights and data retain upstream terms. See the included upstream cards and pinned revisions. No additional commercial permission is granted.\n')
 summary=json.loads((ART/'training_summary.json').read_text())
 lines=['# VLM 기반 Omni 음성 학습 결과','','VLM Thinker·LM head·시각 Projector를 고정하고 Omni 음성 모듈을 학습했습니다.','','| 단계 | 처리 행 | 업데이트 | 분 |','|---|---:|---:|---:|']
 for s in summary['stages']:lines.append(f"| {s['stage']} | {s['processed']} | {s['optimizer_updates']} | {s['elapsed_seconds']/60:.2f} |")
 lines+=['','## 검증 손실','','| 평가 | 텍스트 | 오디오(종료 토큰 10배) |','|---|---:|---:|']
 for p in sorted((ART/'metrics').glob('*.json')):
  m=json.loads(p.read_text())
  if 'tag' in m:lines.append(f"| {m['tag']} | {m['text_loss']:.4f} | {m['audio_weighted_loss']:.4f} |")
 lines+=['','## 생성 검사','',json.dumps(checks,ensure_ascii=False),'','false인 검사는 생성 품질 실패이며, 학습 완료 여부와 별개입니다.','','## 생성 샘플','']
 for r in results:
  lines += [f"### {r['id']}",'',f"입력: {r['input_text']}",'',f"참고: {r['reference_text']}",'',f"응답: {r['response_text']}",'',f"자동 전사: {r.get('output_audio_asr',r.get('output_audio_asr_error','없음'))}",'',f"프레임: {r['frames']}; 길이 제한 도달: {r['possibly_truncated']}",'']
  if r.get('input_file'):lines += [f"[입력](samples/{r['input_file']})",'']
  if r.get('output_audio'):lines += [f"[생성 음성](samples/{r['output_audio']})",'']
 lines+=['## 한계','','데이터셋별 검증 128행과 별도의 테스트 128행을 학습에서 제외했습니다. 생성 샘플도 테스트에서 추출했습니다. 초기 모델 사전학습과의 중복은 알 수 없습니다. 음성은 자동 전사와 비무음 여부를 확인하며, 사람의 청취 품질 평가는 수행하지 않았습니다. 이미지 샘플은 별도 시각 성능 벤치마크가 아닙니다.']
 (ART/'REPORT_ko.md').write_text('\n'.join(lines)+'\n')
 smoke=subprocess.run([sys.executable,str(ART/'source/scripts/runpod_vlm_audio/infer.py'),'--model-dir',str(ART),'--prompt','Hello!','--max-new-tokens','2','--device','cpu'],capture_output=True,text=True,timeout=120,check=True)
 (ART/'portable_inference_smoke.txt').write_text(smoke.stdout)
 shutil.copyfile(ROOT/'run.log',ART/'execution.log')
 write(ART/'execution_context.json',dict(run_root=str(ROOT),working_directory=os.getcwd(),command=f'bash {REPO}/scripts/runpod_vlm_audio/setup.sh',pod=os.environ.get('TRAINING_POD_ID')))
 # Python bytecode and caches are not release artifacts.
 for p in ART.rglob('*.pyc'):p.unlink()
 hashes={str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
 write(ART/'SHA256SUMS.json',hashes)
 with tarfile.open(ROOT/'artifacts.tar.gz.tmp','w:gz',compresslevel=1) as tar:tar.add(ART,arcname='artifacts')
 (ROOT/'artifacts.tar.gz.tmp').replace(ROOT/'artifacts.tar.gz')
 write(ROOT/'SUCCESS.json',dict(bundle_sha256=sha(ROOT/'artifacts.tar.gz'),checkpoint_sha256=sha(ART/'sft_omni_768.pth'),samples=len(results)))
 phase('complete',samples=len(results))
if __name__=='__main__':main()
