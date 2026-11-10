"""Pinned VLM backbone and native Omni audio donors, downloaded without credentials."""
import json,os,time
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(os.environ.get('OMNI_RUN_ROOT','/workspace/vlm-audio-20261109'))
REVISIONS={
 'caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0':'4e6f87f184b27f0692e1f8b3afed959de254894b',
 'jingyaogong/minimind-3o-pytorch':'9e22fb0e51852359ff51bd48d91ec3a345dbd75b',
 'jingyaogong/minimind-o_dataset':'d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de',
 'jingyaogong/SenseVoiceSmall':'964f07d8f0e48309a5f37770ac9b5ed2cdca6b91',
 'google/siglip2-base-patch32-256':'94dffa8cb1179de3e03f091dbc3917e5d5a9ae84',
 'kyutai/mimi':'89091b3e466eb6a9d11e537bf26b144f194978f7'}
def main():
 ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'revisions.json').write_text(json.dumps(REVISIONS,indent=2))
 for repo,folder,patterns,kind in [
  ('caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0','vlm',['*.json','*.txt','*.md','model.safetensors','vision_encoder/*','LICENSE-CODE'],'model'),
  ('jingyaogong/minimind-3o-pytorch','base',['sft_omni_768.pth','README.md'],'model'),
  ('jingyaogong/minimind-o_dataset','data/audio',['sft_t2a_mini.parquet','sft_a2a_mini.parquet','README.md'],'dataset'),
  ('jingyaogong/SenseVoiceSmall','models/audio',None,'model'),
  ('kyutai/mimi','models/mimi',['*.json','*.safetensors'],'model')]:
  print('DOWNLOAD',repo,flush=True)
  snapshot_download(repo,repo_type=kind,revision=REVISIONS[repo],local_dir=ROOT/folder,allow_patterns=patterns,max_workers=4)
 (ROOT/'PREPARED').write_text(str(time.time()))
if __name__=='__main__':main()
