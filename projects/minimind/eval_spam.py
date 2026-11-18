"""Return exactly 스팸아님 or 스팸 for a nonempty message."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from model.model_spam import MiniMindForSpamClassification,LABELS
from dataset.spam_dataset import encode_message

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--text',required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
    path=Path(a.checkpoint);settings=json.loads((path/'spam_config.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=MiniMindForSpamClassification.from_checkpoint(path).to(a.device).eval()
    ids=torch.tensor([encode_message(tokenizer,a.text,settings['max_length'])],device=a.device)
    with torch.inference_mode():prediction=int(model(ids).logits.argmax(-1).item())
    print(LABELS[prediction])
if __name__=='__main__':main()
