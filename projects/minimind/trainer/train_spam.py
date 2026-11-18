"""Train last MiniMind block, final norm and binary head on grouped UCI SMS splits."""
import argparse,collections,csv,datetime,hashlib,json,math,os,random,re,shutil,sys,time,unicodedata,urllib.request,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.model_spam import MiniMindForSpamClassification,LABELS
from dataset.spam_dataset import SpamDataset,SpamCollator

SOURCE='https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'

def write(path,payload):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+'\n');temp.replace(path)

def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''): h.update(chunk)
    return h.hexdigest()

def frozen_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())

def prepare_data(output,raw_path,seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import StratifiedGroupKFold
    data=output/'data';data.mkdir()
    archive=data/'sms_spam_collection.zip'
    if raw_path:
        shutil.copyfile(raw_path,archive)
    else:
        archive.write_bytes(urllib.request.urlopen(SOURCE,timeout=60).read())
    with zipfile.ZipFile(archive) as z:
        raw=z.read('SMSSpamCollection');(data/'SMSSpamCollection').write_bytes(raw)
        (data/'UCI_readme.txt').write_bytes(z.read('readme'))
    rows=[];seen={};duplicates=0
    for index,line in enumerate(raw.decode('utf-8-sig').splitlines()):
        label,text=line.split('\t',1)
        if label not in ('ham','spam'):raise ValueError('Unknown SMS label')
        y=int(label=='spam');key=normalize(text)
        if key in seen:
            if seen[key]!=y:raise ValueError('Conflicting labels for identical text')
            duplicates+=1;continue
        seen[key]=y;rows.append({'source_index':index,'text':text,'label':y,'normalized':key})
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        i,j=find(i),find(j)
        if i!=j:parent[max(i,j)]=min(i,j)
    canonical=[];templates={}
    for i,row in enumerate(rows):
        text=re.sub(r'(?:https?://|www\.)\S+',' URL ',row['normalized'])
        text=re.sub(r'\d+',' NUM ',text)
        canonical.append(text)
        if text in templates:union(i,templates[text])
        else:templates[text]=i
    vectors=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),dtype=np.float32).fit_transform(canonical)
    nn=NearestNeighbors(metric='cosine',radius=0.1,n_jobs=1).fit(vectors)
    neighbors=nn.radius_neighbors(vectors,return_distance=False)
    edges=0
    for i,ids in enumerate(neighbors):
        for j in ids:
            if j>i:union(i,int(j));edges+=1
    groups=np.array([find(i) for i in range(len(rows))]);labels=np.array([r['label'] for r in rows])
    folds=list(StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=seed).split(np.zeros(len(rows)),labels,groups))
    test_ids=set(map(int,folds[0][1]));val_ids=set(map(int,folds[1][1]))
    splits={'train':[],'validation':[],'test':[]}
    for i,row in enumerate(rows):
        split='test' if i in test_ids else 'validation' if i in val_ids else 'train'
        row.update(group_id=int(groups[i]),split=split);splits[split].append(row)
    group_sets={s:{r['group_id'] for r in rs} for s,rs in splits.items()}
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        assert not(group_sets[left]&group_sets[right])
    for split,records in splits.items():
        with (data/f'{split}.jsonl').open('w') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    info={'source':SOURCE,'source_page':'https://archive.ics.uci.edu/dataset/228/sms+spam+collection','license':'CC BY 4.0; Almeida & Hidalgo (2011), doi:10.24432/C5CC84','archive_sha256':digest_file(archive),'raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_rows':len(raw.decode('utf-8-sig').splitlines()),'unique_rows':len(rows),'removed_exact_duplicates':duplicates,'groups':len(set(groups)),'near_duplicate_edges':edges,'grouping':'Deduplicate NFKC/casefold/whitespace-normalized text; union URL/digit-normalized exact templates and char_wb TF-IDF (3,5) cosine similarity >= 0.90; connected components. Heuristic, not a semantic leakage guarantee.','split':'StratifiedGroupKFold 10 folds, seed 42: fold 0 test, fold 1 validation, folds 2-9 train','cross_split_group_overlap':0,'counts':{s:{'total':len(rs),'ham':sum(r['label']==0 for r in rs),'spam':sum(r['label']==1 for r in rs)} for s,rs in splits.items()}}
    write(data/'metadata.json',info)
    return splits,info

def evaluate(model,loader,device,save_path=None,rows=None):
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_auc_score
    model.eval();ys=[];preds=[];scores=[];loss_sum=0
    with torch.inference_mode():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                result=model(**batch)
            loss_sum+=result.loss.item()*len(batch['labels'])
            ys.extend(batch['labels'].cpu().tolist());preds.extend(result.logits.argmax(-1).cpu().tolist());scores.extend(result.logits.float().softmax(-1)[:,1].cpu().tolist())
    precision,recall,f1,_=precision_recall_fscore_support(ys,preds,average='binary',zero_division=0)
    tn,fp,fn,tp=map(int,confusion_matrix(ys,preds,labels=[0,1]).ravel())
    metrics={'n':len(ys),'loss':loss_sum/len(ys),'accuracy':accuracy_score(ys,preds),'spam_precision':precision,'spam_recall':recall,'spam_f1':f1,'ham_false_positive_rate':fp/(tn+fp) if tn+fp else None,'roc_auc':roc_auc_score(ys,scores),'confusion_matrix':{'tn':tn,'fp':fp,'fn':fn,'tp':tp}}
    if save_path:
        with Path(save_path).open('w') as f:
            for row,y,p,s in zip(rows,ys,preds,scores):
                f.write(json.dumps({'source_index':row['source_index'],'group_id':row['group_id'],'text':row['text'],'label':y,'prediction':p,'prediction_label':LABELS[p],'spam_softmax_score':s},ensure_ascii=False)+'\n')
    return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);p.add_argument('--raw-archive');p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--max-length',type=int,default=340);p.add_argument('--seed',type=int,default=42);p.add_argument('--patience',type=int,default=4)
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this training run')
    device=torch.device('cuda');torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    print('PREPARING_DATA',flush=True)
    splits,info=prepare_data(out,args.raw_archive,args.seed);print('DATA',json.dumps(info['counts']),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(args.base,local_files_only=True);tokenizer.padding_side='right'
    collator=SpamCollator(tokenizer.pad_token_id)
    datasets={s:SpamDataset(rows,tokenizer,args.max_length) for s,rows in splits.items()}
    lengths={}
    for s,rows in splits.items():
        lens=[len(tokenizer.encode(r['text'].strip(),add_special_tokens=False))+2 for r in rows]
        lengths[s]={'min':min(lens),'median':float(np.median(lens)),'p95':float(np.percentile(lens,95)),'max':max(lens),'truncated':sum(n>args.max_length for n in lens)}
    write(out/'token_lengths.json',lengths)
    loaders={s:DataLoader(d,batch_size=args.batch_size,shuffle=(s=='train'),collate_fn=collator,num_workers=0) for s,d in datasets.items()}
    model=MiniMindForSpamClassification.from_language_model(args.base).to(device)
    names=model.configure_trainable_parameters();assert all(n.startswith((f'model.layers.{len(model.model.layers)-1}.','model.norm.','lm_head.')) for n in names)
    initial_frozen=frozen_hash(model)
    trainable_before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    import transformers,sklearn
    meta={'args':vars(args),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'sklearn':sklearn.__version__,'base':str(Path(args.base).resolve()),'base_weights_sha256':digest_file(Path(args.base)/'model.safetensors'),'tokenizer_sha256':digest_file(Path(args.base)/'tokenizer.json'),'model_config':model.config.to_dict(),'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_parameter_names':names,'initial_frozen_sha256':initial_frozen,'dataset':info,'token_lengths':lengths,'optimizer':{'type':'AdamW','last_block_and_norm_lr':5e-5,'head_lr':1e-3,'weight_decay':0.01,'schedule':'10% linear warmup, cosine decay to 10% of initial LR'},'selection':'Validation spam F1; validation loss breaks ties. Test evaluated once after checkpoint selection.','source_sha256':{str(path.relative_to(Path(__file__).resolve().parents[1])):digest_file(path) for path in [Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'model/model_spam.py',Path(__file__).resolve().parents[1]/'model/model_minimind.py',Path(__file__).resolve().parents[1]/'dataset/spam_dataset.py']}}
    write(out/'metadata.json',meta)
    before=evaluate(model,loaders['validation'],device);write(out/'before_validation.json',before)
    groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('lm_head.')],'lr':5e-5}, {'params':list(model.lm_head.parameters()),'lr':1e-3}]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    total_steps=len(loaders['train'])*args.epochs;warmup=max(1,int(total_steps*.1))
    def factor(step):
        if step<warmup:return (step+1)/warmup
        progress=(step-warmup)/max(1,total_steps-warmup)
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    best=(-1,float('-inf'));best_epoch=0;step=0;history=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0;count=0
        for batch in loaders['train']:
            batch={k:v.to(device) for k,v in batch.items()};optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):result=model(**batch)
            if not torch.isfinite(result.loss):raise RuntimeError('Nonfinite loss')
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1
            loss_sum+=result.loss.item()*len(batch['labels']);count+=len(batch['labels'])
            if step%25==0:
                write(out/'progress.json',{'stage':'training','epoch':epoch,'step':step,'total_planned_steps':total_steps,'train_loss':loss_sum/count})
        validation=evaluate(model,loaders['validation'],device)
        record={'epoch':epoch,'step':step,'train_loss':loss_sum/count,'validation':validation,'elapsed_seconds':time.monotonic()-started};history.append(record)
        print('EPOCH',json.dumps(record),flush=True);write(out/'history.json',history)
        score=(validation['spam_f1'],-validation['loss'])
        if score>best:
            best=score;best_epoch=epoch;model.save_checkpoint(out/'best',tokenizer,args.max_length)
            evaluate(model,loaders['validation'],device,out/'best_validation_predictions.jsonl',splits['validation'])
        if epoch-best_epoch>=args.patience:break
    final_frozen=frozen_hash(model);assert initial_frozen==final_frozen,'A frozen parameter changed'
    changed=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(p.detach().cpu(),trainable_before[n])]
    for prefix in (f'model.layers.{len(model.model.layers)-1}.','model.norm.','lm_head.'):
        assert any(n.startswith(prefix) for n in changed),f'No update in {prefix}'
    write(out/'freeze_verification.json',{'frozen_unchanged':True,'initial_sha256':initial_frozen,'final_sha256':final_frozen,'updated_trainable_names':changed})
    del model,optimizer,trainable_before;torch.cuda.empty_cache()
    selected=MiniMindForSpamClassification.from_checkpoint(out/'best').to(device).eval()
    # CPU reload must exactly match every saved parameter, not only output shapes.
    saved=torch.load(out/'best/model.pth',map_location='cpu',weights_only=True)
    assert all(torch.equal(t.detach().cpu(),saved[n]) for n,t in selected.state_dict().items())
    del saved
    test=evaluate(selected,loaders['test'],device,out/'test_predictions.jsonl',splits['test'])
    train=evaluate(selected,DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=False,collate_fn=collator),device)
    selected_validation=evaluate(selected,loaders['validation'],device)
    assert abs(selected_validation['spam_f1']-best[0])<1e-10
    probe=collator([datasets['test'][0],datasets['test'][1]])
    with torch.inference_mode():
        pb={k:v.to(device) for k,v in probe.items() if k!='labels'}
        padded=selected(**pb).logits
        l=int(probe['attention_mask'][0].sum())
        single=selected(pb['input_ids'][:1,:l],pb['attention_mask'][:1,:l]).logits
        torch.testing.assert_close(padded[:1],single,rtol=1e-4,atol=1e-4)
    summary={'success':True,'best_epoch':best_epoch,'epochs_completed':len(history),'updates':step,'train':train,'validation':selected_validation,'test':test,'before_validation':before,'majority_test_accuracy':max(info['counts']['test']['ham'],info['counts']['test']['spam'])/info['counts']['test']['total'],'total_parameters':meta['total_parameters'],'trainable_parameters':meta['trainable_parameters'],'frozen_unchanged':True,'checkpoint_exact_reload_verified':True,'padding_invariance_verified':True,'checkpoint':str(out/'best'),'checkpoint_sha256':digest_file(out/'best/model.pth'),'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'elapsed_seconds':time.monotonic()-started,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(out/'summary.json',summary);write(out/'progress.json',{'stage':'complete','best_epoch':best_epoch,'test':test});print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
