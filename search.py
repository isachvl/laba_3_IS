import argparse, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
from retrieval import BM25, ranked, fuse, test
import sys
root=Path(__file__).resolve().parent
lr2=root.parent/'lab2'
sys.path.insert(0,str(lr2))
from common import read_jsonl, write_jsonl, save_json, sha
from evaluation import retrieval_metrics

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('split', choices=['dev','holdout'])
    args=parser.parse_args()
    out=root/'results'
    out.mkdir(exist_ok=True)
    if args.split=='holdout' and not (out/'selection.json').exists():
        raise SystemExit('Freeze dev selection before holdout')
    target=out/f'{args.split}_rankings.jsonl'
    if target.exists(): raise SystemExit(f'Existing result: {target}')
    test()
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder
    torch.set_num_threads(8)
    cfg=json.loads((root/'retrieval.json').read_text(encoding='utf-8-sig'))
    passport=json.loads((lr2/'index/passport.json').read_text(encoding='utf-8'))
    assert sha(lr2/'index/vectors.npy')==passport['matrix_sha256']
    manifest=json.loads((lr2/'data/manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items(): assert sha(lr2/'data'/name)==digest
    ids=manifest[args.split+'_ids']
    claims={c['id']:c for c in read_jsonl(lr2/'data/claims_dev.jsonl') if c['id'] in ids}
    docs={str(d['doc_id']):d['title']+'\n'+' '.join(d['abstract']) for d in read_jsonl(lr2/'data/corpus.jsonl')}
    doc_ids=json.loads((lr2/'index/doc_ids.json').read_text())
    vectors=np.load(lr2/'index/vectors.npy')
    encoder=SentenceTransformer(passport['model'],revision=passport['revision'],device='cpu',cache_folder=str(lr2/'models/encoder'))
    from huggingface_hub import model_info
    ce_revision=(json.loads((out/'manifest.json').read_text())['ce_revision'] if args.split=='holdout' else model_info(cfg['ce']).sha)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(cfg['ce'],revision=ce_revision)
    model=AutoModelForSequenceClassification.from_pretrained(cfg['ce'],revision=ce_revision).eval()
    bm25=BM25(docs)
    previous={r['id']:r['top5'] for r in read_jsonl(lr2/'index/dev_rankings.jsonl')}
    results=[]
    for n,i in enumerate(ids,1):
        q=claims[i]['claim']; timing={}
        start=time.perf_counter()
        vec=encoder.encode(passport['query_prefix']+q,normalize_embeddings=True).astype('float32')
        dense=ranked(dict(zip(doc_ids,vectors@vec)))
        timing['dense']=time.perf_counter()-start
        if args.split=='dev': assert [x['doc_id'] for x in dense[:5]]==[x['doc_id'] for x in previous[i]],i
        start=time.perf_counter(); lexical=bm25.search(q); timing['bm25']=time.perf_counter()-start
        start=time.perf_counter(); rrf=fuse(dense,lexical,cfg['rrf_constant']); timing['fusion']=time.perf_counter()-start
        start=time.perf_counter()
        candidates=[x['doc_id'] for x in rrf]
        scores=[]
        with torch.inference_mode():
            for j in range(0,len(candidates),8):
                batch=candidates[j:j+8]
                encoded=tokenizer([q]*len(batch),[docs[d] for d in batch],truncation='only_second',max_length=512,padding=True,return_tensors='pt')
                scores.extend(model(**encoded).logits.squeeze(-1).tolist())
        ce=ranked(dict(zip(candidates,scores)))
        timing['rerank']=time.perf_counter()-start
        assert {r['doc_id'] for r in ce}==set(candidates)
        results.append(dict(id=i,claim=q,dense=dense,bm25=lexical,rrf=rrf,ce=ce,timing_s=timing))
        if n%10==0: print(args.split,n,len(ids),flush=True)
    write_jsonl(target,results)
    stats={}
    for method in ('dense','bm25','rrf','ce'):
        rows=[(r,claims[r['id']]['evidence']) for r in results if claims[r['id']]['evidence']]
        stats[method]={}
        for k in (3,20):
            vals=[retrieval_metrics(g,[x['doc_id'] for x in r[method]],k) for r,g in rows]
            stats[method][str(k)]={m:float(np.mean([v[m] for v in vals])) for m in ('hit','recall','rr')}
    assert stats['rrf']['20']['recall']==stats['ce']['20']['recall']
    save_json(out/f'{args.split}_retrieval.json',stats)
    if args.split=='dev':
        save_json(out/'manifest.json',dict(lr2=manifest,dense=passport,ce_revision=ce_revision,configuration=cfg))
        winner=max(['bm25','rrf','ce'],key=lambda m:(stats[m]['3']['rr'],stats[m]['3']['recall']))
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        save_json(out/'selection.json',dict(winner=winner,dev=stats,commit=commit,configuration_sha256=sha(root/'retrieval.json')))
        print('Frozen selection:',winner,flush=True)

if __name__=='__main__': main()
