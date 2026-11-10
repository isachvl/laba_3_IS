import json, statistics, sys
from pathlib import Path

root=Path(__file__).resolve().parent
lr2=root.parent/'lab2'
out=root/'results'
sys.path.insert(0,str(lr2))
from common import read_jsonl, save_json
from evaluation import claim_label, rationale_present, citation_correct

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def good(r,g): return r['status']=='ok' and r['parsed']['label']==claim_label(g)
def rate(values): return sum(values)/len(values)

def evaluate():
    selection=read(out/'selection.json')
    winner=selection['winner']
    gold={r['id']:r for r in read_jsonl(lr2/'data/claims_dev.jsonl')}
    rankings={s:read_jsonl(out/f'{s}_rankings.jsonl') for s in ('dev','holdout')}
    baseline=read_jsonl(lr2/'runs/20261106_183751/experiment1/results.jsonl')
    dev={'dense':{r['id']:r for r in baseline if r['mode']=='rag' and r['k']==3}}
    for name in ('rrf','ce'): dev[name]={r['id']:r for r in read_jsonl(out/'dev'/f'{name}.jsonl')}
    hold={m:{r['id']:r for r in read_jsonl(out/'holdout'/f'{m}.jsonl')} for m in ('dense',winner)}
    summary={'selection':selection,'retrieval':{s:read(out/f'{s}_retrieval.json') for s in rankings},'generation':{},'timings':{},'complementarity':{}}
    for split,groups in [('dev',dev),('holdout',hold)]:
        ids={r['id'] for r in rankings[split]}
        assert len(ids)==(90 if split=='dev' else 210)
        summary['generation'][split]={}
        for method,rows in groups.items():
            assert rows.keys()==ids
            assert all(r['prompt_tokens']<=2048 for r in rows.values())
            positive=[i for i in rows if gold[i]['evidence']]
            recall={}
            for label in ('SUPPORT','CONTRADICT','NOINFO'):
                recall[label]=rate([good(rows[i],gold[i]) for i in rows if claim_label(gold[i])==label])
            summary['generation'][split][method]=dict(
                accuracy=rate([good(r,gold[i]) for i,r in rows.items()]),
                class_recall=recall,
                citation=rate([rows[i]['status']=='ok' and citation_correct(gold[i],rows[i]['parsed']) for i in positive]),
                evidence=rate([rationale_present(gold[i]['evidence'],rows[i]['transmitted']) for i in positive]),
                invalid=sum(r['status']!='ok' for r in rows.values()))
    summary['paired']={}
    for split,base,other in [('dev',dev['dense'],dev['ce']),('holdout',hold['dense'],hold[winner])]:
        summary['paired'][split]={kind:[i for i in base if (good(base[i],gold[i]),good(other[i],gold[i]))==transition]
                                 for kind,transition in [('fixed',(False,True)),('broken',(True,False))]}
    dev_rank=rankings['dev']
    def hit(r,m,k): return bool(set(gold[r['id']]['evidence']) & {d['doc_id'] for d in r[m][:k]})
    for k in (3,20):
        groups={name:dict(count=0,rrf_hit3=0) for name in ('both','dense_only','bm25_only','neither')}
        for r in dev_rank:
            if not gold[r['id']]['evidence']: continue
            name={(True,True):'both',(True,False):'dense_only',(False,True):'bm25_only',(False,False):'neither'}[hit(r,'dense',k),hit(r,'bm25',k)]
            groups[name]['count']+=1
            groups[name]['rrf_hit3']+=hit(r,'rrf',3)
        summary['complementarity'][str(k)]=groups
    for name,parts in {'dense':['dense'],'bm25':['bm25'],'fusion':['fusion'],'rerank':['rerank'],
                       'rrf':['dense','bm25','fusion'],'ce':['dense','bm25','fusion','rerank']}.items():
        times=[1000*sum(r['timing_s'][p] for p in parts) for r in dev_rank]
        summary['timings'][name]=dict(median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times))
    used=set()
    traces=[]
    for name,criterion in [
        ('bm25_only',lambda r:hit(r,'bm25',3) and not hit(r,'dense',3)),
        ('dense_only',lambda r:hit(r,'dense',3) and not hit(r,'bm25',3)),
        ('rrf_changed',lambda r:hit(r,'rrf',3)!=hit(r,'dense',3)),
        ('ce_changed_answer',lambda r:dev['rrf'][r['id']]['parsed']!=dev['ce'][r['id']]['parsed'])]:
        r=next((r for r in dev_rank if r['id'] not in used and criterion(r)),None)
        if not r:
            traces.append(dict(case=name,missing=True))
            continue
        i=r['id']; used.add(i)
        traces.append(dict(case=name,ranking=r,gold=gold[i],answers={m:rows[i] for m,rows in dev.items()},
                           evidence={m:rationale_present(gold[i]['evidence'],rows[i]['transmitted']) for m,rows in dev.items()}))
    save_json(out/'summary.json',summary)
    save_json(out/'traces.json',traces)
    print('Analysis complete',flush=True)

if __name__=='__main__': evaluate()
