import argparse, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LR2=ROOT.parent/'lab2'
OUT=ROOT/'results'
sys.path.insert(0,str(LR2))
os.chdir(LR2)
from experiment import Experiment
from common import read_jsonl, save_json, sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument('split',choices=['dev','holdout'])
    args=p.parse_args()
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'retrieval.json')==selection['configuration_sha256']
    methods=['rrf','ce'] if args.split=='dev' else ['dense',selection['winner']]
    rankings=read_jsonl(OUT/f'{args.split}_rankings.jsonl')
    assert len(rankings)==(90 if args.split=='dev' else 210)
    previous=LR2/'runs/20261106_183751'
    for name in ['config.yaml','schema.json','prompts/rag.txt','prompts/llm_only.txt']:
        if name=='config.yaml': continue
        assert sha(LR2/name)==sha(previous/name),name
    expected_ids={r['id'] for r in rankings}
    paths=[OUT/args.split/f'{method}.jsonl' for method in methods]
    if all(path.exists() and {r['id'] for r in read_jsonl(path)}==expected_ids for path in paths):
        print(f'{args.split}: all measured responses are already saved')
        return
    exp=Experiment(str(OUT/args.split))
    try:
        exp.start_server()
        original=json.loads((previous/'metadata.json').read_text(encoding='utf-8'))
        assert exp.meta['model_sha256']==original['model_sha256']
        assert exp.meta['generation']==original['generation']
        save_json(OUT/args.split/'runtime.json',exp.meta)
        for method in methods:
            path=OUT/args.split/f'{method}.jsonl'
            done={r['id'] for r in read_jsonl(path)} if path.exists() else set()
            with path.open('a',encoding='utf-8') as f:
                for i,claim in enumerate(rankings,1):
                    if claim['id'] in done: continue
                    found=claim[method][:3]
                    context=exp.builder.build(claim['claim'],exp.docs(found,3),'rag',2048)
                    t=claim['timing_s']
                    elapsed=t['dense'] if method=='dense' else sum(t[k] for k in ['dense','bm25','fusion']+(['rerank'] if method=='ce' else []))
                    row=exp.call(claim,'rag',3,context,found,elapsed)
                    row['retriever']=method
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    f.flush()
                    print(args.split,method,i,len(rankings),row['status'],flush=True)
        exp.meta['status']='completed'
        save_json(OUT/args.split/'runtime.json',exp.meta)
    finally:
        if exp.builder: exp.builder.http.close()
        exp.client.close()
        if exp.server:
            exp.server.terminate()
            exp.server.wait(timeout=30)
        if exp.log: exp.log.close()

if __name__=='__main__': main()
