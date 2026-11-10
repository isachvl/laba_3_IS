from pathlib import Path
import time
from analyze_lr3 import evaluate
from report_lr3 import main as report
from common import read_jsonl

ROOT=Path(__file__).resolve().parent
def ready():
    out=ROOT/'results'
    selected=out/'selection.json'
    if not selected.exists(): return False
    import json
    winner=json.loads(selected.read_text(encoding='utf-8'))['winner']
    expected={'dev/rrf.jsonl':90,'dev/ce.jsonl':90,
              'holdout/dense.jsonl':210,f'holdout/{winner}.jsonl':210}
    for path,n in expected.items():
        file=out/path
        if not file.exists(): return False
        try: rows=read_jsonl(file)
        except ValueError: return False
        if len(rows)<n: return False
        if len(rows)!=n: raise RuntimeError(f'Unexpected count: {file}')
    return True

if __name__=='__main__':
    deadline=time.monotonic()+8*3600
    while not ready():
        if time.monotonic()>deadline: raise TimeoutError('Generation incomplete')
        time.sleep(20)
    evaluate()
    report()
    import pypdfium2
    doc=pypdfium2.PdfDocument(ROOT/'report.pdf')
    assert 3<=len(doc)<=4, len(doc)
    previews=ROOT/'results'/'report_preview'
    previews.mkdir(exist_ok=True)
    for index,page in enumerate(doc,1):
        page.render(scale=1.25).to_pil().save(previews/f'{index}.png')
    print('Report generated and rendered:',len(doc),'pages',flush=True)
