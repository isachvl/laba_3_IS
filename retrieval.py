import json, math, re
from collections import Counter, defaultdict

def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class BM25:
    def __init__(self, docs):
        counts = {i: Counter(tokens(t)) for i,t in docs.items()}
        average = sum(map(sum, (c.values() for c in counts.values()))) / len(docs)
        self.postings = defaultdict(list)
        for i,c in counts.items():
            norm = 1.2*(.25+.75*sum(c.values())/average)
            for term, frequency in c.items():
                self.postings[term].append((i, 2.2*frequency/(frequency+norm)))
        self.n = len(docs)

    def search(self, query):
        scores = defaultdict(float)
        for term in sorted(set(tokens(query))):
            posting = self.postings.get(term, [])
            idf = math.log1p((self.n-len(posting)+.5)/(len(posting)+.5))
            for i, weight in posting:
                scores[i] += idf*weight
        return ranked(scores)

def ranked(scores):
    return [dict(doc_id=i, score=float(scores[i]), rank=r) for r,i in
            enumerate(sorted(scores, key=lambda i:(-scores[i],int(i)))[:20],1)]

def fuse(dense, bm25, c=60):
    contributions = defaultdict(dict)
    for name, rows in [('dense',dense),('bm25',bm25)]:
        for r in rows[:20]:
            contributions[r['doc_id']][name] = 1/(c+r['rank'])
    rows = ranked({i:sum(parts.values()) for i,parts in contributions.items()})
    for r in rows:
        r['contributions'] = {n:contributions[r['doc_id']].get(n,0.) for n in ('dense','bm25')}
    return rows

def test():
    b = BM25({'1':'aa aa bb','2':'aa bb','3':'cc'})
    expected = math.log1p(1.5/2.5)*4.4/(2+1.2*(.25+.75*3/2))
    assert math.isclose(b.search('aa aa')[0]['score'],expected)
    assert b.search('unknown') == []
    assert tokens('ALDH1 IL-6') == ['aldh1','il','6']
    assert fuse(ranked({'1':2,'2':1}),ranked({'3':2,'2':1}))[0]['doc_id']=='2'

if __name__=='__main__':
    test()
    print('BM25 and RRF checks passed')
