import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def pct(x): return f'{100*x:.1f}%'
def main():
    s=read('summary.json')
    traces=read('traces.json')
    winner=s['selection']['winner']
    names={'dense':'Dense','bm25':'BM25','rrf':'RRF','ce':'RRF + CE'}
    fonts=Path('C:/Windows/Fonts')
    pdfmetrics.registerFont(TTFont('Arial',str(fonts/'arial.ttf')))
    pdfmetrics.registerFont(TTFont('ArialBold',str(fonts/'arialbd.ttf')))
    regular=ParagraphStyle('body',fontName='Arial',fontSize=10,leading=14,spaceAfter=7)
    compact=ParagraphStyle('compact',parent=regular,fontSize=8,leading=10,spaceAfter=4)
    title=ParagraphStyle('title',parent=regular,fontName='ArialBold',fontSize=20,leading=24,spaceAfter=14)
    heading=ParagraphStyle('heading',parent=regular,fontName='ArialBold',fontSize=12,leading=16,spaceBefore=8)
    story=[]
    def p(text,style=regular): story.append(Paragraph(escape(text),style))
    def table(rows,widths):
        t=Table([[Paragraph(escape(str(v)),compact) for v in row] for row in rows],colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2ebf6')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fc')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([t,Spacer(1,9)])
    p('ЛР3. Улучшение поиска в RAG',title)
    p('SciFact и разбиение из ЛР2. Dev: 90 утверждений, из них 60 с обоснованием. Holdout: 210 и 128 соответственно. Бюджет 2048 входных токенов, top-3, модель и промпты ЛР2.')
    p('Прогноз',heading)
    p('BM25 должен помочь с редкими обозначениями, Dense — с перефразированием. RRF может сохранить находки обоих методов. Переранжирование может улучшить порядок выдачи, но ответ зависит и от сборки контекста, и от модели.')
    p('Поиск на dev',heading)
    rows=[['Вариант','Hit@3','Recall@3','MRR@3','Recall@20']]
    for m,x in s['retrieval']['dev'].items(): rows.append([names[m],pct(x['3']['hit']),pct(x['3']['recall']),pct(x['3']['rr']),pct(x['20']['recall'])])
    table(rows,[127,85,85,85,108])
    p('Дополнительность Dense и BM25',heading)
    rows=[['Группа','На глубине 3','RRF Hit@3','На глубине 20','RRF Hit@3']]
    for key,label in [('both','Оба'),('dense_only','Только Dense'),('bm25_only','Только BM25'),('neither','Ни один')]:
        a,b=(s['complementarity'][k][key] for k in ('3','20'))
        rows.append([label,a['count'],a['rrf_hit3'],b['count'],b['rrf_hit3']])
    table(rows,[127,92,84,103,84])
    p('По правилу MRR@3 → Recall@3 → простота выбран '+names[winner]+'. Выбор записан до проверки holdout.')
    story.append(PageBreak())
    p('Ответы и время поиска',title)
    rows=[['Часть','Вариант','Accuracy','Ссылка','Обоснование','SUPPORT','CONTRADICT','NOINFO']]
    for split,methods in s['generation'].items():
        for m,x in methods.items():
            rows.append([split,names[m],pct(x['accuracy']),pct(x['citation']),pct(x['evidence']),*(pct(x['class_recall'][l]) for l in ('SUPPORT','CONTRADICT','NOINFO'))])
    table(rows,[47,64,57,53,76,63,74,56])
    p('На dev RRF + CE относительно Dense исправил '+str(len(s['paired']['dev']['fixed']))+' ответов и испортил '+str(len(s['paired']['dev']['broken']))+'. На holdout выбранный вариант исправил '+str(len(s['paired']['holdout']['fixed']))+' и испортил '+str(len(s['paired']['holdout']['broken']))+'.')
    p('Поиск на holdout, 128 утверждений с обоснованием',heading)
    rows=[['Вариант','Hit@3','Recall@3','MRR@3','Recall@20']]
    for m in ('dense',winner):
        x=s['retrieval']['holdout'][m]
        rows.append([names[m],pct(x['3']['hit']),pct(x['3']['recall']),pct(x['3']['rr']),pct(x['20']['recall'])])
    table(rows,[127,85,85,85,108])
    p('Задержка поиска на dev, мс на запрос',heading)
    rows=[['Этап / вариант','Медиана','Минимум','Максимум']]
    for m,x in s['timings'].items(): rows.append([names.get(m,{'fusion':'Только объединение','rerank':'Только cross-encoder'}.get(m,m)),*(f'{x[k]:.2f}' for k in ('median_ms','min_ms','max_ms'))])
    table(rows,[196,98,98,98])
    p('Для RRF и RRF + CE полное время включает оба исходных поиска. Cross-encoder оценивает 1800 пар на dev и 4200 на holdout. Время чтения результатов не учитывалось.')
    story.append(PageBreak())
    p('Четыре трейса',title)
    labels={'bm25_only':'Находка BM25','dense_only':'Находка Dense','rrf_changed':'RRF изменил первые три результата','ce_changed_answer':'Переранжирование изменило ответ'}
    md=['# Трейсы ЛР3']
    for index,t in enumerate(traces):
        if index==2: story.append(PageBreak()); p('Трейсы и выводы',title)
        p(labels[t['case']],heading)
        if t.get('missing'):
            p('Такой случай в измерениях не найден.',compact)
            continue
        r=t['ranking']; g=t['gold']
        p(f"ID {r['id']}. {r['claim']}",compact)
        focus=set(g['evidence']) | {v['doc_id'] for v in r['ce'][:3]}
        for m in ('dense','bm25','rrf','ce'):
            entries=[x for x in r[m] if x['rank']<=3 or x['doc_id'] in focus]
            detail=', '.join(f"{x['doc_id']} (#{x['rank']}"+ (f", D={x['contributions']['dense']:.4f}, B={x['contributions']['bm25']:.4f}" if m=='rrf' else '')+')' for x in entries)
            p(names[m]+': '+detail,compact)
        p('Обоснование передано: '+', '.join(f"{names[m]} — {'да' if ok else 'нет'}" for m,ok in t['evidence'].items())+'.',compact)
        if g['evidence']:
            stages=[]
            for m,present in t['evidence'].items():
                found=bool(set(g['evidence']) & {x['doc_id'] for x in r[m][:3]})
                state='обоснование в промпте' if present else ('потеря при сборке' if found else 'нет нужного документа в top-3')
                stages.append(names[m]+' — '+state)
            p('Этап: '+', '.join(stages)+'.',compact)
        for m,a in t['answers'].items():
            p(names[m]+' → '+str(a['raw_response']).replace('\n',' ')+'; переданы '+str(a['transmitted']),compact)
        md.extend(['',f"## {labels[t['case']]} — ID {r['id']}",'',r['claim'],'','```json',json.dumps(t,ensure_ascii=False,indent=2),'```'])
    (ROOT/'traces.md').write_text('\n'.join(md),encoding='utf-8')
    p('Выводы',heading)
    for split,methods in s['generation'].items():
        alternative='ce' if split=='dev' else winner
        delta=100*(methods[alternative]['accuracy']-methods['dense']['accuracy'])
        p(f'{split}: {names[alternative]} относительно Dense изменил Accuracy на {delta:+.1f} п.п.')
    delta=s['generation']['holdout'][winner]['accuracy']-s['generation']['holdout']['dense']['accuracy']
    p(('На holdout выше Accuracy у '+names[winner]+'.') if delta>0 else 'По Accuracy на holdout нет основания заменять Dense.')
    h=s['generation']['holdout']
    slow=s['timings'][winner]['median_ms']/s['timings']['dense']['median_ms']
    noinfo=100*(h[winner]['class_recall']['NOINFO']-h['dense']['class_recall']['NOINFO'])
    p(f'На holdout выигрыш составил {len(s["paired"]["holdout"]["fixed"])-len(s["paired"]["holdout"]["broken"])} ответа из 210. Полнота NOINFO изменилась на {noinfo:+.1f} п.п. Медиана поиска на dev больше в {slow:.1f} раза. При приоритете скорости разумно оставить Dense.',compact)
    p('Итоговый выбор оценивается на holdout после фиксации параметров. Метрики поиска и доставки обоснования помогают отделить эффект ранжирования от ошибок модели.')
    p('Файлы для проверки',heading)
    p('selection.json хранит правило и выбранный вариант. manifest.json хранит данные и ревизии. Списки dev и holdout, полные ответы и summary.json находятся в results. В traces.md есть переданные предложения и ответы для четырёх случаев.')
    p('Код и обработка результатов подготовлены с помощью Codex. Таблицы собраны из сохранённых измерений.')
    def footer(canvas, doc):
        canvas.setFont('Arial',8)
        canvas.drawString(50,25,'ЛР3 · SciFact')
        canvas.drawRightString(545,25,str(doc.page))
    SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)

if __name__=='__main__': main()

