import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part
LABELS=['sitting','standing','walking','lying down']
PROMPT='Look at the image of the dog. What is the dog doing? Answer with exactly one of: sitting, standing, lying down, walking.'
vertexai.init(project='cs663-dog-training', location='us-central1')
model=GenerativeModel('gemini-3.5-flash')
def norm(raw):
    t=raw.strip().lower().strip('.').strip()
    if t in LABELS: return t
    f=[x for x in LABELS if x in t]
    return f[0] if len(f)==1 else None
recs=[]
for line in open('datasets/test.jsonl'):
    if line.strip(): recs.append(json.loads(line))
print('records', len(recs), flush=True)
out=[]; errs=[]
for i, r in enumerate(recs):
    uri=None; true=None
    for c in r['contents']:
        if c['role']=='user':
            for p in c['parts']:
                if 'fileData' in p: uri=p['fileData']['fileUri']
        if c['role']=='model':
            for p in c['parts']:
                if 'text' in p: true=p['text'].strip().lower()
    try:
        resp=model.generate_content([Part.from_uri(uri, mime_type='image/jpeg'), PROMPT], generation_config={'temperature': 0.0})
        raw=resp.text
    except Exception as e:
        errs.append({'index': i, 'fileUri': uri, 'error': str(e)[:400]})
        out.append({'index': i, 'fileUri': uri, 'true_label': true, 'raw_prediction': None, 'predicted_label': None, 'correct': False, 'error': str(e)[:400]})
        print(i, 'ERROR', str(e)[:200], flush=True)
        continue
    pred=norm(raw)
    out.append({'index': i, 'fileUri': uri, 'true_label': true, 'raw_prediction': raw, 'predicted_label': pred, 'correct': pred==true, 'error': None})
    print(i, true, pred, repr(raw), flush=True)
conf={t: {p: 0 for p in LABELS} for t in LABELS}
for o in out:
    if o['true_label'] in conf and o['predicted_label'] in LABELS: conf[o['true_label']][o['predicted_label']]+=1
tot=len(out); cor=sum(o['correct'] for o in out)
per={}
for lab in LABELS:
    s=[o for o in out if o['true_label']==lab]
    c=sum(o['correct'] for o in s)
    per[lab]={'support': len(s), 'correct': c, 'accuracy': c/len(s) if s else None}
met={'model': 'gemini-3.5-flash', 'prompt': PROMPT, 'total': tot, 'correct': cor, 'overall_accuracy': cor/tot if tot else None, 'per_class': per, 'confusion_matrix': conf, 'errors': errs, 'error_count': len(errs)}
json.dump({'metrics': met, 'predictions': out}, open('base_eval_results.json', 'w'), indent=2)
print('===BASE_METRICS===')
print(json.dumps(met, indent=2))
