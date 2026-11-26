import json, sys
from google import genai
from google.genai import types
model = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/test_gcs.jsonl'
outpath = sys.argv[3] if len(sys.argv) > 3 else f"base_eval_{model}_us.json"
LABELS = ['sitting', 'standing', 'walking', 'lying down']
PROMPT = 'Look at the image of the dog. What is the dog doing? Answer with exactly one of: sitting, standing, lying down, walking.'
client = genai.Client(vertexai=True, project='cs663-dog-training', location='us')
recs = [json.loads(l) for l in open(path) if l.strip()]
print('records', len(recs), flush=True)
out = []
errs = []
for i, r in enumerate(recs):
    uri = None
    true = None
    if 'contents' in r:
        for c in r['contents']:
            if c.get('role') == 'user':
                for p in c.get('parts', []):
                    if 'fileData' in p:
                        uri = p['fileData'].get('fileUri')
                    if 'file_data' in p:
                        uri = p['file_data'].get('file_uri')
            if c.get('role') == 'model':
                for p in c.get('parts', []):
                    if 'text' in p:
                        true = p['text'].strip().lower()
    if true is None:
        true = (r.get('label') or r.get('true_label') or r.get('expected') or '')
        true = str(true).strip().lower()
    raw = None
    err = None
    try:
        resp = client.models.generate_content(model=model, contents=[types.Part.from_uri(file_uri=uri, mime_type='image/jpeg'), PROMPT])
        raw = resp.text
    except Exception as e:
        err = str(e)
        errs.append({'index': i, 'fileUri': uri, 'error': err})
    norm = raw.strip().lower() if isinstance(raw, str) else ''
    pred = norm if norm in LABELS else 'invalid'
    correct = (pred == true)
    out.append({'index': i, 'fileUri': uri, 'true_label': true, 'raw_prediction': raw, 'predicted_label': pred, 'correct': correct, 'error': err})
    print(i, true, pred, repr(raw), flush=True)
conf = {t: {p: 0 for p in LABELS} for t in LABELS}
for o in out:
    if o['true_label'] in conf and o['predicted_label'] in LABELS:
        conf[o['true_label']][o['predicted_label']] += 1
tot = len(out)
cor = sum(o['correct'] for o in out)
inv = sum(1 for o in out if o['predicted_label'] == 'invalid')
per = {}
for lab in LABELS:
    s = [o for o in out if o['true_label'] == lab]
    c = sum(o['correct'] for o in s)
    per[lab] = {'support': len(s), 'correct': c, 'accuracy': c / len(s) if s else None}
met = {'model': model, 'location': 'us', 'project': 'cs663-dog-training', 'prompt': PROMPT, 'test_file': path, 'total': tot, 'correct': cor, 'overall_accuracy': cor / tot if tot else None, 'per_class': per, 'confusion_matrix': conf, 'invalid_count': inv, 'error_count': len(errs), 'errors': errs}
data = {'metrics': met, 'predictions': out}
import os
json.dump(data, open(outpath, 'w'), indent=2)
print('===METRICS===')
print(json.dumps(met, indent=2))
print('SAVED', outpath)
