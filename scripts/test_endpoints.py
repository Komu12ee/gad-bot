import urllib.request
import json

urls = [
    'http://localhost:3003/api/officers?q=Sammi',
    'http://localhost:3003/api/officer/vikas_sheel',
    'http://localhost:3003/api/officers/secretaries',
    'http://localhost:3003/api/graph/semantic_search?q=Sammi+Abedi'
]

for u in urls:
    try:
        req = urllib.request.urlopen(u, timeout=5)
        data = json.loads(req.read().decode('utf-8'))
        print(f"OK: {u} -> success={data.get('success')}, keys={list(data.keys())[:5]}")
    except Exception as e:
        print(f"FAIL: {u} -> {e}")
