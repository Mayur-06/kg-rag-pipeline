import httpx
import json
r=httpx.post("http://localhost:11434/api/generate", json={"model":"llama3.2","prompt":"Hello","options":{"temperature":0.7},"stream":False}, timeout=20)
print("status", r.status_code)
print(r.text)
print(json.dumps(r.json(), indent=2))
