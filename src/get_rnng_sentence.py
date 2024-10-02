import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    data = json.loads(line)
    if "orig_tokens" in data:
        print("".join(data["orig_tokens"]).replace("\u2581", " "))