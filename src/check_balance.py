import json
from collections import Counter

def count(path):
    c = Counter()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            c[r["label"]] += 1
    return c

for p in ["outputs/train.jsonl", "outputs/val.jsonl", "outputs/test.jsonl"]:
    print(p, count(p))