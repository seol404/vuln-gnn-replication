import json, random

random.seed(42)

in_path = "juliet_dataset.jsonl"
train_path = "outputs/train.jsonl"
val_path   = "outputs/val.jsonl"
test_path  = "outputs/test.jsonl"

with open(in_path, "r") as f:
    rows = [json.loads(line) for line in f if line.strip()]

# keep only labeled rows (0/1)
rows = [r for r in rows if r.get("label") in (0, 1)]

random.shuffle(rows)

n = len(rows)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)
train = rows[:n_train]
val   = rows[n_train:n_train+n_val]
test  = rows[n_train+n_val:]

import os
os.makedirs("outputs", exist_ok=True)

def write_jsonl(path, data):
    with open(path, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

write_jsonl(train_path, train)
write_jsonl(val_path, val)
write_jsonl(test_path, test)

print("total", n, "train", len(train), "val", len(val), "test", len(test))