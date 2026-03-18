import json
from pathlib import Path
from collections import Counter

import torch
from torch_geometric.data import Data

INPUT_PATH = Path("ast_graphs.jsonl")
OUTPUT_PATH = Path("outputs/pyg_graphs.pt")


def build_type_vocab(rows):
    types = set()
    for row in rows:
        for node in row["nodes"]:
            types.add(node["type"])
    return {t: i for i, t in enumerate(sorted(types))}


def one_hot(index, size):
    x = torch.zeros(size, dtype=torch.float)
    x[index] = 1.0
    return x


def row_to_data(row, type_vocab):
    nodes = row["nodes"]
    edges = row["edges"]
    label = row["label"]

    # map original Joern node ids -> 0..N-1
    id_map = {node["id"]: i for i, node in enumerate(nodes)}

    # node feature matrix x
    x = torch.stack([
        one_hot(type_vocab[node["type"]], len(type_vocab))
        for node in nodes
    ])

    # edge_index
    edge_pairs = []
    for e in edges:
        src = e["src"]
        dst = e["dst"]
        if src in id_map and dst in id_map:
            edge_pairs.append([id_map[src], id_map[dst]])

    if len(edge_pairs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    y = torch.tensor([label], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    data.name = row["name"]
    data.code = row["code"]
    return data


def main():
    rows = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} graph records")

    type_vocab = build_type_vocab(rows)
    print(f"Node type vocab size: {len(type_vocab)}")
    print("Top node types:", list(type_vocab.keys())[:15])

    graphs = [row_to_data(row, type_vocab) for row in rows]

    label_counts = Counter(int(g.y.item()) for g in graphs)
    print("Label counts:", label_counts)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "graphs": graphs,
            "type_vocab": type_vocab
        },
        OUTPUT_PATH
    )
    print(f"Saved {len(graphs)} PyG graphs to {OUTPUT_PATH}")

    # show one example
    g = graphs[0]
    print("--- Example graph ---")
    print("name:", g.name)
    print("x shape:", tuple(g.x.shape))
    print("edge_index shape:", tuple(g.edge_index.shape))
    print("label:", int(g.y.item()))


if __name__ == "__main__":
    main()