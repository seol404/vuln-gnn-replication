import torch

obj = torch.load("outputs/pyg_graphs.pt")
graphs = obj["graphs"]
print("num graphs:", len(graphs))

g = graphs[0]
print("name:", g.name)
print("x:", g.x.shape)
print("edge_index:", g.edge_index.shape)
print("y:", g.y)