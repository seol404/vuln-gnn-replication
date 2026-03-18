import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

# Load dataset
obj = torch.load("outputs/pyg_graphs.pt")
graphs = obj["graphs"]

print("Loaded graphs:", len(graphs))

# Simple train/test split
train_graphs = graphs[:300]
test_graphs = graphs[300:]

train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=16)

# Model
class GNN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 32)
        self.lin = torch.nn.Linear(32, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        return self.lin(x)


# Initialize model
in_channels = graphs[0].x.shape[1]
model = GNN(in_channels)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(20):
    model.train()
    total_loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} loss {total_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

for data in test_loader:
    out = model(data)
    pred = out.argmax(dim=1)
    correct += (pred == data.y).sum().item()
    total += data.y.size(0)

print("Test Accuracy:", correct / total)