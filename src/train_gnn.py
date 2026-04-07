# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool
# from torch_geometric.loader import DataLoader

# # Load dataset
# obj = torch.load("outputs/pyg_graphs.pt")
# graphs = obj["graphs"]

# print("Loaded graphs:", len(graphs))

# # Simple train/test split
# train_graphs = graphs[:300]
# test_graphs = graphs[300:]

# train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_graphs, batch_size=16)

# # Model
# class GNN(torch.nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, 32)
#         self.conv2 = GCNConv(32, 32)
#         self.lin = torch.nn.Linear(32, 2)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)

#         x = self.conv2(x, edge_index)
#         x = F.relu(x)

#         x = global_mean_pool(x, batch)

#         return self.lin(x)


# # Initialize model
# in_channels = graphs[0].x.shape[1]
# model = GNN(in_channels)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()

# # Training loop
# for epoch in range(20):
#     model.train()
#     total_loss = 0

#     for data in train_loader:
#         optimizer.zero_grad()
#         out = model(data)
#         loss = criterion(out, data.y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch} loss {total_loss:.4f}")

# # Evaluation
# model.eval()
# correct = 0
# total = 0

# for data in test_loader:
#     out = model(data)
#     pred = out.argmax(dim=1)
#     correct += (pred == data.y).sum().item()
#     total += data.y.size(0)

# print("Test Accuracy:", correct / total)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, classification_report
import json

# ── Load graphs ──────────────────────────────────────────────
obj = torch.load("outputs/pyg_graphs.pt", weights_only=False)
all_graphs = obj["graphs"]
print(f"Total graphs loaded: {len(all_graphs)}")

# ── Build name→graph lookup ──────────────────────────────────
# Each graph has a .name attribute matching the function name in the jsonl splits
graph_by_name = {}
for g in all_graphs:
    graph_by_name[g.name] = g

# ── Load split names from jsonl files ────────────────────────
def load_names_from_jsonl(path):
    names = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            names.append(record["name"])
    return names

train_names = load_names_from_jsonl("outputs/train.jsonl")
val_names = load_names_from_jsonl("outputs/val.jsonl")
test_names = load_names_from_jsonl("outputs/test.jsonl")

# ── Match names to graph objects ─────────────────────────────
def names_to_graphs(names, graph_lookup):
    matched = []
    missing = 0
    for name in names:
        if name in graph_lookup:
            matched.append(graph_lookup[name])
        else:
            missing += 1
    if missing > 0:
        print(f"  Warning: {missing} names not found in graph data")
    return matched

train_graphs = names_to_graphs(train_names, graph_by_name)
val_graphs = names_to_graphs(val_names, graph_by_name)
test_graphs = names_to_graphs(test_names, graph_by_name)

print(f"Train: {len(train_graphs)}  Val: {len(val_graphs)}  Test: {len(test_graphs)}")

# ── Check if graphs have .name attribute, fallback to index split ─
if len(train_graphs) == 0:
    print("Name matching failed — falling back to index-based split")
    train_graphs = all_graphs[:256]
    val_graphs = all_graphs[256:311]
    test_graphs = all_graphs[311:]
    print(f"Train: {len(train_graphs)}  Val: {len(val_graphs)}  Test: {len(test_graphs)}")

# ── Class weights (handle imbalance) ─────────────────────────
train_labels = [g.y.item() for g in train_graphs]
num_safe = train_labels.count(0)
num_vuln = train_labels.count(1)
total_samples = len(train_labels)

# higher weight for minority class (vulnerable)
weight_safe = total_samples / (2.0 * num_safe)
weight_vuln = total_samples / (2.0 * num_vuln)
class_weights = torch.tensor([weight_safe, weight_vuln], dtype=torch.float)
print(f"Class weights — safe: {weight_safe:.3f}, vulnerable: {weight_vuln:.3f}")

# ── Data loaders ─────────────────────────────────────────────
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=16)
test_loader = DataLoader(test_graphs, batch_size=16)

# ── Model ────────────────────────────────────────────────────
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        return self.lin(x)


# ── Initialize ───────────────────────────────────────────────
in_channels = all_graphs[0].x.shape[1]
model = GNN(in_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

print(f"\nModel: 3-layer GCN, hidden=64, dropout=0.3")
print(f"Input features: {in_channels}")
print(f"Training for 100 epochs with early stopping (patience=15)\n")

# ── Evaluation helper ────────────────────────────────────────
def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            all_preds.extend(pred.tolist())
            all_labels.extend(data.y.tolist())

    f1 = f1_score(all_labels, all_preds, pos_label=1)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return f1, accuracy, total_loss, all_preds, all_labels


# ── Training loop with early stopping ────────────────────────
best_val_f1 = 0
patience = 15
patience_counter = 0
best_model_state = None

for epoch in range(100):
    model.train()
    total_loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validate
    val_f1, val_acc, val_loss, _, _ = evaluate(val_loader)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Train loss: {total_loss:.4f} | Val loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

# ── Load best model and evaluate on test set ─────────────────
print("\n" + "=" * 60)
print("FINAL TEST EVALUATION (best model by val F1)")
print("=" * 60)

if best_model_state is not None:
    model.load_state_dict(best_model_state)

test_f1, test_acc, _, test_preds, test_labels = evaluate(test_loader)
print(f"\nTest F1 Score:  {test_f1:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Best Val F1:    {best_val_f1:.4f}")
print(f"\nBaseline (TF-IDF + LR): F1 = 0.7826")
print(f"GNN:                    F1 = {test_f1:.4f}")

if test_f1 > 0.7826:
    print(">> GNN outperforms the baseline!")
else:
    print(">> GNN does not outperform the baseline yet.")

print("\nDetailed Classification Report:")
print(classification_report(test_labels, test_preds, target_names=["safe", "vulnerable"]))

# ── Save model ───────────────────────────────────────────────
torch.save(model.state_dict(), "outputs/gnn_model.pt")
print("Model saved to outputs/gnn_model.pt")