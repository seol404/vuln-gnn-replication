import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, classification_report
import random
import numpy as np

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Load graphs ──────────────────────────────────────────────
obj = torch.load("outputs/pyg_graphs.pt", weights_only=False)
all_graphs = obj["graphs"]
print(f"Total graphs: {len(all_graphs)}")

# ── Split (use same indices as progress_report: 256/55/56) ───
random.shuffle(all_graphs)
train_graphs = all_graphs[:256]
val_graphs = all_graphs[256:311]
test_graphs = all_graphs[311:]

train_labels = [g.y.item() for g in train_graphs]
val_labels = [g.y.item() for g in val_graphs]
test_labels_count = [g.y.item() for g in test_graphs]
print(f"Train: {len(train_graphs)} (vuln: {sum(train_labels)})")
print(f"Val:   {len(val_graphs)} (vuln: {sum(val_labels)})")
print(f"Test:  {len(test_graphs)} (vuln: {sum(test_labels_count)})")

# ── Class weights ────────────────────────────────────────────
num_safe = train_labels.count(0)
num_vuln = train_labels.count(1)
weight_safe = len(train_labels) / (2.0 * num_safe)
weight_vuln = len(train_labels) / (2.0 * num_vuln)
class_weights = torch.tensor([weight_safe, weight_vuln], dtype=torch.float)
print(f"Class weights — safe: {weight_safe:.3f}, vuln: {weight_vuln:.3f}")

# ── Data loaders ─────────────────────────────────────────────
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32)
test_loader = DataLoader(test_graphs, batch_size=32)

# ── Models ───────────────────────────────────────────────────
class GCN_2Layer(torch.nn.Module):
    """Simple 2-layer GCN — better for small graphs"""
    def __init__(self, in_channels, hidden=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden * 2, 2)  # concat mean+max pool
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        # dual pooling: captures both average and extreme features
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.lin(x)


class GAT_2Layer(torch.nn.Module):
    """2-layer GAT — attention helps focus on important nodes"""
    def __init__(self, in_channels, hidden=32, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=0.4)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=0.4)
        self.lin = torch.nn.Linear(hidden * 2, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.lin(x)


# ── Training function ────────────────────────────────────────
def train_and_evaluate(model, model_name, lr=0.005, epochs=150, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
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
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for data in val_loader:
                pred = model(data).argmax(dim=1)
                val_preds.extend(pred.tolist())
                val_true.extend(data.y.tolist())
        val_f1 = f1_score(val_true, val_preds, pos_label=1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Test with best model
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for data in test_loader:
            pred = model(data).argmax(dim=1)
            test_preds.extend(pred.tolist())
            test_true.extend(data.y.tolist())

    test_f1 = f1_score(test_true, test_preds, pos_label=1)
    test_acc = sum(p == l for p, l in zip(test_preds, test_true)) / len(test_true)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Best Val F1:  {best_val_f1:.4f}")
    print(f"  Test F1:      {test_f1:.4f}")
    print(f"  Test Acc:     {test_acc:.4f}")
    print(f"  Stopped at epoch: {epoch+1}")
    print(classification_report(test_true, test_preds, target_names=["safe", "vulnerable"]))

    return test_f1, model


# ── Run experiments ──────────────────────────────────────────
in_channels = all_graphs[0].x.shape[1]
print(f"\nInput features: {in_channels}")
print(f"Avg graph size: ~28 nodes, ~27 edges")
print(f"\nRunning experiments...\n")

results = {}

# Experiment 1: 2-layer GCN
print("Training 2-layer GCN...")
model_gcn = GCN_2Layer(in_channels, hidden=32)
f1_gcn, model_gcn = train_and_evaluate(model_gcn, "2-Layer GCN (hidden=32)")
results["GCN"] = f1_gcn

# Experiment 2: 2-layer GAT
print("Training 2-layer GAT...")
model_gat = GAT_2Layer(in_channels, hidden=16, heads=4)
f1_gat, model_gat = train_and_evaluate(model_gat, "2-Layer GAT (hidden=16, heads=4)")
results["GAT"] = f1_gat

# Experiment 3: GCN with higher hidden dim
print("Training 2-layer GCN (larger)...")
model_gcn_lg = GCN_2Layer(in_channels, hidden=64)
f1_gcn_lg, model_gcn_lg = train_and_evaluate(model_gcn_lg, "2-Layer GCN (hidden=64)")
results["GCN-64"] = f1_gcn_lg

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY: ALL MODELS vs BASELINE")
print("=" * 60)
print(f"  Baseline (TF-IDF + LR):   F1 = 0.7826")
for name, f1 in results.items():
    marker = " << best" if f1 == max(results.values()) else ""
    beat = "✓" if f1 > 0.7826 else "✗"
    print(f"  {name:25s}  F1 = {f1:.4f}  {beat}{marker}")

# Save best model
best_name = max(results, key=results.get)
best_f1 = results[best_name]
print(f"\nBest model: {best_name} (F1 = {best_f1:.4f})")

if best_name == "GCN":
    torch.save(model_gcn.state_dict(), "outputs/gnn_model_best.pt")
elif best_name == "GAT":
    torch.save(model_gat.state_dict(), "outputs/gnn_model_best.pt")
else:
    torch.save(model_gcn_lg.state_dict(), "outputs/gnn_model_best.pt")
print("Saved to outputs/gnn_model_best.pt")