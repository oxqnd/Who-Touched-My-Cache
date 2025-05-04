import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.data import Data
from models import GCN, GAT, GraphSAGE, MLP
from sklearn.preprocessing import StandardScaler

# 📁 그래프 경로
GRAPH_PATH = "./data/graphs/graph_all.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 64
NUM_EPOCHS = 200
LR = 0.01


def load_graph():
    data = torch.load(GRAPH_PATH)

    # ✅ 정규화
    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x), dtype=torch.float)

    return data


def split_nodes(data, train_ratio=0.8):
    num_nodes = data.num_nodes
    num_train = int(num_nodes * train_ratio)
    perm = torch.randperm(num_nodes)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[perm[:num_train]] = True
    data.test_mask = ~data.train_mask
    return data


def train_and_eval(model_class, model_name, data):
    model = model_class(data.num_node_features, HIDDEN_DIM, 2).to(DEVICE)
    data = data.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\n[🚀] {model_name} 학습 시작")

    best_f1 = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

        y_true = data.y[data.test_mask].cpu()
        y_pred = pred[data.test_mask].cpu()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        best_f1 = max(best_f1, f1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] Loss: {loss.item():.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    print(f"[✓] {model_name} 최고 F1-score: {best_f1:.4f}")


if __name__ == "__main__":
    data = load_graph()
    data = split_nodes(data)

    counts = torch.bincount(data.y)
    label_dist = {i: int(c) for i, c in enumerate(counts)}
    print(f"\n[디버깅] 레이블 분포: {label_dist}")
    print(f"[디버깅] 입력 피처 범위: min={data.x.min():.4f}, max={data.x.max():.4f}, std={data.x.std():.4f}")
    print(f"[디버깅] 총 노드 수: {data.num_nodes}")

    models = [
        (GCN, "GCN"),
        (GAT, "GAT"),
        (GraphSAGE, "GraphSAGE"),
        (MLP, "MLP")
    ]

    for model_cls, model_name in models:
        train_and_eval(model_cls, model_name, data)
