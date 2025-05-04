import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from sklearn.metrics import f1_score
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, num_classes=2):
        super(DeepGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden1)
        self.bn1 = torch.nn.BatchNorm1d(hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.bn2 = torch.nn.BatchNorm1d(hidden2)
        self.conv3 = GCNConv(hidden2, num_classes)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.conv3(x, edge_index)
        return x


def load_graphs(graph_dir):
    return [torch.load(f).to(device) for f in glob(os.path.join(graph_dir, "*.pt"))]


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        acc = (pred == data.y).sum().item() / data.num_nodes
        f1 = f1_score(data.y.cpu(), pred.cpu(), average="macro")
    return acc, f1


def main():
    graphs = load_graphs("./data/graphs")
    data = Batch.from_data_list(graphs)

    print(f"\n[디버깅] 레이블 분포: {dict(zip(*torch.unique(data.y, return_counts=True)))}")
    print(f"[디버깅] 입력 피처 범위: min={data.x.min():.4f}, max={data.x.max():.4f}, std={data.x.std():.4f}")

    model = DeepGCN(input_dim=data.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    print(f"\n[!] 학습 시작: {data.num_nodes}개 노드, 입력 차원 {data.num_node_features}")
    for epoch in range(1, 201):
        loss = train(model, data, optimizer)
        acc, f1 = evaluate(model, data)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    print("\n[✓] 학습 완료")


if __name__ == "__main__":
    main()
