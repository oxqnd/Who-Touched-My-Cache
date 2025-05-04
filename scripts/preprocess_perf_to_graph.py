import os
import re
import torch
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

DATA_DIR = "./data"
GRAPH_DIR = "./data/graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

perf_files = [("normal_perf.txt", 0), ("attack_perf.txt", 1)]

event_list = [
    "cache-references",
    "cache-misses",
    "branches",
    "branch-misses"
]


def parse_perf_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if re.match(r'^\s*\d', line):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    time_ms = float(parts[0])
                    count_str = parts[1].replace(',', '')
                    if "<not" in count_str:
                        continue
                    count = float(count_str)

                    for p in reversed(parts[2:]):
                        if p in event_list:
                            data.append((time_ms, p, count))
                            break
                except:
                    continue
    return data


def group_by_time(parsed_data):
    grouped = defaultdict(dict)
    for t, event, count in parsed_data:
        grouped[round(t, 3)][event] = count
    return grouped


def build_graph_from_perf(grouped_data, label):
    time_points = sorted(grouped_data.keys())
    x = []
    for t in time_points:
        feat = [grouped_data[t].get(ev, 0.0) for ev in event_list]
        x.append(feat)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float)

    edge_list = [[i, i + 1] for i in range(len(x_tensor) - 1)]
    edge_index = torch.tensor(edge_list + [[j, i] for i, j in edge_list], dtype=torch.long).t()

    y = torch.tensor([label] * len(x_tensor), dtype=torch.long)
    return Data(x=x_tensor, edge_index=edge_index, y=y)


def main():
    for i, (filename, label) in enumerate(perf_files):
        path = os.path.join(DATA_DIR, filename)
        parsed = parse_perf_file(path)
        grouped = group_by_time(parsed)
        graph = build_graph_from_perf(grouped, label)

        save_path = os.path.join(GRAPH_DIR, f"graph_{label}_{i}.pt")
        torch.save(graph, save_path)
        print(f"[✓] 저장 완료: {save_path} (노드 수: {graph.num_nodes})")


if __name__ == "__main__":
    main()
