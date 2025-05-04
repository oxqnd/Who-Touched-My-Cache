import torch
from torch_geometric.data import Data
import glob
import os

GRAPH_DIR = './data/graphs'
OUT_PATH = os.path.join(GRAPH_DIR, 'graph_all.pt')

def merge_graphs():
    graph_files = sorted(glob.glob(os.path.join(GRAPH_DIR, 'graph_*.pt')))
    print(f"[🔍] {len(graph_files)}개 그래프 병합 중...")

    all_x, all_y, all_edge_index = [], [], []
    node_offset = 0

    for file in graph_files:
        g = torch.load(file)
        n = g.x.size(0)

        all_x.append(g.x)
        all_y.append(g.y)

        # 엣지 인덱스도 위치 보정 필요
        edge = g.edge_index + node_offset
        all_edge_index.append(edge)

        node_offset += n

    big_x = torch.cat(all_x, dim=0)
    big_y = torch.cat(all_y, dim=0)
    big_edge_index = torch.cat(all_edge_index, dim=1)

    new_graph = Data(x=big_x, y=big_y, edge_index=big_edge_index)
    torch.save(new_graph, OUT_PATH)
    print(f"[✓] 저장 완료: {OUT_PATH} (노드 수: {new_graph.num_nodes})")

if __name__ == "__main__":
    merge_graphs()
