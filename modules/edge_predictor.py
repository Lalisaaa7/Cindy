import torch
import torch.nn as nn
from torch_geometric.data import Data


class EdgePredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()  # 输出连接概率
        )

    def forward(self, xi, xj):
        x = torch.cat([xi, xj], dim=-1)
        return self.model(x)  # 输出: [N, 1]


def connect_generated_nodes(original_data, generated_x, edge_predictor, device, threshold=0.8):
    """
    将扩散生成的新节点与原图中的节点建立边，构建增强图 G*
    - original_data: 原始图 Data 对象
    - generated_x: Tensor [num_new_nodes, dim]
    - edge_predictor: 边分类器
    - threshold: 概率阈值（高于此值则建立边）
    """
    original_x = original_data.x.to(device)
    generated_x = generated_x.to(device)
    edge_index = original_data.edge_index.clone().tolist()

    new_node_start_idx = original_x.size(0)
    num_new_nodes = generated_x.size(0)

    edge_predictor.eval()
    new_edges = []

    with torch.no_grad():
        for i_new in range(num_new_nodes):
            xi = generated_x[i_new].unsqueeze(0).repeat(original_x.size(0), 1)
            xj = original_x
            prob = edge_predictor(xi, xj).squeeze()
            selected = (prob > threshold).nonzero(as_tuple=False).view(-1)
            for idx in selected:
                i_old = idx.item()
                new_idx = new_node_start_idx + i_new
                new_edges.append([i_old, new_idx])
                new_edges.append([new_idx, i_old])

    all_x = torch.cat([original_x.cpu(), generated_x.cpu()], dim=0)
    all_y = torch.cat([original_data.y.cpu(), torch.ones(num_new_nodes, dtype=torch.long)], dim=0)

    edge_index.extend(new_edges)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=all_x, edge_index=edge_index, y=all_y)