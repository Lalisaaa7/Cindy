import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_add_pool

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# GAT with MLP Model
class GAT_with_MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, gat_out_channels, mlp_hidden=64, out_classes=2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=8, concat=True)
        self.gat2 = GATConv(hidden_channels * 8, gat_out_channels, heads=8, concat=False)

        self.mlp = nn.Sequential(
            nn.Linear(gat_out_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_classes)
        )

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = self.mlp(x)
        return x



class GINE_with_MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, gine_out_channels, mlp_hidden=64, out_classes=2):
        super().__init__()
        self.gine1 = GINEConv(in_channels, hidden_channels)
        self.gine2 = GINEConv(hidden_channels, gine_out_channels)

        self.mlp = nn.Sequential(
            nn.Linear(gine_out_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_classes)
        )

    def forward(self, x, edge_index, batch):
        x = self.gine1(x, edge_index)
        x = F.relu(x)
        x = self.gine2(x, edge_index)
        x = global_add_pool(x, batch)  # 聚合池化操作
        x = self.mlp(x)
        return x


# 训练函数
def train_model(balanced_data, feature_dim, device):
    model = GAT_with_MLP(
        in_channels=feature_dim,
        hidden_channels=64,
        gat_out_channels=64,
        mlp_hidden=64,
        out_classes=2
    ).to(device)



    # 在训练完成后保存 GAT 模型
    torch.save(model.state_dict(), 'best_gat_model.pt')

    # 加载模型时加载 GAT 权重
    model.load_state_dict(torch.load('best_gat_model.pt'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(50):
        total_loss = 0
        correct = 0
        total = 0
        for data in balanced_data:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_nodes
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(balanced_data):.4f}, Acc = {correct / total:.4f}")
    return model


def test_model(model, test_data, device):
    print("\n Evaluating on test set...")
    model.eval()
    loader = DataLoader(test_data, batch_size=1, shuffle=False)

    total, correct = 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_nodes

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    mcc = matthews_corrcoef(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")

    return accuracy, f1, mcc