import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch
import esm

# 加载模型
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()  # 设置为评估模式（关闭dropout等）

print("torch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("is cuda available:", torch.cuda.is_available())
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef


# ---------- 全局 K-mer 词表 ----------
GLOBAL_VOCAB = {}

# ---------- Simple Diffusion Generator ----------
class SimpleDiffusionGenerator(nn.Module):
    def __init__(self, input_dim, noise_dim=32):
        super(SimpleDiffusionGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 64)
        self.fc2 = nn.Linear(64, input_dim)

    def forward(self, noise):
        x = F.relu(self.fc1(noise))
        return torch.sigmoid(self.fc2(x))

def balance_dataset(original_data, augmented_data):
    """
    将原始数据与生成的正类样本合并，生成平衡的数据集。
    """
    balanced_data = original_data + augmented_data
    return balanced_data

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
        # xi, xj: shape [N, dim]
        x = torch.cat([xi, xj], dim=-1)
        return self.model(x)  # 输出: [N, 1]



class DiffusionModel:
    def __init__(self, input_dim, device='cpu'):
        self.generator = SimpleDiffusionGenerator(input_dim).to(device)
        self.device = device

    def train_on_positive_samples(self, all_data, batch_size=256, epochs=10):
        """
        用已有正类样本训练生成器模型（支持分批训练，防止OOM）
        """
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # 收集所有正类样本特征
        positive_vectors = []
        for data in all_data:
            pos_feats = data.x[data.y == 1]
            if pos_feats.size(0) > 0:
                positive_vectors.append(pos_feats.cpu())  # 先留在 CPU
        if not positive_vectors:
            print("⚠ 没有可用于训练的正类样本")
            return

        full_pos_data = torch.cat(positive_vectors, dim=0)  # 留在 CPU
        data_size = full_pos_data.size(0)

        print(f" 正类样本总量：{data_size}，使用 batch_size={batch_size} 训练扩散生成器")

        self.generator.train()
        for epoch in range(epochs):
            perm = torch.randperm(data_size)
            epoch_loss = 0
            for i in range(0, data_size, batch_size):
                indices = perm[i:i + batch_size]
                batch_data = full_pos_data[indices].to(self.device)
                noise = torch.randn((batch_data.size(0), 32)).to(self.device)

                generated = self.generator(noise)
                loss = criterion(generated, batch_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} - Generator Loss: {epoch_loss:.4f}")

            # 清理显存防止碎片堆积
            torch.cuda.empty_cache()
            import gc
            gc.collect()




    def generate_positive_sample(self, num_samples=10, batch_size=1024):
        self.generator.eval()
        generated_all = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                noise = torch.randn((current_batch_size, 32)).to(self.device)
                generated = self.generator(noise)
                generated_all.append(generated.cpu())  # 放在 CPU，避免显存爆炸

        return torch.cat(generated_all, dim=0)


# ---------- K-mer Encoding ----------

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
    edge_index = original_data.edge_index.clone().tolist()  # 原边

    new_node_start_idx = original_x.size(0)
    num_new_nodes = generated_x.size(0)

    edge_predictor.eval()
    new_edges = []

    with torch.no_grad():
        for i_new in range(num_new_nodes):
            xi = generated_x[i_new].unsqueeze(0).repeat(original_x.size(0), 1)  # [N_old, dim]
            xj = original_x
            prob = edge_predictor(xi, xj).squeeze()  # [N_old]

            # 选择连接的老节点（prob > threshold）
            selected = (prob > threshold).nonzero(as_tuple=False).view(-1)
            for idx in selected:
                i_old = idx.item()
                new_idx = new_node_start_idx + i_new
                new_edges.append([i_old, new_idx])
                new_edges.append([new_idx, i_old])

    # 新特征 + 标签
    all_x = torch.cat([original_x.cpu(), generated_x.cpu()], dim=0)
    all_y = torch.cat([original_data.y.cpu(), torch.ones(num_new_nodes, dtype=torch.long)], dim=0)

    # 构建完整边集
    edge_index.extend(new_edges)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=all_x, edge_index=edge_index, y=all_y)

def get_esm_features(sequence, model, batch_converter, device='cpu'):
    """
    输入：蛋白质序列字符串
    输出：每个残基的表示向量 Ai（Tensor，shape: [L, 1280]）
    """
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    # 提取每个残基的表示（排除起始符 <cls> 和终止符 <eos>）
    token_representations = results["representations"][33]
    residue_representations = token_representations[0, 1:-1]  # [1, L+2, 1280] → [L, 1280]

    return residue_representations.cpu()



# ---------- Graph Construction ----------
def sequence_to_graph(sequence, label, k=3, window_size=5):

    x = get_esm_features(sequence, esm_model, batch_converter, device='cpu')

    y = torch.tensor([int(i) for i in label], dtype=torch.long)

    num_nodes = x.size(0)
    assert len(y) == num_nodes, f"label length mismatch after k-mer: {len(label)} vs {num_nodes}"
    edge_index = []
    for i in range(num_nodes):
        for j in range(i - window_size, i + window_size + 1):
            if i != j and 0 <= j < num_nodes:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=y)

# ---------- 数据读取 ----------
def parse_txt_file(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    data_list = []
    buffer = []
    for line in lines:
        if line.startswith('>'):
            if len(buffer) == 3:
                name = buffer[0][1:]
                seq = buffer[1]
                label = buffer[2]
                if len(seq) == len(label):
                    data = sequence_to_graph(seq, label)
                    data.name = name
                    data.source_file = os.path.basename(path)
                    data_list.append(data)
            buffer = [line]
        else:
            buffer.append(line)
    if len(buffer) == 3:
        name = buffer[0][1:]
        seq = buffer[1]
        label = buffer[2]
        if len(seq) == len(label):
            data = sequence_to_graph(seq, label)
            data.name = name
            data.source_file = os.path.basename(path)
            data_list.append(data)
    return data_list

def build_global_vocab(folder_path, k=3):
    vocab = set()
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            path = os.path.join(folder_path, filename)
            with open(path, 'r') as f:
                lines = f.readlines()
            for i in range(len(lines)):
                if lines[i].startswith('>') and i + 1 < len(lines):
                    seq = lines[i + 1].strip()
                    for j in range(len(seq) - k + 1):
                        vocab.add(seq[j:j + k])
    return {kmer: idx for idx, kmer in enumerate(sorted(vocab))}

# ---------- 数据入口 ----------
def load_raw_dataset(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            file_data = parse_txt_file(file_path)
            all_data.extend(file_data)
    return all_data

def split_dataset_by_filename(all_data):
    train_data = [d for d in all_data if 'Train' in d.source_file]
    test_data = [d for d in all_data if 'Test' in d.source_file]
    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    return train_data, val_data, test_data

# ---------- 扩散增强 ----------
def augment_with_diffusion(diff_model, data_list, device):
    augmented_data = []
    for data in data_list:
        num_pos = int((data.y == 1).sum().item())
        if num_pos == 0:
            continue

        new_x = diff_model.generate_positive_sample(num_samples=num_pos)
        new_x = new_x.cpu()  # 先放 CPU 再转回 GPU 统一管理

        #  构建一个简单线性图：0-1-2-3-...-n
        edge_index = []
        for i in range(num_pos - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])  # 双向

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        #  构造 y 全 1，大小等于 num_pos
        new_y = torch.ones(num_pos, dtype=torch.long)

        new_data = Data(x=new_x, edge_index=edge_index, y=new_y).to(device)
        new_data.name = data.name + "_gen"
        new_data.source_file = data.source_file
        augmented_data.append(new_data)
    return augmented_data



class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # 计算样本的预测概率
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss  # 计算焦点损失
        return focal_loss.mean()  # 返回焦点损失的平均值

# ---------- GCN 模型 ----------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
#提升性能，定义 MLP 分类器模块
class GCN_with_MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, gcn_out_channels, mlp_hidden=64, out_classes=2):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, gcn_out_channels)

        self.mlp = nn.Sequential(
            nn.Linear(gcn_out_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_classes)
        )

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = self.mlp(x)
        return x  # logits

# ---------- 训练 ----------
def train_model(balanced_data, device):
    print("Training model with balanced dataset...")
    loader = DataLoader(balanced_data, batch_size=8, shuffle=True)
    in_channels = balanced_data[0].x.shape[1]
    model = GCN_with_MLP(
        in_channels=in_channels,
        hidden_channels=64,
        gcn_out_channels=64,
        mlp_hidden=64,
        out_classes=2
    ).to(device)

    # 设置正类的权重（例如，正类的权重设置为5倍）
    weight = torch.tensor([1.0, 5.0]).to(device)  # 正类加权5倍
    criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=weight)  # 使用 Focal Loss

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    for epoch in range(50):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in loader:
            batch = batch.to(device)  # Move the batch to the correct device
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            # Ensure both pred and batch.y are on the same device
            correct += (pred == batch.y.to(device)).sum().item()  # Move batch.y to device
            total += batch.num_nodes
            total_loss += loss.item()

        acc = correct / total
        print(f"Epoch {epoch + 1}/50 - Loss: {total_loss / len(loader):.4f} - Acc: {acc:.4f}")

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'Weights/best_model.pt')
            print(f"Saved best model at epoch {epoch + 1} (Acc: {acc:.4f})")

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

        # Collect predictions and labels for metrics calculation
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    # Accuracy calculation
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    # F1 Score
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    print(f"F1 Score: {f1:.4f}")

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(all_labels, all_preds)
    print(f"MCC: {mcc:.4f}")

    # You can return the metrics if needed
    return accuracy, f1, mcc

def auto_augment_until_balanced(diff_model, original_data, device, target_ratio=0.5, batch_size=1024):
    from torch_geometric.data import Data
    import math
    import time

    # 统计当前正负样本数
    total_pos, total_neg = 0, 0
    for data in original_data:
        total_pos += (data.y == 1).sum().item()
        total_neg += (data.y == 0).sum().item()

    current_ratio = total_pos / (total_pos + total_neg)
    print(f"当前正类比例: {current_ratio:.4f}")

    # 如果已经平衡，不再生成
    if current_ratio >= target_ratio:
        print(" 数据已平衡，无需扩增")
        return []

    # 计算所需生成的正类节点数
    total_target_pos = int(target_ratio * (total_neg / (1 - target_ratio)))
    num_to_generate = total_target_pos - total_pos
    print(f"正类样本不足，计划生成 {num_to_generate} 个正类节点")

    generated_data = []
    start_time = time.time()

    # 按 batch_size 分批生成
    num_batches = math.ceil(num_to_generate / batch_size)
    for i in range(num_batches):
        n = batch_size if (i < num_batches - 1) else num_to_generate - batch_size * (num_batches - 1)
        new_x = diff_model.generate_positive_sample(num_samples=n).cpu()

        # 构造简单图结构
        edge_index = []
        for j in range(n - 1):
            edge_index.append([j, j + 1])
            edge_index.append([j + 1, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        new_y = torch.ones(n, dtype=torch.long)
        new_data = Data(x=new_x, edge_index=edge_index, y=new_y).to(device)
        new_data.name = f"gen_{i}"
        new_data.source_file = "generated"
        generated_data.append(new_data)

        print(f" 批次 {i+1}/{num_batches} - 生成 {n} 个节点")

    elapsed = time.time() - start_time
    print(f" 总计生成 {num_to_generate} 个正类节点，用时 {elapsed:.2f} 秒")

    return generated_data


def check_dataset_balance(dataset, threshold=1.5):
    total_pos = 0
    total_neg = 0
    for data in dataset:
        y = data.y
        total_pos += (y == 1).sum().item()
        total_neg += (y == 0).sum().item()

    total = total_pos + total_neg
    pos_ratio = total_pos / total if total > 0 else 0
    neg_ratio = total_neg / total if total > 0 else 0

    print(f"正类样本: {total_pos}, 负类样本: {total_neg}")
    print(f"正类比例: {pos_ratio:.4f}, 负类比例: {neg_ratio:.4f}")

    if pos_ratio > (1 / (1 + threshold)) and pos_ratio < (threshold / (1 + threshold)):
        print(" 数据集已相对平衡！")
    else:
        print(" 数据集仍然不平衡！")




# ---------- 主函数 ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = './Raw_data'
    global GLOBAL_VOCAB
    GLOBAL_VOCAB = build_global_vocab(folder, k=3)
    data_all = load_raw_dataset(folder)
    train_data, val_data, test_data = split_dataset_by_filename(data_all)
    train_data = [d.to(device) for d in train_data]
    val_data = [d.to(device) for d in val_data]
    test_data = [d.to(device) for d in test_data]
    if len(train_data) == 0:
        print(" 无有效训练数据")
        return
    feature_dim = train_data[0].x.shape[1]

    # Create Diffusion Model and train on positive samples
    diff_model = DiffusionModel(input_dim=feature_dim, device=device)
    diff_model.train_on_positive_samples(train_data)

    # 动态生成直到平衡
    augmented_train_data = auto_augment_until_balanced(diff_model, train_data, device, target_ratio=0.5,
                                                       batch_size=64)
    balanced_train_data = balance_dataset(train_data, augmented_train_data)

    check_dataset_balance(balanced_train_data, threshold=2.0)

    # 建立边预测模型（预训练 or 联合训练）
    edge_predictor = EdgePredictor(feature_dim).to(device)

    # 举例：我们只使用一个图 + 新节点来构建 G*
    # 你可以挑一个大图（比如 train_data[0]）或合并多个图先试验
    target_graph = train_data[0].to(device)
    generated_x = diff_model.generate_positive_sample(num_samples=64).to(device)

    G_star = connect_generated_nodes(
        original_data=target_graph,
        generated_x=generated_x,
        edge_predictor=edge_predictor,
        device=device,
        threshold=0.8
    )

    print(f"增强图 G* 节点数: {G_star.num_nodes}, 边数: {G_star.num_edges}")

    # Train model
    model = train_model(balanced_train_data, device)

    # Reload the best model (safety measure)
    model.load_state_dict(torch.load('Weights/best_model.pt'))

    # Evaluate on test set
    accuracy, f1, mcc = test_model(model, test_data, device)

    print(f"Test Results: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}, MCC = {mcc:.4f}")


if __name__ == '__main__':
    main  ()