import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torch_geometric.data import Data


class SimpleDiffusionGenerator(nn.Module):
    def __init__(self, input_dim, noise_dim=32):
        super(SimpleDiffusionGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 64)
        self.fc2 = nn.Linear(64, input_dim)

    def forward(self, noise):
        x = F.relu(self.fc1(noise))
        return torch.sigmoid(self.fc2(x))


class DiffusionModel:
    def __init__(self, input_dim, device='cpu'):
        self.generator = SimpleDiffusionGenerator(input_dim).to(device)
        self.device = device

    def train_on_positive_samples(self, all_data, batch_size=256, epochs=10):
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        positive_vectors = []
        for data in all_data:
            pos_feats = data.x[data.y == 1]
            if pos_feats.size(0) > 0:
                positive_vectors.append(pos_feats.cpu())

        if not positive_vectors:
            print("⚠ 没有可用于训练的正类样本")
            return

        full_pos_data = torch.cat(positive_vectors, dim=0)
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
            torch.cuda.empty_cache()

    def generate_positive_sample(self, num_samples=10, batch_size=1024):
        self.generator.eval()
        generated_all = []
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                noise = torch.randn((current_batch_size, 32)).to(self.device)
                generated = self.generator(noise)
                generated_all.append(generated.cpu())
        return torch.cat(generated_all, dim=0)


def train_diffusion_model(train_data, feature_dim, device):
    diff_model = DiffusionModel(input_dim=feature_dim, device=device)
    diff_model.train_on_positive_samples(train_data)
    return diff_model


def generate_augmented_data(diff_model, original_data, device, target_ratio=0.5, batch_size=1024):
    total_pos, total_neg = 0, 0
    for data in original_data:
        total_pos += (data.y == 1).sum().item()
        total_neg += (data.y == 0).sum().item()

    current_ratio = total_pos / (total_pos + total_neg)
    print(f"当前正类比例: {current_ratio:.4f}")

    if current_ratio >= target_ratio:
        print(" 数据已平衡，无需扩增")
        return []

    total_target_pos = int(target_ratio * (total_neg / (1 - target_ratio)))
    num_to_generate = total_target_pos - total_pos
    print(f"正类样本不足，计划生成 {num_to_generate} 个正类节点")

    generated_data = []
    start_time = time.time()

    num_batches = math.ceil(num_to_generate / batch_size)
    for i in range(num_batches):
        n = batch_size if (i < num_batches - 1) else num_to_generate - batch_size * (num_batches - 1)
        new_x = diff_model.generate_positive_sample(num_samples=n).cpu()

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