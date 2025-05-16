# 新建一个统一主控逻辑文件：main.py

from modules.data_loader import load_and_prepare_data
from modules.diffusion import train_diffusion_model, generate_augmented_data
from modules.edge_predictor import connect_generated_nodes, EdgePredictor
from modules.model import GAT_with_MLP, train_model, test_model

import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 数据加载与处理 ----------
    folder = './Raw_data'
    train_data, val_data, test_data, feature_dim = load_and_prepare_data(folder, device)

    # ---------- 训练扩散模型 ----------
    diff_model = train_diffusion_model(train_data, feature_dim, device)

    # ---------- 扩散生成并增强数据 ----------
    augmented_train_data = generate_augmented_data(diff_model, train_data, device)
    balanced_train_data = train_data + augmented_train_data

    # ---------- 构建 G* 示例（1个图） ----------
    edge_predictor = EdgePredictor(feature_dim).to(device)
    generated_x = diff_model.generate_positive_sample(num_samples=64).to(device)
    G_star = connect_generated_nodes(train_data[0], generated_x, edge_predictor, device)
    print(f"增强图 G* 节点数: {G_star.num_nodes}, 边数: {G_star.num_edges}")

    # ---------- 模型训练与评估 ----------
    model = train_model(balanced_train_data, feature_dim, device)
    model.load_state_dict(torch.load('./best_gat_model.pt'))
    accuracy, f1, mcc = test_model(model, test_data, device)
    print(f"Test Results: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}, MCC = {mcc:.4f}")


if __name__ == '__main__':
    main()
