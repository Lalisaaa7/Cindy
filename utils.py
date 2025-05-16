import torch
from torch_geometric.data import Data
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
import random
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# 处理训练集
def process_train_dataset(train_data):
    new_train_data = []
    for i in range(len(train_data)):
        features = train_data[i]['x']
        old_labels = train_data[i]['y'].squeeze()

        original_adj = train_data[i]['edge_index'].long()
        adj_dim = features.shape[0]
        # 创建一个 adj_dimxadj_dim 的零张量作为邻接矩阵
        adjacency_matrix = torch.zeros(adj_dim, adj_dim)

        for i in range(original_adj.size(1)):
            node1 = original_adj[0, i]
            node2 = original_adj[1, i]
            adjacency_matrix[node1, node2] = 1
            adjacency_matrix[node2, node1] = 1
        adj = adjacency_matrix

        labels_indices = torch.arange(old_labels.numel()).reshape(old_labels.shape)
        # 打乱索引
        idx_train = labels_indices.flatten()[torch.randperm(labels_indices.numel())].reshape(labels_indices.shape)
        adj, features, new_labels, idx_train = src_smote(adj, features, old_labels, idx_train, portion=0, im_class_num=1)

        # 创建 Data 对象
        new_graph = Data(x=features, edge_index=adj.to_sparse().coalesce().indices().long(), y=new_labels)
        new_train_data.append(new_graph)

    return new_train_data


def process_val_dataset(val_data):
    new_val_data = []
    for i in range(len(val_data)):
        features = val_data[i]['x']
        labels = val_data[i]['y'].squeeze()
        original_adj = val_data[i]['edge_index'].long()

        adj_dim = features.shape[0]
        adjacency_matrix = torch.zeros(adj_dim, adj_dim)
        for i in range(original_adj.size(1)):
            node1 = original_adj[0, i]
            node2 = original_adj[1, i]
            adjacency_matrix[node1, node2] = 1
            adjacency_matrix[node2, node1] = 1
        adj = adjacency_matrix

        new_graph = Data(x=features, edge_index=adj.to_sparse().coalesce().indices().long(), y=labels)
        new_val_data.append(new_graph)
    return new_val_data


def process_test_dataset(test_data):
    new_test_data = []
    for i in range(len(test_data)):
        features = test_data[i]['x']
        labels = test_data[i]['y'].squeeze()
        original_adj = test_data[i]['edge_index'].long()

        adj_dim = features.shape[0]
        adjacency_matrix = torch.zeros(adj_dim, adj_dim)
        for i in range(original_adj.size(1)):
            node1 = original_adj[0, i]
            node2 = original_adj[1, i]
            adjacency_matrix[node1, node2] = 1
            adjacency_matrix[node2, node1] = 1
        adj = adjacency_matrix

        new_graph = Data(x=features, edge_index=adj.to_sparse().coalesce().indices().long(), y=labels)
        new_test_data.append(new_graph)
    return new_test_data


def src_smote(adj, features, labels, idx_train, portion=0, im_class_num=1):
    c_largest = labels.max().item()
    adj_back = adj
    chosen = None
    new_features = None

    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        if portion == 0:
            c_portion = int(avg_number / new_chosen.shape[0])
            portion_rest = (avg_number / new_chosen.shape[0]) - c_portion
        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion

        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]
            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)
            idx_neighbor = distance.argmin(axis=-1)
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

        if new_chosen.shape[0] * portion_rest < 1:
            num = math.ceil(new_chosen.shape[0] * portion_rest) + 1
        else:
            num = int(new_chosen.shape[0] * portion_rest)

        new_chosen = new_chosen[:num]
        chosen_embed = features[new_chosen, :]
        distance = squareform(pdist(chosen_embed.cpu().detach()))
        np.fill_diagonal(distance, distance.max() + 100)
        idx_neighbor = distance.argmin(axis=-1)
        interp_place = random.random()
        embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

        if chosen is None:
            chosen = new_chosen
            new_features = embed
        else:
            chosen = torch.cat((chosen, new_chosen), 0)
            new_features = torch.cat((new_features, embed), 0)

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:, :]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen, :]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:, chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen, :][:, chosen]

    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0] + add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train


# 传入损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(loss)
        else:
            return loss
