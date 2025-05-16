import os
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# ---------- 全局 K-mer 词表 ----------
GLOBAL_VOCAB = {}

# ---------- K-mer Encoding ----------
def extract_kmer_features(sequence, k=3):
    kmer_list = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer not in GLOBAL_VOCAB:
            GLOBAL_VOCAB[kmer] = len(GLOBAL_VOCAB)
        kmer_list.append(GLOBAL_VOCAB[kmer])

    one_hot = torch.zeros(len(kmer_list), len(GLOBAL_VOCAB))
    for i, idx in enumerate(kmer_list):
        one_hot[i, idx] = 1
    return one_hot

# ---------- Graph Construction ----------
def sequence_to_graph(sequence, label, k=3, window_size=5):
    x = extract_kmer_features(sequence, k)  # shape: [L', D]
    y = torch.tensor([int(i) for i in label[k - 1:]], dtype=torch.long)

    num_nodes = x.size(0)
    assert len(y) == num_nodes, f"label length mismatch after k-mer: {len(label)} vs {num_nodes}"

    edge_index = []
    for i in range(num_nodes):
        for j in range(i - window_size, i + window_size + 1):
            if i != j and 0 <= j < num_nodes:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=y)

# ---------- TXT 文件读取 ----------
def parse_txt_file(path):
    from data_loader_from_raw import sequence_to_graph  # 如果你在同文件中定义就不用加这行
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
                    data.source_file = os.path.basename(path)  # ✅ 添加这句
                    data_list.append(data)
                else:
                    print(f"⚠️ Skipping {name}: length mismatch ({len(seq)} vs {len(label)})")
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
            data.source_file = os.path.basename(path)  # ✅ 添加这句
            data_list.append(data)
        else:
            print(f"⚠️ Skipping {name}: length mismatch ({len(seq)} vs {len(label)})")

    return data_list


# ---------- 数据加载入口 ----------
def load_raw_dataset(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            file_data = parse_txt_file(file_path)
            all_data.extend(file_data)
    return all_data

# ---------- 数据集划分逻辑 ----------
def split_dataset_by_filename(all_data):
    print("Available data names:")
    for data in all_data:
        print(data.name)

    # 更改匹配条件，例如只匹配包含 "Train" 或 "Test" 的文件名
    train_data = [data for data in all_data if "Train" in data.name]
    test_data = [data for data in all_data if "Test" in data.name]

    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")

    if len(train_data) == 0 or len(test_data) == 0:
        print("❌ No matching files found. Please check the data in the 'Raw_data' folder.")
        return [], [], []

    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)

    return train_data, val_data, test_data




if __name__ == '__main__':
    folder = './Raw_data'
    data_all = load_raw_dataset(folder)
    train_data, val_data, test_data = split_dataset_by_filename(data_all)
    print(f"Loaded: {len(data_all)} samples")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
