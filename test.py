import torch
from diffusion_gnn_mask import evaluate_model, DiffusionGNN, get_diffusion_schedule
from data_loader_from_raw import load_raw_dataset, split_dataset_by_filename

import warnings

warnings.filterwarnings("ignore")


def main():
    print("🧪 Loading test set from ./Raw_data ...")

    all_data = load_raw_dataset('./Raw_data')
    print(f"Loaded {len(all_data)} data samples")  # 添加这行代码以打印数据样本数量

    train_data, val_data, test_data = split_dataset_by_filename(all_data)

    if len(test_data) == 0:
        print("❌ No test data found. Check if file names contain 'DNA-46_Test', 'DNA-129_Test', or 'DNA-181_Test'.")
        return

    # 接下来的代码...


    # Model config (must match training config)
    in_dim = test_data[0].x.shape[1]  # Feature dimension
    hidden_dim = 64
    T = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model weights
    model = DiffusionGNN(in_dim, hidden_dim, T).to(device)

    # Ensure the saved model exists at the specified path
    try:
        model.load_state_dict(torch.load('./saved_model.pt'))
        model.eval()
    except FileNotFoundError:
        print("❌ Model weights not found at './saved_model.pt'. Please check the path.")
        return

    # Diffusion schedule
    betas, alphas, alphas_bar = get_diffusion_schedule(T)

    print("\n📊 Evaluating on test set...")
    evaluate_model(model, test_data, alphas, alphas_bar, T, device)


if __name__ == '__main__':
    main()
