# Protein Binding Site Prediction with Diffusion-Augmented Graph Neural Networks

This project aims to predict protein-DNA binding sites using a diffusion model and Graph Neural Networks (GNNs). The process starts by converting protein sequences into graphs, followed by the use of graph neural networks (GNN) to model these structures. A diffusion-based model is applied to augment the dataset, improving the model's ability to predict protein-DNA binding sites. The ultimate goal is to provide an efficient prediction tool for studying protein-DNA interactions.


## 🔍 Overview
### Key Components:
- **ESM-2 Representation**: Residue-level embeddings using [Meta's ESM-2](https://huggingface.co/facebook/esm2_t33_650M_UR50D).
- **Diffusion Generator**: Generates synthetic positive (binding) samples from noise.
- **Edge Predictor**: Builds structural edges between generated and real nodes to form enhanced graph G*.
- **GCN + MLP**: Classifies binding residues using graph-based neighborhood aggregation and final dense layers.
- **Focal Loss**: Handles severe class imbalance during training.
- ## 📁 Project Structure
.
├── main.py # Entrypoint script
├── Raw_data/ # Raw .txt format sequences and labels
├── modules/
│ ├── data_loader.py # Data loading and ESM embedding
│ ├── diffusion.py # Diffusion-based generator
│ ├── edge_predictor.py # Edge prediction and G* construction
│ └── model.py # GCN + MLP model + training + evaluation
├── best_model.pt # Trained model checkpoint
└── README.md

## ⚙️ Dependencies

- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)  
- ESM (for protein sequence representations)
- scikit-learn  
- matplotlib

Install requirements:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install transformers
pip install scikit-learn
pip install matplotlib

### 🚀 How to Run
1.Prepare raw data in Raw_data/ folder:
>protein_name
SEQUENCE...
LABELS...(0 or 1 per residue)
2.Run the pipeline:
python main.py
3.Output includes:
Training logs with loss & accuracy

Final test results (Accuracy, F1, MCC)

best_model.pt as the saved model
📊 Example Result
Train Accuracy: 96.4%
Test Accuracy : 87.5%
F1 Score      : 0.31
MCC           : 0.28

📌 Highlights
✅ Novel use of diffusion-inspired generation for residue-level positive augmentation

✅ Combines protein language model with graph structure and edge learning

✅ Effective on severely imbalanced datasets
✏️ Citation
f you use this code or idea in your work, please cite or acknowledge:

"Diffusion-Augmented Graph Learning for Protein Binding Site Prediction"
(Author: [hanqing zhang], 2025)
📬 Contact
For questions or collaborations, contact:


hanqing zhang
Email: 3165619783@qq.com

