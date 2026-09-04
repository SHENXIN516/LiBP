import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch. optim import Adam
from torch_geometric.data import Dataset, Data, DataLoader
from sklearn.model_selection import train_test_split  # 导入划分训练集和测试集的工具
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix, balanced_accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append('/home/shenxin/LiBP')
from plat_model. model import SubGT, GraphTransformer  # 你自己的 SubGT 文件路径


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def calc_atom_features(atom, explicit_H=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    ) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
           [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
           one_of_k_encoding_unk(atom.GetHybridization(), [
               Chem.rdchem.HybridizationType.SP, Chem.rdchem. HybridizationType.SP2,
               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
               Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(results)


def calc_bond_features(bond, use_chirality=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType. SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(), bond.IsInRing()
    ]
    if use_chirality:
        bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # -------------------------
    # 节点特征
    # -------------------------
    x = np.array([calc_atom_features(a) for a in mol.GetAtoms()])

    # -------------------------
    # 边特征
    # -------------------------
    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        bond_feats = calc_bond_features(bond)

        row += [a, b]
        col += [b, a]

        edge_attr.append(bond_feats)
        edge_attr.append(bond_feats)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)


class BBBP_Dataset(Dataset):
    def __init__(self, csv_path, cache_dir="/home/shenxin/LiBP/dataset", cache_name="bbbp_graphs4.pt"):
        super().__init__()
        self.cache_dir = cache_dir
        self.csv_path = csv_path

        cache_file = os.path.join(self.cache_dir, cache_name)
        if os.path.exists(cache_file):
            print(f"Loading preprocessed data from {cache_file}")
            data = torch.load(cache_file, weights_only=False)
            self.smiles = data["smiles"]
            self.labels = data["labels"]
            self.graphs = data["graphs"]
        else:
            df = pd.read_csv(self.csv_path)
            df = df[df["type"] == "SMILES"]

            smiles = df["sequence"].astype(str).tolist()
            labels = df["label"].astype(int).tolist()

            self.graphs = []
            self.labels = []
            self.smiles = []
            for smi, label in zip(smiles, labels):
                g = mol_to_graph(smi)
                if g is not None:
                    self.graphs.append(g)
                    self.labels.append(label)
                    self.smiles.append(smi)

            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save({"smiles": self.smiles, "labels": self.labels, "graphs": self.graphs}, cache_file)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs[idx]
        graph.y = torch.tensor([self.labels[idx]], dtype=torch.float)
        return graph


def evaluate(model, loader, device):
    model.eval()
    preds, trues, probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            # 使用 softmax 转换为概率
            out = torch.softmax(out, dim=-1)

            # 预测类别：选择最大概率对应的类别
            pred = torch.argmax(out, dim=-1). cpu().numpy()  # 预测类别
            probs_batch = out[:, 1].cpu().numpy()  # 取得正类别的概率（适用于二分类）

            preds.extend(pred)
            trues. extend(batch.y.cpu().numpy())
            probs.extend(probs_batch)  # 保存概率（用于 AUC 计算）

    # 转换为 PyTorch tensor
    preds = torch.tensor(preds)
    trues = torch.tensor(trues)

    # 计算 AUC (假设是二分类，probs 是正类的概率)
    auc = roc_auc_score(trues. numpy(), probs)

    # 计算 F1-Score (二分类)
    f1 = f1_score(trues. numpy(), preds.numpy())

    # 计算 MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(trues.numpy(), preds. numpy())

    # 计算准确率
    acc = (preds == trues).float().mean(). item()

    # ========== 新增：计算 BA, SE, SP ==========
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(trues. numpy(), preds.numpy()).ravel()
    
    # BA - Balanced Accuracy (平衡准确率)
    ba = balanced_accuracy_score(trues. numpy(), preds.numpy())
    
    # SE - Sensitivity (灵敏度/召回率/真正例率)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # SP - Specificity (特异度/真负例率)
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # ==========================================

    return auc, f1, mcc, acc, ba, se, sp


def main():
    # ==================== 1. 加载外部测试集 ====================
    csv_path = "/home/shenxin/LiBP/dataset/benchmark_2_from_reference.csv"  # 外部测试集路径
    dataset = BBBP_Dataset(csv_path, cache_dir="/home/shenxin/LiBP/dataset", cache_name="bbbp_graphs3.pt")

    # ==================== 2.  不划分，直接全部用作测试集 ====================
    # 创建测试集（使用全部数据）
    test_dataset = [dataset. get(i) for i in range(len(dataset))]

    # 创建 DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== 3. 加载训练好的模型 ====================
    checkpoint_path = '/home/shenxin/LiBP/ckpt/02best_model_epoch154_acc0.8693.pt'
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 从配置重建模型
    config = checkpoint['model_config']
    model = GraphTransformer(
        in_channels=config['in_channels'],
        edge_features=config['edge_features'],
        num_hidden_channels=config['num_hidden_channels'],
    ).to(device)
    
    # 加载训练好的权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"📊 Training Test metrics:")
    print(f"   Test ACC: {checkpoint['metrics']['test_acc']:.4f}, "
          f"Test AUC: {checkpoint['metrics']['test_auc']:.4f}, "
          f"Test F1: {checkpoint['metrics']['test_f1']:.4f}, "
          f"Test MCC: {checkpoint['metrics']['test_mcc']:.4f}")
    
    # ==================== 4. 在外部测试集上评估 ====================
    print(f"\n{'='*80}")
    print(f"Evaluating on External Test Set")
    print(f"{'='*80}")
    print(f"Total samples: {len(dataset)}")
    print(f"Positive samples: {sum(dataset.labels)}")
    print(f"Negative samples: {len(dataset. labels) - sum(dataset.labels)}")
    print(f"{'='*80}\n")
    
    auc_test, f1_test, mcc_test, acc_test, ba_test, se_test, sp_test = evaluate(model, test_loader, device)

    # ==================== 5. 输出结果 ====================
    print(f"{'='*80}")
    print(f"🎯 External Test Results:")
    print(f"{'='*80}")
    print(f"Test AUC: {auc_test:.4f}, Test F1: {f1_test:.4f}, Test MCC: {mcc_test:.4f}, "
          f"Test ACC: {acc_test:.4f}, Test BA: {ba_test:.4f}, Test SE: {se_test:.4f}, Test SP: {sp_test:.4f}")
    print(f"{'='*80}")
    
    # ==================== 6. 保存结果到文件 ====================
    results_file = '/home/shenxin/LiBP/results/external_test_results02.txt'
    os.makedirs(os.path. dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"External Test Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset: {csv_path}\n")
        f.write(f"Cache: bbbp_graphs2.pt\n")
        f.write(f"Total Samples: {len(dataset)}\n")
        f.write(f"Positive: {sum(dataset.labels)}, Negative: {len(dataset. labels) - sum(dataset.labels)}\n")
        f. write(f"\nMetrics:\n")
        f. write(f"Test AUC: {auc_test:.4f}\n")
        f. write(f"Test F1:  {f1_test:.4f}\n")
        f. write(f"Test MCC: {mcc_test:.4f}\n")
        f.write(f"Test ACC: {acc_test:.4f}\n")
        f.write(f"Test BA:  {ba_test:.4f}\n")
        f.write(f"Test SE:  {se_test:.4f}\n")
        f. write(f"Test SP:  {sp_test:.4f}\n")
    
    print(f"\n✅ Results saved to {results_file}")


if __name__ == "__main__":
    main()