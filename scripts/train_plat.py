import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch_geometric.data import Dataset, Data, DataLoader
from sklearn.model_selection import train_test_split  # 导入划分训练集和测试集的工具
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix, balanced_accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append('/home/shenxin/LiBP')
from plat_model.model import SubGT, GraphTransformer  # 你自己的 SubGT 文件路径


NODE_FEATURE_NAMES = [
    "atom_type_onehot",
    "degree_onehot",
    "formal_charge",
    "num_radical_electrons",
    "hybridization_onehot",
    "aromatic",
    "num_hydrogens_onehot",
]

EDGE_FEATURE_NAMES = [
    "bond_type_single",
    "bond_type_double",
    "bond_type_triple",
    "bond_type_aromatic",
    "conjugation",
    "ring",
]


def build_model_config(model):
    hidden_dim = int(model.node_encoder.out_features)
    num_heads = int(getattr(model, "num_attention_heads", 1))
    return {
        "model_name": model.__class__.__name__,
        "num_layers": int(getattr(model, "num_layers", len(getattr(model, "gt_block", [])))),
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "head_dim": hidden_dim // num_heads if num_heads > 0 else hidden_dim,
        "dropout": float(getattr(model, "dropout_rate", 0.0)),
        "edge_dim": int(model.edge_encoder.in_features),
        "node_dim": int(model.node_encoder.in_features),
        "activation": model.activ_fn.__class__.__name__ if hasattr(model, "activ_fn") else "Unknown",
        "readout": "global_mean_pool + MLP",
        # Keep old keys for backward compatibility with existing eval scripts.
        "in_channels": int(model.node_encoder.in_features),
        "edge_features": int(model.edge_encoder.in_features),
        "num_hidden_channels": hidden_dim,
        "num_attention_heads": num_heads,
        "dropout_rate": float(getattr(model, "dropout_rate", 0.0)),
        "norm_to_apply": str(getattr(model, "norm_to_apply", "batch")),
    }


def build_feature_config(use_chirality=False, explicit_h=False):
    return {
        "node_features": NODE_FEATURE_NAMES,
        "edge_features": EDGE_FEATURE_NAMES,
        "featurizer": "RDKit",
        "smiles_processing": "canonical (RDKit default) + stereochemistry preserved",
        "use_chirality": bool(use_chirality),
        "explicit_H": bool(explicit_h),
    }


def build_training_config(optimizer, scheduler, batch_size, epochs, criterion):
    return {
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "loss_function": criterion.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "scheduler_mode": getattr(scheduler, "mode", None),
        "scheduler_factor": float(getattr(scheduler, "factor", 1.0)),
        "scheduler_patience": int(getattr(scheduler, "patience", 0)),
        "early_stopping": False,
    }


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
               Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
               Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(results)


def calc_bond_features(bond, use_chirality=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
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
    mol = Chem.RemoveHs(mol)
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
    def __init__(self, csv_path, cache_dir="/home/shenxin/LiBP/dataset"):
        super().__init__()
        self.cache_dir = cache_dir
        self.csv_path = csv_path

        cache_file = os.path.join(self.cache_dir, "bbbp_graphs3.pt")
        if os.path.exists(cache_file):
            print(f"Loading preprocessed data from {cache_file}")
            data = torch.load(cache_file,weights_only=False)
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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)  # SubGT forward

        # 交叉熵损失
        loss = criterion(out, batch.y.long())  # 对于交叉熵，标签需要是整数型 (long)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


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
            probs_batch = out[:, 1]. cpu().numpy()  # 取得正类别的概率（适用于二分类）

            preds.extend(pred)
            trues.extend(batch.y.cpu().numpy())
            probs.extend(probs_batch)  # 保存概率（用于 AUC 计算）

    # 转换为 PyTorch tensor
    preds = torch.tensor(preds)
    trues = torch.tensor(trues)

    # 计算 AUC (假设是二分类，probs 是正类的概率)
    auc = roc_auc_score(trues.numpy(), probs)

    # 计算 F1-Score (二分类)
    f1 = f1_score(trues. numpy(), preds.numpy())

    # 计算 MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(trues.numpy(), preds.numpy())

    # 计算准确率
    acc = (preds == trues). float().mean().item()

    # ========== 新增：计算 BA, SE, SP ==========
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(trues.numpy(), preds.numpy()).ravel()
    
    # BA - Balanced Accuracy (平衡准确率)
    ba = balanced_accuracy_score(trues. numpy(), preds.numpy())
    
    # SE - Sensitivity (灵敏度/召回率/真正例率)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # SP - Specificity (特异度/真负例率)
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # ==========================================

    return auc, f1, mcc, acc, ba, se, sp


def main():
    csv_path = "/home/shenxin/LiBP/dataset/train_9.csv"  # 修改为你的路径
    dataset = BBBP_Dataset(csv_path)
    split_random_state = 42
    split_strategy = "random"
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    batch_size = 32

    # 划分数据集，80% 训练集，10% 验证集，10% 测试集
    train_smiles, temp_smiles, train_labels, temp_labels = train_test_split(
        dataset.smiles, dataset.labels, test_size=0.2, random_state=split_random_state)

    # 从临时集进一步划分，50% 用于验证集，50% 用于测试集（相当于原数据集的10%）
    val_smiles, test_smiles, val_labels, test_labels = train_test_split(
        temp_smiles, temp_labels, test_size=0.5, random_state=split_random_state)

    # 创建训练集、验证集和测试集
    train_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in train_smiles]
    val_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in val_smiles]
    test_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in test_smiles]

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphTransformer(
        in_channels=38,
        edge_features=6,
        num_hidden_channels=256,
    ).to(device)

    # model = SubGT(
    #     in_channels=38,
    #     edge_features=6,
    #     num_hidden_channels=256,
    #     num_layers=6
    # ).to(device)

    optimizer = Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 使用 ReduceLROnPlateau 调度器，监控 val_loss 来调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.9, verbose=True)

    epochs=500  # 设定训练轮数
    best_acc = 0.0

    feature_config = build_feature_config(use_chirality=False, explicit_h=False)
    data_config = {
        "dataset": "BBBP",
        "train_csv": csv_path,
        "cache_file": os.path.join(dataset.cache_dir, "bbbp_graphs3.pt"),
        "split_strategy": split_strategy,
        "split_random_state": split_random_state,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "num_total": len(dataset),
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
        "external_test_set": None,
        "label_definition": "1: BBB+ / 0: BBB-",
    }

    for epoch in range(epochs):
        # 训练过程
        loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 在测试集上评估
        auc_test, f1_test, mcc_test, acc_test, ba_test, se_test, sp_test = evaluate(model, test_loader, device)

        # 输出训练过程中的各项指标
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {loss:.4f}, Test AUC: {auc_test:.4f}, Test F1: {f1_test:.4f}, Test MCC: {mcc_test:.4f}, "
            f"Test ACC: {acc_test:.4f}, Test BA: {ba_test:.4f}, Test SE: {se_test:.4f}, Test SP: {sp_test:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        print("-" * 80)

        # 在每个epoch结束后调用scheduler.step(val_loss)来调整学习率
        scheduler.step(acc_test)
        
        if acc_test > best_acc:
            best_acc = acc_test
            model_config = build_model_config(model)
            training_config = build_training_config(optimizer, scheduler, batch_size, epochs, criterion)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'model_config': model_config,
                'feature_config': feature_config,
                'training_config': training_config,
                'data_config': data_config,
                'preprocessing': {
                    'atom_feature_dim': 38,
                    'bond_feature_dim': 6,
                    'use_chirality': False,
                    'explicit_H': False,
                },
                'metrics': {
                    'train_loss': loss,
                    'test_acc': acc_test,
                    'test_auc': auc_test,
                    'test_f1': f1_test,
                    'test_mcc': mcc_test,
                    'test_ba': ba_test,
                    'test_se': se_test,
                    'test_sp': sp_test,
                },
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }
            
            os.makedirs('/home/shenxin/LiBP/ckpt', exist_ok=True)
            save_path = f'/home/shenxin/LiBP/ckpt/09best_model_epoch{epoch+1}_acc{acc_test:.4f}.pt'
            torch.save(checkpoint, save_path)
            print(f"✅ Saved best model to {save_path}")

if __name__ == "__main__":
    main()
