# csv2graph.py
import os
import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

# ------------------ 复用训练脚本中的函数 ------------------
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def calc_atom_features(atom, explicit_H=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    ) + one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
           [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
           one_of_k_encoding_unk(atom.GetHybridization(), [
               Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
               Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    if not explicit_H:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return results

def calc_bond_features(bond, use_chirality=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(), bond.IsInRing()
    ]
    if use_chirality:
        bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return [int(f) for f in bond_feats]

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = [calc_atom_features(a) for a in mol.GetAtoms()]

    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        feats = calc_bond_features(bond)
        row += [a, b]
        col += [b, a]
        edge_attr += [feats, feats]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.tensor(x, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ------------------ CSV -> graph.pt ------------------
def csv_to_graph(csv_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 假设你的 CSV 有 'sequence' 列存 SMILES
    smiles_list = df['sequence'].astype(str).tolist()

    for idx, smi in enumerate(tqdm(smiles_list)):
        graph = mol_to_graph(smi)
        if graph is not None:
            torch.save(graph, os.path.join(save_dir, f"graph_{idx}.pt"))

    print(f"Saved {len(smiles_list)} graph files to {save_dir}")

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1]      # 外部 CSV
    save_dir = sys.argv[2]      # 输出 graph.pt 文件夹
    csv_to_graph(csv_path, save_dir)
