import torch
import pandas as pd
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse

# ---------- 这段与你训练用的 featurizer 必须一致 ----------
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        atom.GetExplicitValence(),
        atom.GetImplicitValence(),
        atom.GetIsAromatic(),
    ], dtype=torch.float)

def bond_features(bond):
    return torch.tensor([
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.IsInRing(),
        int(bond.GetStereo()),
        bond.GetBeginAtomIdx(),
        bond.GetEndAtomIdx()
    ], dtype=torch.float)

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # atoms
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])

    # bonds
    edge_index = []
    edge_attr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        f = bond_features(b)

        # UNDIRECTED
        edge_index.append([i, j])
        edge_attr.append(f)
        edge_index.append([j, i])
        edge_attr.append(f)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.stack(edge_attr)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# -------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

df = pd.read_csv(args.csv)
smiles_list = df['smiles'].tolist()

graphs = []
labels = []  # 外部测试集没有 label → 全 None
orig_idx = []

for i, smi in enumerate(smiles_list):
    g = mol_to_graph(smi)
    graphs.append(g)
    labels.append(None)
    orig_idx.append(i)

final = {
    "source_csv": args.csv,
    "smiles": smiles_list,
    "labels": labels,
    "graphs": graphs,
    "orig_idx": orig_idx
}

torch.serialization.add_safe_globals([Data])
torch.save(final, args.out)

print(f"Saved external graphs → {args.out}")
