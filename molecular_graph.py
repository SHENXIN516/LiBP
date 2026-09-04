import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

df = pd.read_csv("bbb_mispredicted.csv")

df_fp = df[df["Pred_BBB_Minus"] == "BBB-"]

mols = [Chem.MolFromSmiles(s) for s in df_fp["SMILES"] if Chem.MolFromSmiles(s)]
legends = ["False Negative"] * len(mols)

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=4,
    subImgSize=(500, 500)  # 以前 250×250 太小了
)

img.save("bbb_mispredicted_fn.png", dpi=(300, 300))
