#!/usr/bin/env python3
"""
Minimal evaluator that follows the training evaluation style exactly.
Usage (example):
python eval_with_original_train_style.py \
  --graphs /home/shenxin/LiBP/dataset/graphs_from_smiles.pt \
  --ckpt  /home/shenxin/LiBP/best_model.pt \
  --out_dir ./eval_out \
  --model GraphTransformer \
  --batch_size 32
"""

import os, json, argparse
import torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score, recall_score, confusion_matrix
import sys
sys.path.append('/home/shenxin/LiBP')
from plat_model.model import SubGT, GraphTransformer
from torch_geometric.data import DataLoader

def safe_load_graphs(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    if 'graphs' not in d:
        raise RuntimeError("graphs key not in file")
    smiles = d.get('smiles', None)
    labels = d.get('labels', None)
    graphs = d['graphs']
    orig_idx = d.get('orig_idx', list(range(len(graphs))))
    # make sure every Data has .y set and aligned with labels/orig_idx
    if labels is not None:
        if len(labels) != len(graphs):
            raise RuntimeError(f"labels length {len(labels)} != graphs length {len(graphs)}")
        for i, g in enumerate(graphs):
            g.y = torch.tensor([int(labels[i])], dtype=torch.float)
    else:
        for i, g in enumerate(graphs):
            # mark unknown
            g.y = torch.tensor([-1], dtype=torch.float)
    return smiles, labels, graphs, orig_idx, d

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    preds_list, trues_list, probs_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            if out.dim() == 2 and out.size(1) == 2:
                probs = torch.softmax(out, dim=-1)[:,1]
            else:
                probs = torch.sigmoid(out.view(-1))
            preds = (probs >= threshold).long().cpu().numpy()
            trues = batch.y.long().cpu().numpy()
            probs = probs.cpu().numpy()
            preds_list.append(preds)
            trues_list.append(trues)
            probs_list.append(probs)
    if len(preds_list)==0:
        raise RuntimeError("Empty loader")
    preds_all = np.concatenate(preds_list, axis=0)
    trues_all = np.concatenate(trues_list, axis=0)
    probs_all = np.concatenate(probs_list, axis=0)
    # metrics
    try:
        auc = float(roc_auc_score(trues_all, probs_all))
    except Exception:
        auc = float('nan')
    try:
        f1 = float(f1_score(trues_all, preds_all))
    except Exception:
        f1 = float('nan')
    try:
        mcc = float(matthews_corrcoef(trues_all, preds_all))
    except Exception:
        mcc = float('nan')
    acc = float((preds_all == trues_all).astype(float).mean())
    try:
        ba = float(balanced_accuracy_score(trues_all, preds_all))
    except Exception:
        ba = float('nan')
    try:
        se = float(recall_score(trues_all, preds_all, pos_label=1))
    except Exception:
        se = float('nan')
    try:
        cm = confusion_matrix(trues_all, preds_all)
        if cm.size==4:
            tn, fp, fn, tp = cm.ravel()
            sp = float(tn/(tn+fp)) if (tn+fp)>0 else float('nan')
        else:
            tn=fp=fn=tp=None; sp=float('nan')
    except Exception:
        sp=float('nan'); tn=fp=fn=tp=None
    metrics = {'AUC':auc,'F1':f1,'MCC':mcc,'ACC':acc,'BA':ba,'SE':se,'SP':sp,
               'tn': (int(tn) if tn is not None else None),
               'fp': (int(fp) if fp is not None else None),
               'fn': (int(fn) if fn is not None else None),
               'tp': (int(tp) if tp is not None else None)}
    return metrics, probs_all, preds_all, trues_all

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graphs', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out_dir', default='./eval_out')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--model', choices=['GraphTransformer','SubGT'], default='GraphTransformer')
    p.add_argument('--device', default=None)
    p.add_argument('--threshold', type=float, default=None, help="If set, use this threshold; otherwise use 0.5")
    args = p.parse_args()

    device = torch.device('cuda' if (args.device is None and torch.cuda.is_available()) else (args.device or 'cpu'))
    os.makedirs(args.out_dir, exist_ok=True)
    print("Device:", device)

    smiles, labels, graphs, orig_idx, raw = safe_load_graphs(args.graphs)
    print("Loaded graphs:", len(graphs))
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    # build model
    # try to infer in/out dims from checkpoint if needed (simple fallback values)
    ckpt_obj = torch.load(args.ckpt, map_location='cpu')
    if isinstance(ckpt_obj, dict) and 'model_state_dict' in ckpt_obj:
        sd = ckpt_obj['model_state_dict']
    elif isinstance(ckpt_obj, dict) and any(k.endswith('.weight') for k in ckpt_obj.keys()):
        sd = ckpt_obj
    else:
        sd = None
    # defaults
    in_ch, edge_feats, hidden = 38, 6, 256
    if sd is not None:
        if 'node_encoder.weight' in sd:
            in_ch = int(sd['node_encoder.weight'].shape[1])
            hidden = int(sd['node_encoder.weight'].shape[0])
        if 'edge_encoder.weight' in sd:
            edge_feats = int(sd['edge_encoder.weight'].shape[1])

    if args.model == 'GraphTransformer':
        model = GraphTransformer(in_channels=in_ch, edge_features=edge_feats, num_hidden_channels=hidden).to(device)
    else:
        model = SubGT(in_channels=in_ch, edge_features=edge_feats, num_hidden_channels=hidden).to(device)

    # load weights
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("Loaded state_dict; missing_keys:", missing, "unexpected_keys:", unexpected)
    else:
        # try to load whole object
        try:
            model = ckpt_obj.to(device)
            print("Loaded whole model object from ckpt")
        except Exception:
            raise RuntimeError("Unknown ckpt format and no state_dict found.")

    model.eval()

    # if ckpt contains saved threshold or meta, use it
    saved_threshold = None
    if isinstance(ckpt_obj, dict):
        for cand in ['best_threshold','threshold','best_t']:
            if cand in ckpt_obj:
                saved_threshold = float(ckpt_obj[cand])
                break

    if args.threshold is not None:
        threshold = float(args.threshold)
    elif saved_threshold is not None:
        threshold = saved_threshold
        print("Using saved threshold from ckpt:", threshold)
    else:
        threshold = 0.5
        print("No saved threshold found; using default 0.5")

    metrics, probs_all, preds_all, trues_all = evaluate(model, loader, device, threshold=threshold)

    # save metrics
    metrics['threshold_used'] = threshold
    metrics_path = os.path.join(args.out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics:", metrics_path)
    print(metrics)

    # create per-sample CSV aligned with graphs order (and orig_idx if available)
    out_df = pd.DataFrame({
        'smiles': (smiles if smiles is not None else [None]*len(probs_all)),
        'label': list(trues_all.tolist()),
        'prob_positive': list(probs_all.tolist()),
        'pred': list(preds_all.tolist())
    })
    out_csv = os.path.join(args.out_dir, 'per_sample_preds.csv')
    out_df.to_csv(out_csv, index=False)
    print("Saved per-sample csv:", out_csv)

if __name__ == '__main__':
    main()
