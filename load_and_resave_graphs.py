#!/usr/bin/env python3
# eval_graphs_with_ckpt_exact.py
"""
Usage example:
python eval_graphs_with_ckpt_exact.py \
  --graphs /home/shenxin/LiBP/dataset/graphs_from_smiles_resaved.pt \
  --ckpt  /home/shenxin/LiBP/best_model.pt \
  --out_dir ./eval_out \
  --model GraphTransformer
"""

import os
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score, recall_score, confusion_matrix

# --- adjust project import path if needed ---
import sys
sys.path.append('/home/shenxin/LiBP')
from plat_model.model import SubGT, GraphTransformer
from torch_geometric.data import DataLoader

# -------------------- helpers --------------------
def infer_model_params_from_state_dict(sd: dict):
    """Try to infer model params from a state_dict (shapes)."""
    info = {}
    # node_encoder.weight -> shape (hidden, in_channels)
    k_node = next((k for k in sd.keys() if 'node_encoder.weight' in k), None)
    k_edge = next((k for k in sd.keys() if 'edge_encoder.weight' in k), None)
    # count gt_block layers by seeing gt_block.N...
    n_blocks = len({k.split('.')[1] for k in sd.keys() if k.startswith('gt_block.')})
    if k_node is not None:
        w = sd[k_node]
        if isinstance(w, torch.Tensor):
            info['num_hidden_channels'] = int(w.shape[0])
            info['in_channels'] = int(w.shape[1])
    if k_edge is not None:
        w = sd[k_edge]
        if isinstance(w, torch.Tensor):
            # edge_encoder.weight shape might be (hidden, edge_feat)
            info['edge_features'] = int(w.shape[1])
    if n_blocks:
        info['num_gt_blocks'] = int(max(int(i) for i in n_blocks) + 1) if isinstance(n_blocks, set) else int(n_blocks)
        # safer: set from set size:
        try:
            info['num_gt_blocks'] = len({int(k.split('.')[1]) for k in sd.keys() if k.startswith('gt_block.')})
        except Exception:
            pass
    return info

def safe_load_graphs(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    # expect keys: smiles, labels, graphs, orig_idx (orig_idx optional)
    keys = list(d.keys())
    assert 'graphs' in d, f"'graphs' not found in {path}. keys: {keys}"
    smiles = d.get('smiles', None)
    labels = d.get('labels', None)
    graphs = d['graphs']
    orig_idx = d.get('orig_idx', list(range(len(graphs))))
    return smiles, labels, graphs, orig_idx, d

def batched_forward_collect(model, loader, device, threshold=0.5):
    model.eval()
    probs_list = []
    preds_list = []
    trues_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            # convert to probability of positive class
            if out.dim() == 2 and out.size(1) == 2:
                probs = torch.softmax(out, dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(out.view(-1))
            preds = (probs >= threshold).long().cpu().numpy()
            probs_list.append(probs.cpu().numpy())
            preds_list.append(preds)
            trues_list.append(batch.y.long().cpu().numpy())
    probs_all = np.concatenate(probs_list, axis=0)
    preds_all = np.concatenate(preds_list, axis=0)
    trues_all = np.concatenate(trues_list, axis=0)
    return probs_all, preds_all, trues_all

def compute_metrics(trues, probs, preds):
    # careful if only single class present for AUC
    try:
        auc = float(roc_auc_score(trues, probs)) if len(np.unique(trues))>1 else float('nan')
    except Exception:
        auc = float('nan')
    try:
        f1 = float(f1_score(trues, preds))
    except Exception:
        f1 = float('nan')
    try:
        mcc = float(matthews_corrcoef(trues, preds))
    except Exception:
        mcc = float('nan')
    acc = float((preds == trues).astype(float).mean())
    try:
        ba = float(balanced_accuracy_score(trues, preds))
    except Exception:
        ba = float('nan')
    try:
        se = float(recall_score(trues, preds, pos_label=1))
    except Exception:
        se = float('nan')
    try:
        cm = confusion_matrix(trues, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sp = float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan')
        else:
            tn = fp = fn = tp = None
            sp = float('nan')
    except Exception:
        sp = float('nan'); tn=fp=fn=tp=None
    metrics = {
        'AUC': auc, 'F1': f1, 'MCC': mcc, 'ACC': acc, 'BA': ba, 'SE': se, 'SP': sp,
        'tn': int(tn) if tn is not None else None, 'fp': int(fp) if fp is not None else None,
        'fn': int(fn) if fn is not None else None, 'tp': int(tp) if tp is not None else None
    }
    return metrics

# -------------------- main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graphs', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out_dir', default='./eval_out')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--model', choices=['GraphTransformer','SubGT'], default='GraphTransformer')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if (args.device is None and torch.cuda.is_available()) else (args.device or 'cpu'))
    print("Device:", device)

    # load graphs file
    smiles, labels, graphs, orig_idx, raw_graphs_obj = safe_load_graphs(args.graphs)
    print("Loaded graphs:", len(graphs), "smiles:", None if smiles is None else len(smiles))
    # build PyG DataLoader (preserve original order)
    dataset = graphs  # list of Data objects
    loader = DataLoader([g for g in dataset], batch_size=args.batch_size, shuffle=False)

    # load checkpoint (could be state_dict or dict)
    ckpt_obj = torch.load(args.ckpt, map_location='cpu')
    if isinstance(ckpt_obj, dict) and 'model_state_dict' in ckpt_obj:
        ckpt_sd = ckpt_obj['model_state_dict']
        ckpt_meta = {k:v for k,v in ckpt_obj.items() if k!='model_state_dict'}
    elif isinstance(ckpt_obj, dict) and any(k.endswith('.weight') for k in ckpt_obj.keys()):
        ckpt_sd = ckpt_obj
        ckpt_meta = {}
    else:
        # whole-model object or unknown - try to load as object later
        ckpt_sd = None
        ckpt_meta = {'loaded_object': True}

    print("Checkpoint loaded. meta keys:", list(ckpt_meta.keys()))

    # If we have a state_dict, infer model params
    model_params = {}
    if ckpt_sd is not None:
        inferred = infer_model_params_from_state_dict(ckpt_sd)
        print("Inferred params from state_dict:", inferred)
        # set defaults if not found
        in_ch = inferred.get('in_channels', 38)
        edge_feats = inferred.get('edge_features', 6)
        hidden = inferred.get('num_hidden_channels', 256)
        # num_layers (for SubGT or GraphTransformer internal)
        num_blocks = inferred.get('num_gt_blocks', None)
    else:
        # fallback defaults
        in_ch, edge_feats, hidden = 38, 6, 256
        num_blocks = None

    # instantiate model
    if args.model == 'GraphTransformer':
        model = GraphTransformer(in_channels=in_ch, edge_features=edge_feats, num_hidden_channels=hidden).to(device)
    else:
        # SubGT signature may vary — adjust if your SubGT requires num_layers
        kwargs = {'in_channels': in_ch, 'edge_features': edge_feats, 'num_hidden_channels': hidden}
        if num_blocks is not None:
            kwargs['num_layers'] = num_blocks
        model = SubGT(**kwargs).to(device)
    print("Model instantiated:", args.model)

    # load state_dict if available
    missing_keys = unexpected_keys = None
    if ckpt_sd is not None:
        missing_keys, unexpected_keys = model.load_state_dict(ckpt_sd, strict=False)
        print("Loaded state_dict into model (strict=False).")
        print("missing_keys:", missing_keys)
        print("unexpected_keys:", unexpected_keys)
        # print shapes of key tensors for a few keys (sanity)
        sample_keys = ['node_encoder.weight','edge_encoder.weight']
        for k in sample_keys:
            if k in ckpt_sd:
                t = ckpt_sd[k]
                if isinstance(t, torch.Tensor):
                    print(f"{k} shape in ckpt: {tuple(t.shape)}")
    else:
        # attempt to use whole-object
        print("Checkpoint did not contain state_dict; attempting to use ckpt_obj as model (not recommended).")
        try:
            model = ckpt_obj.to(device)
            print("Loaded model object directly.")
        except Exception as e:
            raise RuntimeError("Cannot load checkpoint as state_dict nor model object.") from e

    model.eval()

    # 1) forward once for shape check: feed first batch
    with torch.no_grad():
        for sample_batch in loader:
            b = sample_batch.to(device)
            out = model(b)
            print("Sample forward output shape:", tuple(out.shape))
            break

    # 2) collect logits/probs/preds/trues (no threshold yet)
    probs_all, _, trues_all = batched_forward_collect(model, loader, device, threshold=0.5)  # preds here unused
    # note: loader yields in-order of dataset list, which we created same order as graphs

    # 3) find best threshold by maximizing F1 on this set (sweep)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1.0
    best_t = 0.5
    best_pred = None
    for t_ in thresholds:
        p_ = (probs_all >= t_).astype(int)
        try:
            cur = f1_score(trues_all, p_)
        except Exception:
            cur = -1.0
        if cur > best_f1:
            best_f1 = cur
            best_t = float(t_)
            best_pred = p_.copy()
    print("Best threshold by F1 sweep:", best_t, "F1:", best_f1)

    # 4) compute all metrics for that threshold
    metrics = compute_metrics(trues_all, probs_all, best_pred)
    metrics['best_threshold'] = best_t
    # attach checkpoint meta if any (epoch, best_metrics, etc.)
    metrics['ckpt_meta'] = ckpt_meta

    # 5) save metrics.json
    out_metrics_path = os.path.join(args.out_dir, 'metrics.json')
    with open(out_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to", out_metrics_path)
    print(metrics)

    # 6) save per-sample csv aligned with graphs order
    out_df = pd.DataFrame({
        'smiles': smiles if smiles is not None else [None]*len(probs_all),
        'label': trues_all.tolist(),
        'prob_positive': probs_all.tolist(),
        'pred': best_pred.tolist()
    })
    out_csv = os.path.join(args.out_dir, 'per_sample_preds.csv')
    out_df.to_csv(out_csv, index=False)
    print("Saved per-sample CSV to", out_csv)

if __name__ == '__main__':
    main()
