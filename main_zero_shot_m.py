"""ScaleWeave zero-shot M4 ↔ M3 entry point.

Train on a source M-series bucket; without finetuning, evaluate on the
target bucket. Reports SMAPE / MAPE / MASE — OWA is intentionally omitted
(community convention for cross-dataset M3↔M4: GPT4TS / SE-LLM both report
SMAPE only). Forecast tensors are also dumped as .npy so OWA can be added
later if needed.
"""
import argparse
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_provider.data_loader_zero_shot import Dataset_MSeries
from models.ScaleWeave import ScaleWeave
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')
torch.set_num_threads(4)


# ---------- losses (numpy) ----------

def _smape(pred, true):
    return float(np.mean(200.0 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8)))


def _mape(pred, true):
    return float(np.mean(100.0 * np.abs(pred - true) / (np.abs(true) + 1e-8)))


def _mase(pred, true, insample, freq):
    """Naive seasonal scale per-series, then averaged."""
    pred = np.asarray(pred); true = np.asarray(true); insample = np.asarray(insample)
    N = pred.shape[0]
    out = np.empty(N)
    for i in range(N):
        x = insample[i]
        denom = np.mean(np.abs(x[freq:] - x[:-freq])) if x.shape[0] > freq else 1.0
        denom = denom if denom > 1e-8 else 1.0
        out[i] = np.mean(np.abs(pred[i] - true[i])) / denom
    return float(np.mean(out))


# ---------- args ----------

def build_parser():
    p = argparse.ArgumentParser('ScaleWeave zero-shot M-series')
    # source / target dataset selection
    p.add_argument('--source', choices=['m3', 'm4'], required=True)
    p.add_argument('--target', choices=['m3', 'm4'], required=True)
    p.add_argument('--source_root', type=str, required=True)
    p.add_argument('--target_root', type=str, required=True)
    p.add_argument('--source_path', type=str, default='')
    p.add_argument('--target_path', type=str, default='')
    p.add_argument('--source_pattern', type=str, default='')   # M4 bucket name
    p.add_argument('--target_pattern', type=str, default='')
    # window
    p.add_argument('--seq_len', type=int, required=True)
    p.add_argument('--pred_len', type=int, required=True)
    p.add_argument('--label_len', type=int, default=0)
    p.add_argument('--test_seq_len', type=int, default=0)
    p.add_argument('--test_pred_len', type=int, default=0)
    p.add_argument('--train_all', type=int, default=0,
                   help='If 1, dropping no holdout — train+val both span the full series '
                        'minus the last pred_len. Mirrors GPT4TS train_all flag for short-source cells.')
    p.add_argument('--token_len', type=int, default=0,
                   help='Per-step prediction length used at training. If 0 → token_len=pred_len. '
                        'Set < test_pred_len to enable autoregressive rolling forecast at evaluation '
                        '(matches SE-LLM exp_zero_shot_forecasting._rolling_forecast).')
    p.add_argument('--m4_freq', type=int, default=12,
                   help='naive-seasonal step for MASE on the *target* bucket '
                        '(Yearly=1, Quarterly=4, Monthly=12, Weekly=1, Daily=1, Hourly=24).')
    # training
    p.add_argument('--loss', choices=['SMAPE', 'MSE', 'MAPE'], default='SMAPE')
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--test_batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--train_epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lradj', choices=['COS', 'type1'], default='COS')
    p.add_argument('--tmax', type=int, default=10)
    p.add_argument('--eta_min', type=float, default=1e-8)
    p.add_argument('--seed', type=int, default=2021)
    # ScaleWeave model knobs (mirror main_m4.py defaults)
    p.add_argument('--model', type=str, default='ScaleWeave')
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=16)
    p.add_argument('--e_layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--in_dropout', type=float, default=0.1)
    p.add_argument('--out_dropout', type=float, default=0.1)
    p.add_argument('--enc_in', type=int, default=1)
    p.add_argument('--c_out', type=int, default=1)
    p.add_argument('--multi', type=int, default=0)
    p.add_argument('--revin_flag', type=int, default=1)
    p.add_argument('--transformer_ff_inner', type=int, default=256)
    # ScaleWeave __init__ also references these
    p.add_argument('--embed', type=str, default='timeF')
    p.add_argument('--scale_patch_sizes', type=str, default='2 4 6')
    p.add_argument('--scale_strides', type=str, default='2 4 6')
    p.add_argument('--hsg_layers', type=int, default=2)
    p.add_argument('--cross_scale_g', type=int, default=1)
    p.add_argument('--g_gate', type=int, default=1)
    p.add_argument('--learned_hyperedge_weights', type=int, default=0)
    # ScaleWeave __init__ requires these even though zero-shot doesn't use them
    p.add_argument('--sch_inject_at', type=str, default='0')
    p.add_argument('--gate_init_prg', type=float, default=0.5)
    # output
    p.add_argument('--out_dir', type=str, default='./zero_shot_results/')
    p.add_argument('--tag', type=str, default='run')
    return p


def setup(args):
    if args.label_len == 0:
        args.label_len = args.pred_len
    if args.test_seq_len == 0:
        args.test_seq_len = args.seq_len
    if args.test_pred_len == 0:
        args.test_pred_len = args.pred_len
    if args.token_len <= 0:
        args.token_len = args.pred_len
    assert args.token_len <= args.seq_len, \
        f'token_len ({args.token_len}) must be <= seq_len ({args.seq_len}) for rolling slide'
    args.sch_inject_at = [int(x) for x in args.sch_inject_at.split('_')] if args.sch_inject_at else []
    args.freq = 'h'  # ScaleWeave requires this attribute
    return args


# ---------- helpers ----------

def make_loader(args, which, flag):
    src_or_tgt = args.source if which == 'source' else args.target
    root = args.source_root if which == 'source' else args.target_root
    path = args.source_path if which == 'source' else args.target_path
    pat = args.source_pattern if which == 'source' else args.target_pattern
    seq, pred = (args.seq_len, args.pred_len) if which == 'source' \
        else (args.test_seq_len, args.test_pred_len)
    bs = args.batch_size if flag == 'train' else args.test_batch_size

    ds = Dataset_MSeries(
        root_path=root, flag=flag,
        size=[seq, args.label_len, pred],
        data_path=path, seasonal_patterns=pat, source=src_or_tgt,
        train_all=bool(args.train_all),
    )
    loader = DataLoader(ds, batch_size=bs,
                        shuffle=(flag == 'train'),
                        num_workers=args.num_workers,
                        drop_last=(flag == 'train'))
    print(f'[{which}/{flag}] N={len(ds)} (series={len(ds.timeseries)})')
    return ds, loader


def get_loss(name):
    if name == 'MSE':
        return nn.MSELoss()
    if name == 'MAPE':
        return lambda p, y: torch.mean(100. * torch.abs(p - y) / (torch.abs(y) + 1e-8))
    return lambda p, y: torch.mean(200. * torch.abs(p - y) / (torch.abs(p) + torch.abs(y) + 1e-8))


# ---------- main ----------

def main():
    args = setup(build_parser().parse_args())
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print('Args:', vars(args))
    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    # source train/val
    _, tr_loader = make_loader(args, 'source', 'train')
    _, va_loader = make_loader(args, 'source', 'val')
    # target test (single batched window per series via last_insample_window)
    te_ds, _ = make_loader(args, 'target', 'test')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ScaleWeave(args, device=device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] ScaleWeave params={n_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.lradj == 'COS':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=args.eta_min)
    else:
        sched = None
    loss_fn = get_loss(args.loss)
    stopper = EarlyStopping(patience=args.patience, verbose=True)

    ckpt_path = os.path.join(out_dir, 'checkpoint.pth')
    t0 = time.time()
    for ep in range(args.train_epochs):
        model.train()
        losses = []
        for bx, by, _, _ in tqdm(tr_loader, desc=f'ep{ep+1}', mininterval=10):
            bx = bx.to(device); by = by.to(device)
            optimizer.zero_grad()
            out = model(bx, 0)[:, -args.pred_len:, :]
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # quick val on source
        model.eval()
        v_losses = []
        with torch.no_grad():
            for bx, by, _, _ in va_loader:
                bx = bx.to(device); by = by.to(device)
                out = model(bx, 0)[:, -args.pred_len:, :]
                v_losses.append(loss_fn(out, by).item())
        v_loss = float(np.mean(v_losses)) if v_losses else float('inf')
        print(f'epoch {ep+1}: train={np.mean(losses):.4f}  source_val={v_loss:.4f}  ({time.time()-t0:.0f}s)')
        stopper(v_loss, model, out_dir)
        if stopper.early_stop:
            print('early stop'); break
        if sched is not None:
            sched.step()

    # load best
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'[ckpt] loaded best from {ckpt_path}')

    # ---------- target zero-shot evaluation ----------
    insample, _ = te_ds.last_insample_window()       # (N, test_seq_len)
    target_y = np.stack([ts[-args.test_pred_len:] for ts in te_ds.timeseries])  # (N, test_pred_len)

    x = torch.from_numpy(insample).float().unsqueeze(-1).to(device)  # (N, sl, 1)
    model.eval()

    def rolling_forecast(model, ctx_x, token_len, target_len):
        """Autoregressive rolling: predict token_len chunks until total >= target_len.
        Slides a fixed-size context window. Mirrors SE-LLM _rolling_forecast.
        """
        chunks = []
        ctx = ctx_x
        total = 0
        while total < target_len:
            if chunks:
                ctx = torch.cat([ctx[:, token_len:, :], chunks[-1]], dim=1)
            out = model(ctx, 0)            # (B, pred_len(=token_len at training), 1)
            chunks.append(out[:, -token_len:, :])
            total += token_len
        return torch.cat(chunks, dim=1)[:, :target_len, :]

    rolling = args.token_len < args.test_pred_len
    print(f'[eval] token_len={args.token_len} test_pred_len={args.test_pred_len} '
          f'rolling={rolling} (steps={int(np.ceil(args.test_pred_len / args.token_len))})')

    preds = np.zeros((x.shape[0], args.test_pred_len), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, x.shape[0], args.test_batch_size):
            chunk = x[s:s + args.test_batch_size]
            if rolling:
                out = rolling_forecast(model, chunk, args.token_len, args.test_pred_len)[..., 0].cpu().numpy()
            else:
                out = model(chunk, 0)[:, -args.test_pred_len:, 0].cpu().numpy()
            preds[s:s + chunk.shape[0]] = out

    smape = _smape(preds, target_y)
    mape = _mape(preds, target_y)
    mase = _mase(preds, target_y, insample, args.m4_freq)
    print(f'[zero-shot] {args.source}->{args.target} target_pattern={args.target_pattern or args.target_path}'
          f'  SMAPE={smape:.4f}  MAPE={mape:.4f}  MASE={mase:.4f}')

    # persist for later (OWA / aggregation)
    np.save(os.path.join(out_dir, 'preds.npy'), preds)
    np.save(os.path.join(out_dir, 'trues.npy'), target_y)
    np.save(os.path.join(out_dir, 'insample.npy'), insample)
    pd.DataFrame([dict(
        source=args.source, target=args.target,
        source_pattern=args.source_pattern, source_path=args.source_path,
        target_pattern=args.target_pattern, target_path=args.target_path,
        seq_len=args.seq_len, pred_len=args.pred_len,
        test_seq_len=args.test_seq_len, test_pred_len=args.test_pred_len,
        smape=smape, mape=mape, mase=mase, m4_freq=args.m4_freq,
        n_series=int(target_y.shape[0]), tag=args.tag,
    )]).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)


if __name__ == '__main__':
    main()
