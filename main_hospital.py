"""Hospital short-term forecasting entry point for ScaleWeave."""
import sys
sys.path.append('.')

import argparse
import datetime
import os
import random
import time
import uuid
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from data_provider.data_factory import data_provider
from data_provider.hospital import HospitalMeta
from models.ScaleWeave import ScaleWeave
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.hospital_summary import HospitalSummary
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')
torch.set_num_threads(4)


def build_parser():
    p = argparse.ArgumentParser(description='ScaleWeave Hospital short-term forecasting')
    p.add_argument('--root_path', type=str, default='./datasets/hospital/')
    p.add_argument('--data', type=str, default='hospital')
    p.add_argument('--seasonal_patterns', type=str, required=True,
                   choices=HospitalMeta.seasonal_patterns)
    p.add_argument('--features', type=str, default='M')
    p.add_argument('--freq', type=int, default=0)
    p.add_argument('--target', type=str, default='OT')
    p.add_argument('--embed', type=str, default='timeF')
    p.add_argument('--percent', type=int, default=100)
    p.add_argument('--max_len', type=int, default=-1)
    p.add_argument('--seq_len', type=int, default=0)
    p.add_argument('--label_len', type=int, default=0)
    p.add_argument('--pred_len', type=int, default=0)
    p.add_argument('--loss', type=str, default='SMAPE',
                   choices=['SMAPE', 'MAPE', 'MASE', 'MSE'])
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--train_epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lradj', type=str, default='COS')
    p.add_argument('--tmax', type=int, default=20)
    p.add_argument('--eta_min', type=float, default=1e-8)
    p.add_argument('--pct_start', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', type=str, default='ScaleWeave')
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--d_ff', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--e_layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--in_dropout', type=float, default=0.1)
    p.add_argument('--out_dropout', type=float, default=0.1)
    p.add_argument('--enc_in', type=int, default=1)
    p.add_argument('--c_out', type=int, default=1)
    p.add_argument('--multi', type=int, default=0)
    p.add_argument('--revin_flag', type=int, default=1)
    p.add_argument('--transformer_ff_inner', type=int, default=128)
    p.add_argument('--scale_patch_sizes', type=str, default='2 4 6')
    p.add_argument('--scale_strides', type=str, default='2 4 6')
    p.add_argument('--sch_layers', type=int, default=2)
    p.add_argument('--cross_scale_g', type=int, default=1)
    p.add_argument('--g_gate', type=int, default=1)
    p.add_argument('--learned_hyperedge_weights', type=int, default=0)
    p.add_argument('--sch_inject_at', type=str, default='0')
    p.add_argument('--gate_init_prg', type=float, default=0.5)
    p.add_argument('--fname', type=str, default='./checkpoints/hospital')
    p.add_argument('--wd_project', type=str, default='hospital')
    p.add_argument('--run_name', type=str, default='scaleweave_hospital')
    p.add_argument('--itr', type=int, default=1)
    return p


def setup_args(args):
    if args.pred_len == 0:
        args.pred_len = HospitalMeta.horizons_map[args.seasonal_patterns]
    if args.seq_len == 0:
        args.seq_len = 2 * args.pred_len
    if args.label_len == 0:
        args.label_len = args.pred_len
    args.frequency_map = HospitalMeta.frequency_map[args.seasonal_patterns]
    args.data_name = 'hospital_' + args.seasonal_patterns
    args.train_shuffle = True
    args.train_shuffle_int = 1
    args.sch_inject_at = [int(x) for x in args.sch_inject_at.split('_')] if args.sch_inject_at else []
    if args.freq == 0:
        args.freq = 'h'
    if not args.multi:
        args.enc_in = 1
        args.c_out = 1
    return args


def build_model(args, device):
    model = ScaleWeave(args, device=device).to(device)
    total = sum(p.numel() for p in model.parameters())
    sch = sum(p.numel() for n, p in model.named_parameters() if 'sch' in n)
    print(f"[param_count] total={total:,} | sch={sch:,}")
    return model


def get_criterion(name):
    return {
        'MSE': nn.MSELoss(),
        'MAPE': mape_loss(),
        'MASE': mase_loss(),
        'SMAPE': smape_loss(),
    }[name]


def vali(model, train_loader, vali_loader, criterion, args, device):
    """Validation: full-dataset forecast from last insample window, then loss."""
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B = x.shape[0]
        outputs = torch.zeros((B, args.pred_len, 1), dtype=torch.float32)
        id_list = list(np.arange(0, B, args.batch_size)) + [B]
        for i in range(len(id_list) - 1):
            chunk = x[id_list[i]:id_list[i + 1]]
            outputs[id_list[i]:id_list[i + 1]] = model(chunk, 0).detach().cpu()[:, -args.pred_len:, :]
        true = torch.from_numpy(np.array(y, dtype=np.float32))
        mask = torch.ones_like(true)
        loss = criterion(x.detach().cpu()[:, :, 0], args.frequency_map,
                         outputs[:, :, 0], true, mask)
    model.train()
    return float(loss.item())


def run_test(model, train_loader, test_loader, args, device, out_dir):
    x, _ = train_loader.dataset.last_insample_window()
    y = test_loader.dataset.timeseries
    ids = test_loader.dataset.ids
    x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B = x.shape[0]
        outputs = torch.zeros((B, args.pred_len, 1), dtype=torch.float32)
        id_list = list(np.arange(0, B, max(args.batch_size, 1))) + [B]
        for i in range(len(id_list) - 1):
            chunk = x[id_list[i]:id_list[i + 1]]
            outputs[id_list[i]:id_list[i + 1]] = model(chunk, 0).detach().cpu()[:, -args.pred_len:, :]

    preds = outputs.numpy()[:, :, 0]
    trues = np.array(y, dtype=object)

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(preds, columns=[f'V{i + 1}' for i in range(args.pred_len)])
    df.index = ids[:preds.shape[0]]
    df.index.name = 'id'
    df.set_index(df.columns[0], inplace=True)
    out_csv = os.path.join(out_dir, args.seasonal_patterns + '_forecast.csv')
    df.to_csv(out_csv)
    print(f'Saved forecasts → {out_csv}  shape={preds.shape}')
    return out_csv, preds, trues


def maybe_summarize(out_dir, root_path):
    needed = {f'{s}_forecast.csv' for s in HospitalMeta.seasonal_patterns}
    present = set(os.listdir(out_dir))
    if needed.issubset(present):
        smapes, mapes = HospitalSummary(out_dir + '/', root_path).evaluate()
        print('smape:', smapes)
        print('mape :', mapes)
        return smapes, mapes
    missing = needed - present
    print(f'[HospitalSummary] waiting for subsets: {sorted(missing)}')
    return None


def main():
    args = setup_args(build_parser().parse_args())

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Args:', args)

    run_tag = f"{args.model}_{args.seasonal_patterns}_sl{args.seq_len}_pl{args.pred_len}" \
              f"_dm{args.d_model}_df{args.d_ff}_lr{args.learning_rate}_bs{args.batch_size}" \
              f"_ps{args.scale_patch_sizes.replace(' ', '_')}"
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.it_fname = os.path.join(args.fname, run_tag, f"seed-{args.seed}", f"{now}_{uuid.uuid4().hex[:6]}")
    args.log_path = os.path.join(args.it_fname, 'log')
    args.pth_path = os.path.join(args.it_fname, 'pth')
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.pth_path, exist_ok=True)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_model(args, device)

    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = get_criterion(args.loss)
    mse = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=args.tmax, eta_min=args.eta_min
        )
    else:
        scheduler = None

    train_steps = len(train_loader)
    print(f"train_steps={train_steps} seq_len={args.seq_len} pred_len={args.pred_len}")

    for epoch in range(args.train_epochs):
        model.train()
        t0 = time.time()
        losses = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x, 0)
            outputs = outputs[:, -args.pred_len:, :]
            target = batch_y[:, -args.pred_len:, :]
            target_mask = batch_y_mark[:, -args.pred_len:, :]

            if args.loss in ('SMAPE', 'MAPE', 'MASE'):
                insample = batch_x[:, :, 0]
                loss = criterion(insample, args.frequency_map,
                                 outputs[:, :, 0], target[:, :, 0], target_mask[:, :, 0])
            else:
                loss = criterion(outputs, target)
            losses.append(loss.item())
            loss.backward()
            model_optim.step()

        train_loss = float(np.mean(losses)) if losses else 0.0
        vali_loss = vali(model, train_loader, vali_loader, criterion, args, device)
        log_line = (f"Epoch {epoch + 1} | time {time.time() - t0:.1f}s | "
                    f"train {train_loss:.6f} | val {vali_loss:.6f}")
        print(log_line)
        with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
            f.write(log_line + '\n')

        if scheduler is not None:
            scheduler.step()

        early_stopping(vali_loss, model, args.pth_path)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    best_pth = os.path.join(args.pth_path, 'checkpoint.pth')
    if os.path.exists(best_pth):
        model.load_state_dict(torch.load(best_pth, map_location=device))

    out_dir = os.path.join('./hospital_results', args.model)
    run_test(model, train_loader, test_loader, args, device, out_dir)
    maybe_summarize(out_dir, args.root_path)


if __name__ == '__main__':
    main()
