"""Unified M3 / M4 univariate loader for cross-dataset zero-shot.

- M3 → reads Monash `.tsf` files via the existing convert_tsf_to_dataframe.
- M4 → reads the project's existing `training.npz` + `test.npz` + `M4-info.csv`
  and concatenates train+horizon per series (no extra download needed).

Both branches yield the same `list[np.ndarray]` shape, then share one
sliding-window scheme:
  - train split: windows from start to (full_len - seq_len - 2*pred_len + 1)
  - val   split: window ending at  (full_len - pred_len)
  - test  split: window ending at   full_len  (last `pred_len` is GT)
"""

import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.tools import convert_tsf_to_dataframe


def _read_m3_tsf(root_path, data_path):
    """Return list[np.ndarray(float32)] from a Monash .tsf file."""
    df, *_ = convert_tsf_to_dataframe(os.path.join(root_path, data_path))
    out = []
    for ts in df.series_value:
        a = np.asarray(ts, dtype=np.float32)
        out.append(a[~np.isnan(a)])
    return out


def _read_m4_csv(root_path, seasonal_patterns):
    """Load M4 official train+test arrays for one bucket and concatenate per series.

    Uses M4-info.csv to mask the bucket (Yearly / Quarterly / Monthly / ...).
    """
    info = pd.read_csv(os.path.join(root_path, 'M4-info.csv'))
    train_v = np.load(os.path.join(root_path, 'training.npz'), allow_pickle=True)
    test_v = np.load(os.path.join(root_path, 'test.npz'), allow_pickle=True)

    mask = info.SP.values == seasonal_patterns
    out = []
    for tr, te in zip(train_v[mask], test_v[mask]):
        tr = np.asarray(tr, dtype=np.float32)
        te = np.asarray(te, dtype=np.float32)
        tr = tr[~np.isnan(tr)]
        te = te[~np.isnan(te)]
        out.append(np.concatenate([tr, te]))
    return out


class Dataset_MSeries(Dataset):
    """Universal M3/M4 zero-shot dataset.

    Parameters
    ----------
    root_path : str
        Either ./datasets/m3 (with .tsf files) or ./datasets/m4 (with .npz/.csv).
    data_path : str | None
        Required for M3: e.g. ``m3_monthly_dataset.tsf``. Ignored for M4.
    seasonal_patterns : str | None
        Required for M4: one of Yearly/Quarterly/Monthly/Weekly/Daily/Hourly.
        Ignored for M3.
    flag : {'train','val','test'}
    size : (seq_len, label_len, pred_len)
    """

    def __init__(self, root_path, flag='train', size=None,
                 data_path=None, seasonal_patterns=None, source='m3',
                 train_all=False, **_):
        assert flag in ('train', 'val', 'test')
        assert source in ('m3', 'm4')
        self.seq_len, self.label_len, self.pred_len = size
        self.flag = flag
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.train_all = bool(train_all)

        if source == 'm3':
            assert data_path is not None, 'data_path required for M3'
            ts_list = _read_m3_tsf(root_path, data_path)
        else:
            assert seasonal_patterns is not None, 'seasonal_patterns required for M4'
            ts_list = _read_m4_csv(root_path, seasonal_patterns)

        self.timeseries, self.len_seq, self.seq_id, self.tot_len = \
            self._build_index(ts_list)

    @property
    def ids(self):
        return np.arange(len(self.timeseries))

    def _build_index(self, ts_list):
        timeseries = []
        len_seq, seq_id = [], []
        tot = 0
        for i, raw in enumerate(ts_list):
            need = self.seq_len + self.pred_len - raw.shape[0]
            ts = np.concatenate([np.zeros(need, dtype=np.float32), raw]) if need > 0 else raw
            timeseries.append(ts)

            _len = ts.shape[0]
            train_len = _len - self.pred_len
            if self.train_all:
                border1s = [0, 0, train_len - self.seq_len]
                border2s = [train_len, train_len, _len]
            else:
                border1s = [0,
                            train_len - self.seq_len - self.pred_len,
                            train_len - self.seq_len]
                border2s = [train_len - self.pred_len,
                            train_len,
                            _len]
            cur = border2s[self.set_type] - max(border1s[self.set_type], 0) \
                  - self.pred_len - self.seq_len + 1
            cur = max(0, cur)
            len_seq.append(np.full(cur, tot))
            seq_id.append(np.full(cur, i))
            tot += cur

        return (timeseries,
                np.hstack(len_seq) if len_seq else np.array([]),
                np.hstack(seq_id) if seq_id else np.array([]),
                tot)

    def __len__(self):
        return self.tot_len

    def __getitem__(self, index):
        offset = int(self.len_seq[index])
        sid = int(self.seq_id[index])
        local = index - offset

        ts = self.timeseries[sid]
        _len = ts.shape[0]
        train_len = _len - self.pred_len
        if self.train_all:
            border1s = [0, 0, train_len - self.seq_len]
        else:
            border1s = [0,
                        train_len - self.seq_len - self.pred_len,
                        train_len - self.seq_len]

        s_begin = local + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        x = ts[s_begin:s_end][:, None]
        y = ts[r_begin:r_end][:, None]
        return x.astype(np.float32), y.astype(np.float32), x, y

    def last_insample_window(self):
        """One window per series, last `seq_len` of train portion. Used for batch eval."""
        N = len(self.timeseries)
        x = np.zeros((N, self.seq_len), dtype=np.float32)
        m = np.zeros((N, self.seq_len), dtype=np.float32)
        for i, ts in enumerate(self.timeseries):
            train_len = ts.shape[0] - self.pred_len
            w = ts[max(0, train_len - self.seq_len):train_len]
            x[i, -len(w):] = w
            m[i, -len(w):] = 1.0
        return x, m
