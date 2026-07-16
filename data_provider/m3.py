# M3 Dataset loader for short-term forecasting.
# Format follows Monash .tsf (https://forecastingdata.org/).
# Mirrors data_provider/m4.py so Dataset_M3 / M3Summary can be near-clones.

import os
from dataclasses import dataclass

import numpy as np


def _make_1d_object(vals):
    arr = np.empty(len(vals), dtype=object)
    arr[:] = vals
    return arr


@dataclass()
class M3Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Other']
    horizons = [6, 8, 18, 8]
    frequencies = [1, 4, 12, 1]
    horizons_map = {
        'Yearly': 6, 'Quarterly': 8, 'Monthly': 18, 'Other': 8,
    }
    frequency_map = {
        'Yearly': 1, 'Quarterly': 4, 'Monthly': 12, 'Other': 1,
    }
    # window_sampling_limit = history_size * pred_len in Dataset_M3.
    # M3 series are short (Yearly avg ~22 pts), so keep N-BEATS-style 1.5.
    history_size = {
        'Yearly': 1.5, 'Quarterly': 1.5, 'Monthly': 1.5, 'Other': 1.5,
    }
    file_map = {
        'Yearly': 'm3_yearly_dataset.tsf',
        'Quarterly': 'm3_quarterly_dataset.tsf',
        'Monthly': 'm3_monthly_dataset.tsf',
        'Other': 'm3_other_dataset.tsf',
    }


def _parse_tsf(path):
    """Parse a Monash .tsf file. Returns (ids, list_of_float_arrays).

    Generalised: counts @attribute lines in the header to know how many
    colon-separated fields precede the values. Handles M1/M3/Tourism (2
    attrs), M3-Other (1 attr), KDD Cup (5 attrs: name/city/station/measure/ts),
    etc.
    """
    n_attrs = 0
    ids, values = [], []
    in_data = False
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@'):
                if line.lower() == '@data':
                    in_data = True
                elif line.lower().startswith('@attribute'):
                    n_attrs += 1
                continue
            if not in_data:
                continue
            parts = line.split(':')
            sid = parts[0]
            vals_str = parts[n_attrs] if n_attrs >= 1 else parts[-1]
            vals = np.array(
                [float(v) for v in vals_str.split(',') if v not in ('?', '')],
                dtype=np.float32,
            )
            ids.append(sid)
            values.append(vals)
    return ids, values


@dataclass()
class M3Dataset:
    """Container mirroring M4Dataset's interface.

    `values[i]` is either the training prefix (training=True) or the held-out
    last-`horizon` window (training=False) of series i.
    """
    ids: np.ndarray
    groups: np.ndarray         # seasonal pattern per series
    frequencies: np.ndarray    # int frequency per series (1/4/12)
    horizons: np.ndarray       # int horizon per series
    values: np.ndarray         # object array of 1-D float32 arrays

    @staticmethod
    def load(training: bool = True, dataset_file: str = './datasets/m3') -> 'M3Dataset':
        all_ids, all_groups, all_freqs, all_horizons, all_values = [], [], [], [], []
        for sp in M3Meta.seasonal_patterns:
            fname = os.path.join(dataset_file, M3Meta.file_map[sp])
            ids, series = _parse_tsf(fname)
            h = M3Meta.horizons_map[sp]
            freq = M3Meta.frequency_map[sp]
            for sid, s in zip(ids, series):
                if len(s) <= h:
                    # series too short to evaluate — skip
                    continue
                if training:
                    arr = s[:-h]
                else:
                    arr = s[-h:]
                all_ids.append(sid)
                all_groups.append(sp)
                all_freqs.append(freq)
                all_horizons.append(h)
                all_values.append(arr)
        return M3Dataset(
            ids=np.array(all_ids),
            groups=np.array(all_groups),
            frequencies=np.array(all_freqs),
            horizons=np.array(all_horizons),
            values=_make_1d_object(all_values),
        )
