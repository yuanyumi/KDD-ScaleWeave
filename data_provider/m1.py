
import os
from dataclasses import dataclass

import numpy as np

from data_provider.m3 import _parse_tsf


def _make_1d_object(vals):
    arr = np.empty(len(vals), dtype=object)
    arr[:] = vals
    return arr


@dataclass()
class M1Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly']
    horizons = [6, 8, 18]
    frequencies = [1, 4, 12]
    horizons_map = {
        'Yearly': 6, 'Quarterly': 8, 'Monthly': 18,
    }
    frequency_map = {
        'Yearly': 1, 'Quarterly': 4, 'Monthly': 12,
    }
    history_size = {
        'Yearly': 1.5, 'Quarterly': 1.5, 'Monthly': 1.5,
    }
    file_map = {
        'Yearly': 'm1_yearly_dataset.tsf',
        'Quarterly': 'm1_quarterly_dataset.tsf',
        'Monthly': 'm1_monthly_dataset.tsf',
    }


@dataclass()
class M1Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True, dataset_file: str = './datasets/m1') -> 'M1Dataset':
        all_ids, all_groups, all_freqs, all_horizons, all_values = [], [], [], [], []
        for sp in M1Meta.seasonal_patterns:
            fname = os.path.join(dataset_file, M1Meta.file_map[sp])
            ids, series = _parse_tsf(fname)
            h = M1Meta.horizons_map[sp]
            freq = M1Meta.frequency_map[sp]
            for sid, s in zip(ids, series):
                if len(s) <= h:
                    continue
                arr = s[:-h] if training else s[-h:]
                all_ids.append(sid)
                all_groups.append(sp)
                all_freqs.append(freq)
                all_horizons.append(h)
                all_values.append(arr)
        return M1Dataset(
            ids=np.array(all_ids),
            groups=np.array(all_groups),
            frequencies=np.array(all_freqs),
            horizons=np.array(all_horizons),
            values=_make_1d_object(all_values),
        )
