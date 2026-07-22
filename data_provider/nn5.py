
import os
from dataclasses import dataclass

import numpy as np

from data_provider.m3 import _parse_tsf


def _make_1d_object(vals):
    arr = np.empty(len(vals), dtype=object)
    arr[:] = vals
    return arr


@dataclass()
class NN5Meta:
    seasonal_patterns = ['Weekly']
    horizons = [8]
    frequencies = [1]
    horizons_map = {'Weekly': 8}
    frequency_map = {'Weekly': 1}
    history_size = {'Weekly': 1.5}
    file_map = {'Weekly': 'nn5_weekly_dataset.tsf'}


@dataclass()
class NN5Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True, dataset_file: str = './datasets/nn5') -> 'NN5Dataset':
        all_ids, all_groups, all_freqs, all_horizons, all_values = [], [], [], [], []
        for sp in NN5Meta.seasonal_patterns:
            fname = os.path.join(dataset_file, NN5Meta.file_map[sp])
            ids, series = _parse_tsf(fname)
            h = NN5Meta.horizons_map[sp]
            freq = NN5Meta.frequency_map[sp]
            for sid, s in zip(ids, series):
                if len(s) <= h:
                    continue
                arr = s[:-h] if training else s[-h:]
                all_ids.append(sid)
                all_groups.append(sp)
                all_freqs.append(freq)
                all_horizons.append(h)
                all_values.append(arr)
        return NN5Dataset(
            ids=np.array(all_ids),
            groups=np.array(all_groups),
            frequencies=np.array(all_freqs),
            horizons=np.array(all_horizons),
            values=_make_1d_object(all_values),
        )
