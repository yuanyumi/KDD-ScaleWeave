# Hospital Dataset (767 monthly patient-count series, Jan 2000 - Dec 2006).
# Single subset "Monthly", horizon 12 (1 year) per N-BEATS-style convention.
# Note: the .tsf has no @horizon header — h=12 is the literature standard.

import os
from dataclasses import dataclass

import numpy as np

from data_provider.m3 import _parse_tsf


def _make_1d_object(vals):
    arr = np.empty(len(vals), dtype=object)
    arr[:] = vals
    return arr


@dataclass()
class HospitalMeta:
    seasonal_patterns = ['Monthly']
    horizons = [12]
    frequencies = [12]
    horizons_map = {'Monthly': 12}
    frequency_map = {'Monthly': 12}
    history_size = {'Monthly': 1.5}
    file_map = {'Monthly': 'hospital_dataset.tsf'}


@dataclass()
class HospitalDataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True, dataset_file: str = './datasets/hospital') -> 'HospitalDataset':
        all_ids, all_groups, all_freqs, all_horizons, all_values = [], [], [], [], []
        for sp in HospitalMeta.seasonal_patterns:
            fname = os.path.join(dataset_file, HospitalMeta.file_map[sp])
            ids, series = _parse_tsf(fname)
            h = HospitalMeta.horizons_map[sp]
            freq = HospitalMeta.frequency_map[sp]
            for sid, s in zip(ids, series):
                if len(s) <= h:
                    continue
                arr = s[:-h] if training else s[-h:]
                all_ids.append(sid)
                all_groups.append(sp)
                all_freqs.append(freq)
                all_horizons.append(h)
                all_values.append(arr)
        return HospitalDataset(
            ids=np.array(all_ids),
            groups=np.array(all_groups),
            frequencies=np.array(all_freqs),
            horizons=np.array(all_horizons),
            values=_make_1d_object(all_values),
        )
