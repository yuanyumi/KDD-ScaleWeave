# KDD Cup 2018 Dataset (270 hourly air-quality time series, Beijing+London).
# Single subset "Hourly", horizon 168 (1 week of hourly forecasts).

import os
from dataclasses import dataclass

import numpy as np

from data_provider.m3 import _parse_tsf


def _make_1d_object(vals):
    arr = np.empty(len(vals), dtype=object)
    arr[:] = vals
    return arr


@dataclass()
class KDDCupMeta:
    seasonal_patterns = ['Hourly']
    horizons = [168]
    frequencies = [24]  # daily seasonality on hourly data
    horizons_map = {'Hourly': 168}
    frequency_map = {'Hourly': 24}
    history_size = {'Hourly': 1.5}
    file_map = {'Hourly': 'kdd_cup_2018_dataset_without_missing_values.tsf'}


@dataclass()
class KDDCupDataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True, dataset_file: str = './datasets/kdd_cup') -> 'KDDCupDataset':
        all_ids, all_groups, all_freqs, all_horizons, all_values = [], [], [], [], []
        for sp in KDDCupMeta.seasonal_patterns:
            fname = os.path.join(dataset_file, KDDCupMeta.file_map[sp])
            ids, series = _parse_tsf(fname)
            h = KDDCupMeta.horizons_map[sp]
            freq = KDDCupMeta.frequency_map[sp]
            for sid, s in zip(ids, series):
                if len(s) <= h:
                    continue
                arr = s[:-h] if training else s[-h:]
                # KDD Cup series_name can repeat across (city,station,measurement)
                # — disambiguate the saved id by appending a row index.
                all_ids.append(f'{sid}_{len(all_ids)}')
                all_groups.append(sp)
                all_freqs.append(freq)
                all_horizons.append(h)
                all_values.append(arr)
        return KDDCupDataset(
            ids=np.array(all_ids),
            groups=np.array(all_groups),
            frequencies=np.array(all_freqs),
            horizons=np.array(all_horizons),
            values=_make_1d_object(all_values),
        )
