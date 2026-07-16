# M4 Dataset loader (adapted from N-BEATS / Short-term_Forecasting).
# CC BY-NC 4.0 (Element AI Inc., 2020)

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True, dataset_file: str = './datasets/m4') -> 'M4Dataset':
        info_file = os.path.join(dataset_file, 'M4-info.csv')
        train_cache_file = os.path.join(dataset_file, 'training.npz')
        test_cache_file = os.path.join(dataset_file, 'test.npz')
        m4_info = pd.read_csv(info_file)
        return M4Dataset(
            ids=m4_info.M4id.values,
            groups=m4_info.SP.values,
            frequencies=m4_info.Frequency.values,
            horizons=m4_info.Horizon.values,
            values=np.load(
                train_cache_file if training else test_cache_file,
                allow_pickle=True,
            ),
        )


@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6, 'Quarterly': 8, 'Monthly': 18,
        'Weekly': 13, 'Daily': 14, 'Hourly': 48,
    }
    frequency_map = {
        'Yearly': 1, 'Quarterly': 4, 'Monthly': 12,
        'Weekly': 1, 'Daily': 1, 'Hourly': 24,
    }
    history_size = {
        'Yearly': 1.5, 'Quarterly': 1.5, 'Monthly': 1.5,
        'Weekly': 10, 'Daily': 10, 'Hourly': 10,
    }
