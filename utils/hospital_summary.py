# Hospital summary utility — SMAPE-only.
# Mirrors utils/m3_summary.py; no "Other" subset.

import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from data_provider.hospital import HospitalDataset, HospitalMeta


def group_values(values, groups, group_name):
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])


def smape_2(forecast, target):
    denom = np.abs(target) + np.abs(forecast)
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom


def mape(forecast, target):
    denom = np.abs(target)
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom


class HospitalSummary:
    def __init__(self, file_path, root_path):
        self.file_path = file_path
        self.test_set = HospitalDataset.load(training=False, dataset_file=root_path)

    def evaluate(self):
        grouped_smapes, grouped_mapes = {}, {}
        for group_name in HospitalMeta.seasonal_patterns:
            file_name = self.file_path + group_name + '_forecast.csv'
            if not os.path.exists(file_name):
                continue
            model_forecast = pd.read_csv(file_name).values
            target = group_values(self.test_set.values, self.test_set.groups, group_name)
            target = np.stack([np.asarray(t, dtype=np.float32) for t in target])
            grouped_smapes[group_name] = float(np.mean(smape_2(forecast=model_forecast, target=target)))
            grouped_mapes[group_name] = float(np.mean(mape(forecast=model_forecast, target=target)))

        grouped_smapes = self.summarize_groups(grouped_smapes)
        grouped_mapes = self.summarize_groups(grouped_mapes)

        def round_all(d):
            return dict(map(lambda kv: (kv[0], round(float(kv[1]), 3)), d.items()))

        return round_all(grouped_smapes), round_all(grouped_mapes)

    def summarize_groups(self, scores):
        scores_summary = OrderedDict()

        def group_count(group_name):
            return int(np.sum(self.test_set.groups == group_name))

        weighted = {}
        total_count = 0
        for g in HospitalMeta.seasonal_patterns:
            if g not in scores:
                continue
            c = group_count(g)
            weighted[g] = scores[g] * c
            total_count += c
            scores_summary[g] = scores[g]

        if total_count > 0:
            scores_summary['Average'] = sum(weighted.values()) / total_count
        return scores_summary
