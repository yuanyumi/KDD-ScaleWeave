import numpy as np
from torch.utils.data import Dataset

from data_provider.kdd_cup import KDDCupDataset, KDDCupMeta


class Dataset_KDDCup(Dataset):
    """KDDCup dataset for short-term forecasting.
    Returns (insample, outsample, insample_mask, outsample_mask).
    insample:  (seq_len, 1)
    outsample: (label_len + pred_len, 1)
    masks:     0/1 same shape as corresponding sample.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='kdd_cup', target='OT',
                 scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly', **kwargs):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = KDDCupMeta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        if self.flag == 'train':
            dataset = KDDCupDataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = KDDCupDataset.load(training=False, dataset_file=self.root_path)

        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]],
            dtype=object,
        )
        self.ids = np.array(
            [i for i in dataset.ids[dataset.groups == self.seasonal_patterns]]
        )
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(
            low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
            high=len(sampled_timeseries),
            size=1,
        )[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0

        outsample_window = sampled_timeseries[
            cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)
        ]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def last_insample_window(self):
        """Last insample window of all timeseries. Shape (num_ts, seq_len)."""
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts_last_window):] = ts_last_window
            insample_mask[i, -len(ts_last_window):] = 1.0
        return insample, insample_mask
