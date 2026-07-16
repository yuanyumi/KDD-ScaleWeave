import torch
from torch.utils.data import DataLoader


def data_provider(args, flag, drop_last_test=True, train_all=False, train_shuffle=True):
    if args.data in ('m4', 'm3', 'm1', 'tourism', 'nn5', 'hospital', 'kdd_cup'):
        if args.data == 'm4':
            from data_provider.data_loader_m4 import Dataset_M4 as DatasetShort
        elif args.data == 'm3':
            from data_provider.data_loader_m3 import Dataset_M3 as DatasetShort
        elif args.data == 'm1':
            from data_provider.data_loader_m1 import Dataset_M1 as DatasetShort
        elif args.data == 'tourism':
            from data_provider.data_loader_tourism import Dataset_Tourism as DatasetShort
        elif args.data == 'nn5':
            from data_provider.data_loader_nn5 import Dataset_NN5 as DatasetShort
        elif args.data == 'hospital':
            from data_provider.data_loader_hospital import Dataset_Hospital as DatasetShort
        else:
            from data_provider.data_loader_kdd_cup import Dataset_KDDCup as DatasetShort
        data_set = DatasetShort(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=getattr(args, 'features', 'M'),
            seasonal_patterns=args.seasonal_patterns,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=(flag == 'train'),
            num_workers=args.num_workers,
            drop_last=False,
        )
        return data_set, data_loader

    if args.multi:
        from data_provider.data_loader_mul import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
    else:
        from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute

    data_dict = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,

        'custom_illness': Dataset_Custom,
        'custom_flight': Dataset_Custom,
        'custom_cars': Dataset_Custom,
        'custom_metr_la': Dataset_Custom,
        'custom_dowjones': Dataset_Custom,
        'custom_nasdaq': Dataset_Custom,
        'custom_sp500': Dataset_Custom,
        'custom_wiki': Dataset_Custom,
    }

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    max_len = args.max_len

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = train_shuffle
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        max_len=max_len,
        train_all=train_all
    )

    if args.percent < 1.0 and flag == 'train':
        num_samples = int(len(data_set) * args.percent)
        order = getattr(args, 'few_shot_order', 'random')
        if order == 'front':
            indices = torch.arange(num_samples)
        else:
            indices = torch.randperm(len(data_set))[:num_samples]
        data_set = torch.utils.data.Subset(data_set, indices)
        drop_last = False

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
