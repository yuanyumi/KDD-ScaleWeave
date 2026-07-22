set -e
SOURCE_ROOT=./datasets/m4
TARGET_ROOT=./datasets/m3
OUT=./zero_shot_results

CUDA_VISIBLE_DEVICES=${GPU:-0} python -u main_zero_shot_m.py \
 --source m4 --target m3 \
 --source_root $SOURCE_ROOT --source_pattern Yearly \
 --target_root $TARGET_ROOT --target_path m3_yearly_dataset.tsf \
 --seq_len 12 --pred_len 6 --m4_freq 1 \
 --batch_size 128 --learning_rate 2.5e-3 \
 --train_epochs 60 --patience 10 \
 --loss SMAPE --lradj COS --tmax 20 \
 --dropout 0.10 --in_dropout 0.10 --out_dropout 0.10 \
 --scale_patch_sizes "1 4 12" --scale_strides "1 4 12" \
 --sch_layers 2 --cross_scale_g 1 --g_gate 1 --revin_flag 0 \
 --num_workers 8 --out_dir $OUT --tag m4y_to_m3y \
 --seed 2021

CUDA_VISIBLE_DEVICES=${GPU:-0} python -u main_zero_shot_m.py \
 --source m4 --target m3 \
 --source_root $SOURCE_ROOT --source_pattern Quarterly \
 --target_root $TARGET_ROOT --target_path m3_quarterly_dataset.tsf \
 --seq_len 16 --pred_len 8 --m4_freq 4 \
 --batch_size 128 --learning_rate 7e-4 \
 --train_epochs 80 --patience 10 \
 --loss SMAPE --lradj COS --tmax 20 \
 --dropout 0.10 --in_dropout 0.10 --out_dropout 0.10 \
 --scale_patch_sizes "2 4 8" --scale_strides "1 2 4" \
 --sch_layers 2 --cross_scale_g 1 --g_gate 1 --revin_flag 0 \
 --num_workers 4 --out_dir $OUT --tag m4q_to_m3q \
 --seed 2021

CUDA_VISIBLE_DEVICES=${GPU:-0} python -u main_zero_shot_m.py \
 --source m4 --target m3 \
 --source_root $SOURCE_ROOT --source_pattern Monthly \
 --target_root $TARGET_ROOT --target_path m3_monthly_dataset.tsf \
 --seq_len 48 --pred_len 18 --m4_freq 12 \
 --batch_size 1024 --learning_rate 4.8e-3 \
 --train_epochs 50 --patience 10 \
 --loss SMAPE --lradj COS --tmax 15 \
 --dropout 0.10 --in_dropout 0.10 --out_dropout 0.10 \
 --scale_patch_sizes "3 6 12" --scale_strides "3 6 12" \
 --sch_layers 2 --cross_scale_g 1 --g_gate 1 --revin_flag 0 \
 --num_workers 8 --out_dir $OUT --tag m4m_to_m3m \
 --seed 2021

CUDA_VISIBLE_DEVICES=${GPU:-0} python -u main_zero_shot_m.py \
 --source m4 --target m3 \
 --source_root $SOURCE_ROOT --source_pattern Quarterly \
 --target_root $TARGET_ROOT --target_path m3_other_dataset.tsf \
 --seq_len 24 --pred_len 8 --m4_freq 1 \
 --batch_size 128 --learning_rate 7e-4 \
 --train_epochs 80 --patience 10 \
 --loss SMAPE --lradj COS --tmax 20 \
 --dropout 0.05 --in_dropout 0.05 --out_dropout 0.05 \
 --scale_patch_sizes "3 6 12" --scale_strides "1 2 4" \
 --sch_layers 2 --cross_scale_g 1 --g_gate 1 --revin_flag 0 \
 --num_workers 4 --out_dir $OUT --tag m4q_to_m3o \
 --seed 2021

echo
echo "============================================================"
echo "M4 → M3 final per-cell results:"
echo "============================================================"
python -c "
import os, pandas as pd
cells = [
    ('Yearly', 'm4y_to_m3y', 645),
    ('Quarterly', 'm4q_to_m3q', 756),
    ('Monthly', 'm4m_to_m3m', 1428),
    ('Others', 'm4q_to_m3o', 174),
]
total_n, total_w = 0, 0.0
print(f'{\"Cell\":<12} {\"SMAPE\":>10} {\"n_series\":>10}')
for label, tag, n in cells:
    p = f'./zero_shot_results/{tag}/metrics.csv'
    if os.path.exists(p):
        s = float(pd.read_csv(p).iloc[0]['smape'])
        print(f'{label:<12} {s:>10.4f} {n:>10}')
        total_n += n; total_w += s * n
print('---')
print(f'Avg (n-weighted) = {total_w/total_n:.4f} (n={total_n})')
"
