set -e
SOURCE_ROOT=./datasets/m3
TARGET_ROOT=./datasets/m4
OUT=./zero_shot_results

CUDA_VISIBLE_DEVICES=${GPU:-0} python -u main_zero_shot_m.py \
 --source m3 --target m4 \
 --source_root $SOURCE_ROOT --source_path m3_yearly_dataset.tsf \
 --target_root $TARGET_ROOT --target_pattern Yearly \
 --seq_len 10 --pred_len 6 --m4_freq 1 \
 --batch_size 128 --learning_rate 2.5e-3 \
 --train_epochs 80 --patience 10 \
 --loss SMAPE --lradj COS --tmax 20 \
 --dropout 0.10 --in_dropout 0.10 --out_dropout 0.10 \
 --scale_patch_sizes "1 2 5" --scale_strides "1 1 3" \
 --sch_layers 2 --cross_scale_g 1 --g_gate 1 --revin_flag 0 \
 --num_workers 8 --out_dir $OUT --tag m3y_to_m4y \
 --seed 2021

CUDA_VISIBLE_DEVICES=${GPU:-0} python -u main_zero_shot_m.py \
 --source m3 --target m4 \
 --source_root $SOURCE_ROOT --source_path m3_quarterly_dataset.tsf \
 --target_root $TARGET_ROOT --target_pattern Quarterly \
 --seq_len 12 --pred_len 8 --m4_freq 4 \
 --batch_size 128 --learning_rate 7e-4 \
 --train_epochs 80 --patience 10 \
 --loss SMAPE --lradj COS --tmax 20 \
 --dropout 0.15 --in_dropout 0.15 --out_dropout 0.15 \
 --scale_patch_sizes "1 2 4" --scale_strides "1 2 4" \
 --sch_layers 2 --cross_scale_g 1 --g_gate 1 --revin_flag 0 \
 --num_workers 8 --out_dir $OUT --tag m3q_to_m4q \
 --seed 2021

CUDA_VISIBLE_DEVICES=${GPU:-0} python -u main_zero_shot_m.py \
 --source m3 --target m4 \
 --source_root $SOURCE_ROOT --source_path m3_monthly_dataset.tsf \
 --target_root $TARGET_ROOT --target_pattern Monthly \
 --seq_len 24 --pred_len 18 --m4_freq 12 \
 --batch_size 128 --learning_rate 2e-3 \
 --train_epochs 50 --patience 10 \
 --loss SMAPE --lradj COS --tmax 20 \
 --dropout 0.05 --in_dropout 0.05 --out_dropout 0.05 \
 --scale_patch_sizes "3 6 12" --scale_strides "1 2 4" \
 --sch_layers 2 --cross_scale_g 1 --g_gate 1 --revin_flag 1 \
 --num_workers 8 --out_dir $OUT --tag m3m_to_m4m \
 --seed 2021

CUDA_VISIBLE_DEVICES=${GPU:-0} python -u main_zero_shot_m.py \
 --source m3 --target m4 \
 --source_root $SOURCE_ROOT --source_path m3_monthly_dataset.tsf \
 --target_root $TARGET_ROOT --target_pattern Weekly \
 --seq_len 26 --pred_len 13 --m4_freq 1 \
 --batch_size 16 --learning_rate 1e-3 \
 --train_epochs 80 --patience 15 \
 --loss SMAPE --lradj COS --tmax 20 \
 --dropout 0.0 --in_dropout 0.0 --out_dropout 0.0 \
 --scale_patch_sizes "3 6 13" --scale_strides "3 6 7" \
 --sch_layers 2 --cross_scale_g 1 --g_gate 1 --revin_flag 1 \
 --num_workers 8 --out_dir $OUT --tag m3m_to_m4w \
 --seed 2021

CUDA_VISIBLE_DEVICES=${GPU:-0} python -u main_zero_shot_m.py \
 --source m3 --target m4 \
 --source_root $SOURCE_ROOT --source_path m3_monthly_dataset.tsf \
 --target_root $TARGET_ROOT --target_pattern Daily \
 --seq_len 28 --pred_len 14 --m4_freq 1 \
 --batch_size 16 --learning_rate 3e-4 \
 --train_epochs 80 --patience 15 \
 --loss SMAPE --lradj COS --tmax 20 \
 --dropout 0.0 --in_dropout 0.0 --out_dropout 0.0 \
 --scale_patch_sizes "3 7 14" --scale_strides "2 4 7" \
 --sch_layers 2 --cross_scale_g 1 --g_gate 1 --revin_flag 1 \
 --num_workers 8 --out_dir $OUT --tag m3m_to_m4d \
 --seed 2021

CUDA_VISIBLE_DEVICES=${GPU:-0} python -u main_zero_shot_m.py \
 --source m3 --target m4 \
 --source_root $SOURCE_ROOT --source_path m3_monthly_dataset.tsf \
 --target_root $TARGET_ROOT --target_pattern Hourly \
 --seq_len 48 --pred_len 24 --token_len 24 \
 --test_seq_len 48 --test_pred_len 48 \
 --m4_freq 24 \
 --batch_size 128 --learning_rate 1e-3 \
 --train_epochs 30 --patience 5 \
 --loss SMAPE --lradj COS --tmax 15 \
 --dropout 0.10 --in_dropout 0.10 --out_dropout 0.10 \
 --scale_patch_sizes "1 6 24" --scale_strides "1 1 4" \
 --sch_layers 2 --cross_scale_g 1 --g_gate 1 --revin_flag 1 \
 --num_workers 4 --out_dir $OUT --tag m3m_to_m4h \
 --seed 2021

echo
echo "============================================================"
echo "M3 → M4 final per-cell results (run python aggregator):"
echo "============================================================"
python -c "
import os, pandas as pd
cells = [
    ('Yearly', 'm3y_to_m4y', 23000),
    ('Quarterly', 'm3q_to_m4q', 24000),
    ('Monthly', 'm3m_to_m4m', 48000),
    ('Weekly', 'm3m_to_m4w', 359),
    ('Daily', 'm3m_to_m4d', 4227),
    ('Hourly', 'm3m_to_m4h', 414),
]
total_n, total_w = 0, 0.0
others_n, others_w = 0, 0.0
print(f'{\"Cell\":<12} {\"SMAPE\":>10} {\"n_series\":>10}')
for label, tag, n in cells:
    p = f'./zero_shot_results/{tag}/metrics.csv'
    if os.path.exists(p):
        s = float(pd.read_csv(p).iloc[0]['smape'])
        print(f'{label:<12} {s:>10.4f} {n:>10}')
        total_n += n; total_w += s * n
        if label in ('Weekly', 'Daily', 'Hourly'):
            others_n += n; others_w += s * n
print('---')
print(f'Others bucket (W+D+H) = {others_w/others_n:.4f} (n={others_n})')
print(f'Avg (n-weighted) = {total_w/total_n:.4f} (n={total_n})')
"
