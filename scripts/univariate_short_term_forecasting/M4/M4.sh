python -u main_m4.py \
 --seasonal_patterns Yearly --root_path ./datasets/m4/ \
 --train_epochs 80 --patience 15 --batch_size 128 --num_workers 4 \
 --learning_rate 2.5e-3 --loss SMAPE --lradj COS --tmax 20 \
 \
 --d_model 128 --d_ff 256 --n_heads 16 --e_layers 3 \
 --dropout 0.1 --in_dropout 0.1 --out_dropout 0.1 \
 --hsg_layers 2 --cross_scale_g 1 --g_gate 1 \
 --scale_patch_sizes "1 4 12" --scale_strides "1 4 12" \
 --multi 0 --revin_flag 0 --seed 42 --model ScaleWeave

python -u main_m4.py \
 --seasonal_patterns Quarterly --root_path ./datasets/m4/ \
 --seq_len 32 \
 --train_epochs 200 --patience 30 --batch_size 128 --num_workers 4 \
 --learning_rate 7e-4 --loss SMAPE --lradj COS --tmax 20 \
 \
 --d_model 128 --d_ff 256 --n_heads 16 --e_layers 3 \
 --dropout 0.1 --in_dropout 0.1 --out_dropout 0.1 \
 --hsg_layers 2 --cross_scale_g 1 --g_gate 1 \
 --scale_patch_sizes "4 8 16" --scale_strides "4 8 16" \
 --multi 0 --revin_flag 0 --seed 42 --model ScaleWeave

python -u main_m4.py \
 --seasonal_patterns Monthly --root_path ./datasets/m4/ \
 --seq_len 42 \
 --train_epochs 200 --patience 30 --batch_size 128 --num_workers 4 \
 --learning_rate 1.2e-3 --loss SMAPE --lradj COS --tmax 20 \
 \
 --d_model 128 --d_ff 256 --n_heads 16 --e_layers 3 \
 --dropout 0.05 --in_dropout 0.05 --out_dropout 0.05 \
 --hsg_layers 2 --cross_scale_g 1 --g_gate 1 \
 --scale_patch_sizes "3 7 14" --scale_strides "3 7 14" \
 --multi 0 --revin_flag 1 --seed 42 --model ScaleWeave

python -u main_m4.py \
 --seasonal_patterns Hourly --root_path ./datasets/m4/ \
 --train_epochs 200 --patience 30 --batch_size 16 --num_workers 4 \
 --learning_rate 1e-4 --loss SMAPE --lradj COS --tmax 20 \
 \
 --d_model 128 --d_ff 256 --n_heads 16 --e_layers 3 \
 --dropout 0.0 --in_dropout 0.0 --out_dropout 0.0 \
 --hsg_layers 2 --cross_scale_g 1 --g_gate 1 \
 --scale_patch_sizes "12 24 48" --scale_strides "6 12 24" \
 --multi 0 --revin_flag 1 --seed 42 --model ScaleWeave

python -u main_m4.py \
 --seasonal_patterns Daily --root_path ./datasets/m4/ \
 --train_epochs 80 --patience 15 --batch_size 16 --num_workers 4 \
 --learning_rate 3e-4 --loss SMAPE --lradj COS --tmax 20 \
 \
 --d_model 128 --d_ff 256 --n_heads 16 --e_layers 3 \
 --dropout 0.0 --in_dropout 0.0 --out_dropout 0.0 \
 --hsg_layers 2 --cross_scale_g 1 --g_gate 1 \
 --scale_patch_sizes "3 7 14" --scale_strides "2 4 7" \
 --multi 0 --revin_flag 1 --seed 42 --model ScaleWeave

python -u main_m4.py \
 --seasonal_patterns Weekly --root_path ./datasets/m4/ \
 --train_epochs 80 --patience 15 --batch_size 16 --num_workers 4 \
 --learning_rate 1e-3 --loss SMAPE --lradj COS --tmax 20 \
 \
 --d_model 128 --d_ff 256 --n_heads 16 --e_layers 3 \
 --dropout 0.0 --in_dropout 0.0 --out_dropout 0.0 \
 --hsg_layers 2 --cross_scale_g 1 --g_gate 1 \
 --scale_patch_sizes "3 6 13" --scale_strides "3 6 7" \
 --multi 0 --revin_flag 1 --seed 42 --model ScaleWeave

python -u -c "
from utils.m4_summary import M4Summary
s, owa, mp, ms = M4Summary('./m4_results/ScaleWeave/', './datasets/m4/').evaluate()
print('Bucket | SMAPE | MASE | OWA')
print('-----------|---------|---------|--------')
for k in ['Yearly','Quarterly','Monthly','Others','Average']:
 print(f'{k:<11}| {s[k]:>7} | {ms[k]:>7} | {owa[k]:>6}')
"
