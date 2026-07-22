python -u main_tourism.py \
  --seasonal_patterns Quarterly \
  --root_path ./datasets/tourism/ \
  --train_epochs 50 --patience 10 --batch_size 16 --num_workers 4 \
  --learning_rate 2e-3 --loss SMAPE --lradj COS --tmax 20 \
  --d_model 64 --d_ff 128 --n_heads 8 --e_layers 3 --transformer_ff_inner 128 \
  --dropout 0.0 --in_dropout 0.0 --out_dropout 0.0 \
  --sch_layers 2 --cross_scale_g 1 --g_gate 1 \
  --scale_patch_sizes "2 4 8" --scale_strides "2 4 8" \
  --multi 0 --revin_flag 1 --seed 42
