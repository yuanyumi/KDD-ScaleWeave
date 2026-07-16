python -u main_tourism.py \
  --seasonal_patterns Monthly \
  --root_path ./datasets/tourism/ \
  --train_epochs 50 --patience 10 --batch_size 16 --num_workers 4 \
  --learning_rate 5e-4 --loss SMAPE --lradj COS --tmax 20 \
  --d_model 128 --d_ff 256 --n_heads 16 --e_layers 3 --transformer_ff_inner 256 \
  --dropout 0.0 --in_dropout 0.0 --out_dropout 0.0 \
  --hsg_layers 2 --cross_scale_g 1 --g_gate 1 \
  --scale_patch_sizes "4 12 24" --scale_strides "4 12 24" \
  --multi 0 --revin_flag 1 --seed 42
