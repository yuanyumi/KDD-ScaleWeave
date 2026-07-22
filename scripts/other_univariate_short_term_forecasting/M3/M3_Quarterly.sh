python -u main_m3.py \
  --seasonal_patterns Quarterly \
  --root_path ./datasets/m3/ \
  --train_epochs 50 --patience 10 --batch_size 16 --num_workers 2 \
  --learning_rate 1e-3 --loss SMAPE --lradj COS --tmax 20 \
  --d_model 64 --d_ff 128 --n_heads 8 --e_layers 3 --transformer_ff_inner 128 \
  --dropout 0.1 --in_dropout 0.1 --out_dropout 0.1 \
  --hsg_layers 2 --cross_scale_g 1 --g_gate 1 \
  --scale_patch_sizes "2 4 8" --scale_strides "2 4 8" \
  --multi 0 --revin_flag 0 --seed 42 --exp_tag M3_Quarterly
