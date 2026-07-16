# ScaleWeave: Weaving Variable Relations Across Temporal Scales for Multivariate Time Series Forecasting

## Environment

This project was tested using Python 3.8 and CUDA 12.1.

```bash
conda create -n scaleweave python=3.8 -y
conda activate scaleweave
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Training & Reproduction

We provide experiment scripts for all benchmark datasets in the ./scripts folder. To reproduce our results, you can run the corresponding script.

For example:

```bash
conda activate scaleweave
bash scripts/long_term_forecasting/ETTh1/ETTh1_96.sh
bash scripts/long_term_forecasting/Flight/Flight_96.sh
bash scripts/long_term_forecasting/ILI/ILI_24.sh
bash scripts/multivariate_short_term_forecasting/Cars/Cars_h24.sh
bash scripts/univariate_short_term_forecasting/M4/M4.sh
bash scripts/few_shot_forecasting/ETTh1/ETTh1_96.sh
bash scripts/multivariate_zero_shot/ETTh1_ETTh2.sh
bash scripts/univariate_zero_shot/m3_to_m4.sh
bash scripts/other_univariate_short_term_forecasting/M1/M1_Yearly.sh
```
