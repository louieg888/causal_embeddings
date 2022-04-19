# grid hyperparmeter search example
# comparing off and defaults
spell hyper grid \
    -t V100 \
    -d "param off and defaults, ablation v0" \
    --project causally_consistent_AE \
    --pip-req requirements.txt \
    --param alpha=0,0.00000001 \
    --param beta=0,0.01 \
    --param gamma=0,0.00001 \
    --param rho=0,1 \
    --param batch=64 \
    --param lr=0.001 \
    --mount s3://gred-ai-sandbox-prod/users/iriondoc/:/spell/causal_embeddings/datasets \
    "python train_causal_autoencoder.py \
    --alpha :alpha: \
    --beta :beta: \
    --gamma :gamma: \
    --rho :rho: \
    --batch :batch: \
    --lr :lr: \
    "

# bayesian hyperparmeter search example
# note: we can't use this now because the val_loss depends on these weights,
# minimizing val loss will just set alpha/beta/gamma/rho to 0
# also, check how batch size is affecting val_loss, the last batch might impact
# averaging (can change val batch size to 1 to avoid this)

spell hyper bayesian \
    --num-runs 20 \
    --parallel-runs 3 \
    --metric val_loss_avg \
    --metric-agg min \
    --param rate=.001:1.0 \
    --param layers=2:100:int \
    python train.py --learning_rate :rate: --num_layers :layers:
