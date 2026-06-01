#!/bin/bash
# ============================================================
# DCASTFormer: Dual-Channel Adaptive Spatio-Temporal Transformer
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

TP_ONLY=true
TP_CHANNEL=7
DEFAULT_TARGET_CHANNEL=-4

DATASETS=("桔子洲2" "三角洲2" "捞刀河2")
MODEL_NAME="dcastformer.DCASTformer"
MODEL_DIR="DCASTformer"

HORIZON=24
TARGET_CHANNEL=$TP_CHANNEL

# 基础超参
BATCH_SIZE=64
SEQ_LEN=96
PERIOD=24
D_MODEL=128
D_FF=128
N_HEADS=4
DROPOUT=0
ATTN_DROPOUT=0.15
ACTIVATION="gelu"
STABLE_LEN=2
REVIN=1
IA_LAYERS=1
CA_LAYERS=1
ATTN_MODE="full"
LAYER_ORDER="int_coint"
NUM_EPOCHS=100
PATIENCE=10
LRADJ="type1"
LR=0.001
LOSS="MSE"

# 按数据集配置 alpha_init
declare -A ALPHA_INIT
ALPHA_INIT["桔子洲2"]=0.0
ALPHA_INIT["三角洲2"]=-0.5
ALPHA_INIT["捞刀河2"]=3.0

STRATEGY_ARGS="{\"horizon\": $HORIZON, \"target_channel\": [$TARGET_CHANNEL]}"

for DATASET in "${DATASETS[@]}"; do
    ALPHA=${ALPHA_INIT[$DATASET]}
    SAVE_PATH="${DATASET}/${MODEL_DIR}"
    if [ "$TP_ONLY" = true ]; then
        SAVE_PATH="${SAVE_PATH}_TP"
    fi

    MODEL_HYPER_PARAMS="{\"batch_size\": $BATCH_SIZE, \"seq_len\": $SEQ_LEN, \"horizon\": $HORIZON, \"period\": $PERIOD, \"d_model\": $D_MODEL, \"d_ff\": $D_FF, \"n_heads\": $N_HEADS, \"dropout\": $DROPOUT, \"attn_dropout\": $ATTN_DROPOUT, \"activation\": \"$ACTIVATION\", \"stable_len\": $STABLE_LEN, \"revin\": $REVIN, \"ia_layers\": $IA_LAYERS, \"ca_layers\": $CA_LAYERS, \"attn_mode\": \"$ATTN_MODE\", \"layer_order\": \"$LAYER_ORDER\", \"use_future_exog\": true, \"use_history_exog\": true, \"infer_use_future\": true, \"alpha_init\": $ALPHA, \"num_epochs\": $NUM_EPOCHS, \"patience\": $PATIENCE, \"lradj\": \"$LRADJ\", \"lr\": $LR, \"loss\": \"$LOSS\"}"

    echo "============================================================"
    echo "DCASTFormer on $DATASET"
    echo "  lr=$LR, alpha_init=$ALPHA"
    echo "============================================================"

    python ./scripts/run_benchmark.py \
        --config-path rolling_forecast_config.json \
        --data-name-list "$DATASET.csv" \
        --strategy-args "$STRATEGY_ARGS" \
        --model-name "$MODEL_NAME" \
        --model-hyper-params "$MODEL_HYPER_PARAMS" \
        --gpus 0 \
        --num-workers 1 \
        --timeout 60000 \
        --save-path "$SAVE_PATH" \
        --deterministic full

    echo ""
done

echo "Done."
