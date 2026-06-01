# -*- coding: utf-8 -*-
"""
DCASTformer: Dual-Channel Adaptive Spatio-Temporal Transformer

修复 TB_DualExogFusion 的 fusion 参数未训练问题。
支持按数据集调参。
"""
import json, os, subprocess, sys

TP_ONLY_MODE = True
TP_CHANNEL = 7
DEFAULT_TARGET_CHANNEL = -4

MODEL_NAME = "dcastformer.DCASTformer"

strategy_args = {
    "horizon": 24,
    "target_channel": [TP_CHANNEL if TP_ONLY_MODE else DEFAULT_TARGET_CHANNEL],
}

# =====================================================
# 基础超参（所有数据集共享）
# =====================================================
base_params = {
    "batch_size": 64, "seq_len": 96, "horizon": 24, "period": 24,
    "d_model": 128, "d_ff": 128, "n_heads": 4,
    "dropout": 0, "attn_dropout": 0.15, "activation": "gelu",
    "stable_len": 2, "revin": 1,
    "ia_layers": 1, "ca_layers": 1,
    "attn_mode": "full", "layer_order": "int_coint",
    "use_future_exog": True, "use_history_exog": True,
    "infer_use_future": True,
    "num_epochs": 100, "patience": 10, "lradj": "type1", "lr": 0.001, "loss": "MSE",
}

# =====================================================
# 按数据集配置参数（方便调参）
# =====================================================
# sigmoid(alpha_init) = 初始融合权重
#   -2.0 → 0.12 (偏向 embedding_concat)
#    0.0 → 0.50 (中性)
#    2.0 → 0.88 (偏向 gated_overwrite)

DATASET_CONFIGS = {
    "桔子洲2": {
        "lr": 0.001,
        "alpha_init": 0.0,       # 搜参最优 (sigmoid=0.50)
        "说明": "搜参最优: alpha=0.0, mse=0.2843",
    },
    "三角洲2": {
        "lr": 0.001,
        "alpha_init": -0.5,      # 搜参最优 (sigmoid=0.38)
        "说明": "搜参最优: alpha=-0.5, mse=0.2968",
    },
    "捞刀河2": {
        "lr": 0.001,
        "alpha_init": 3.0,       # 搜参最优 (sigmoid=0.95)
        "说明": "搜参最优: alpha=3.0, mse=0.0726",
    },
}

for dataset_name, cfg in DATASET_CONFIGS.items():
    model_hyper_params = {
        **base_params,
        "lr": cfg["lr"],
        "alpha_init": cfg["alpha_init"],
    }

    save_path = f"{dataset_name}/DCASTformer" + ("_TP" if TP_ONLY_MODE else "")

    print(f"\n{'='*60}")
    print(f"DCASTformer on {dataset_name}")
    print(f"  lr={cfg['lr']}, alpha_init={cfg['alpha_init']}")
    print(f"  {cfg['说明']}")
    print(f"{'='*60}\n")

    args = [
        sys.executable, "./scripts/run_benchmark.py",
        "--config-path", "rolling_forecast_config.json",
        "--data-name-list", dataset_name + ".csv",
        "--strategy-args", json.dumps(strategy_args),
        "--model-name", MODEL_NAME,
        "--model-hyper-params", json.dumps(model_hyper_params),
        "--gpus", "0", "--num-workers", "1", "--timeout", "60000",
        "--save-path", save_path, "--deterministic", "full",
    ]

    subprocess.run(args, cwd=os.path.dirname(__file__) if __file__ else os.getcwd())
