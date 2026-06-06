# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys


TP_ONLY_MODE = True
TP_CHANNEL = 7
DEFAULT_TARGET_CHANNEL = -4

MODEL_NAME = "p2dformer.P2DFormer"
MODEL_DIR = "P2DFormer"

strategy_args = {
    "horizon": 24,
    "target_channel": [TP_CHANNEL if TP_ONLY_MODE else DEFAULT_TARGET_CHANNEL],
}

# Base params
base_params = {
    "batch_size": 64,
    "seq_len": 96,
    "horizon": 24,
    "pred_len": 24,
    "d_model": 128,
    "d_ff": 256,
    "n_heads": 4,
    "e_layers": 2,
    "top_k": 3,
    "use_future_exog": True,
    "use_history_exog": True,
    "num_epochs": 100,
    "patience": 10,
    "lr": 0.001,
    "lradj": "type1",
    "loss": "MSE",
    "norm": True,
}

# Per-dataset configs
DATASET_CONFIGS = {
    "juzizhou": {
        "dropout": 0.0,
        "patch_len": 16,
        "note": "Best config: dropout=0.0, patch_len=16"
    },
    "sanjiaozhou": {
        "dropout": 0.1,
        "patch_len": 8,
        "note": "Bad config: dropout=0.1, patch_len=8"
    },
    "laodaohe": {
        "dropout": 0.1,
        "patch_len": 8,
        "note": "Bad config: dropout=0.1, patch_len=8"
    },
}

for dataset_name, cfg in DATASET_CONFIGS.items():
    model_hyper_params = {
        **base_params,
        "dropout": cfg["dropout"],
        "patch_len": cfg["patch_len"],
    }

    save_path = f"{dataset_name}/{MODEL_DIR}"
    if TP_ONLY_MODE:
        save_path += "_TP"

    print(f"\n{'='*60}")
    print(f"P2DFormer on {dataset_name}")
    print(f"  dropout={cfg['dropout']}, patch_len={cfg['patch_len']}")
    print(f"  {cfg['note']}")
    print(f"{'='*60}\n")

    args = [
        sys.executable,
        "./scripts/run_benchmark.py",
        "--config-path",
        "rolling_forecast_config.json",
        "--data-name-list",
        dataset_name + ".csv",
        "--strategy-args",
        json.dumps(strategy_args),
        "--model-name",
        MODEL_NAME,
        "--model-hyper-params",
        json.dumps(model_hyper_params),
        "--gpus",
        "0",
        "--num-workers",
        "1",
        "--timeout",
        "60000",
        "--save-path",
        save_path,
        "--deterministic",
        "full",
    ]

    subprocess.run(args, cwd=os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd())
