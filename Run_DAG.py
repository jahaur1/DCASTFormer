# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys


TP_ONLY_MODE = True
TP_CHANNEL = 7
DEFAULT_TARGET_CHANNEL = -4

DATASETS = [
    "juzizhou",
    "sanjiaozhou",
    "laodaohe",
]
MODEL_NAME = "dag.DAG"
MODEL_DIR = "DAG"

# 每个数据集的配置
DATASET_CONFIGS = {
    "juzizhou": {
        "seq_len": 96,
        "d_model": 256,
        "d_ff": 1024,
        "n_heads": 8,
        "patch_len": 16,
        "stride": 8,
        "factor": 1,
        "e_layers": 1,
        "dropout": 0.1,
        "lr": 0.01,
        "num_epochs": 100,
        "patience": 10,
        "loss": "MSE",
        "lradj": "type3",
        "batch_size": 256,
        "activation": "gelu",
        "dbloss_alpha": 0.35,
        "dbloss_beta": 0.65,
        "alpha": 0.35,
        "beta": 0.175,
        "use_c_exog": True,
        "use_t_exog": True,
        "use_c": True,
        "use_t": True,
        "infer_use_future": True,
        "horizon": 24,
        "norm": True,
    },
    "sanjiaozhou": {
        "seq_len": 96,
        "d_model": 256,
        "d_ff": 1024,
        "n_heads": 8,
        "patch_len": 16,
        "stride": 8,
        "factor": 1,
        "e_layers": 1,
        "dropout": 0.1,
        "lr": 0.01,
        "num_epochs": 100,
        "patience": 10,
        "loss": "MSE",
        "lradj": "type3",
        "batch_size": 256,
        "activation": "gelu",
        "dbloss_alpha": 0.25,
        "dbloss_beta": 0.6,
        "alpha": 0.25,
        "beta": 0.12,
        "use_c_exog": True,
        "use_t_exog": True,
        "use_c": True,
        "use_t": True,
        "infer_use_future": True,
        "horizon": 24,
        "norm": True,
    },
    "laodaohe": {
        "seq_len": 96,
        "d_model": 512,
        "d_ff": 2048,
        "n_heads": 8,
        "patch_len": 16,
        "stride": 8,
        "factor": 1,
        "e_layers": 1,
        "dropout": 0.1,
        "lr": 0.01,
        "num_epochs": 100,
        "patience": 10,
        "loss": "MSE",
        "lradj": "type3",
        "batch_size": 256,
        "activation": "gelu",
        "dbloss_alpha": 0.2,
        "dbloss_beta": 0.5,
        "alpha": 0.2,
        "beta": 0.1,
        "use_c_exog": True,
        "use_t_exog": True,
        "use_c": True,
        "use_t": True,
        "infer_use_future": True,
        "horizon": 24,
        "norm": True,
    },
}

for dataset_name in DATASETS:
    model_hyper_params = DATASET_CONFIGS[dataset_name]

    save_path = f"{dataset_name}/{MODEL_DIR}"
    if TP_ONLY_MODE:
        save_path += "_TP"

    strategy_args = {
        "horizon": 24,
        "target_channel": [TP_CHANNEL if TP_ONLY_MODE else DEFAULT_TARGET_CHANNEL],
    }

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

    subprocess.run(args, cwd=os.path.dirname(__file__) if __file__ else os.getcwd())
