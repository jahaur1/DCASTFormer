import torch
import torch.nn as nn
from torch import optim

from ts_benchmark.baselines.dcastformer.models.dcastformer import Model as DCASTformer_model
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase


MODEL_HYPER_PARAMS = {
    "seq_len": 96, "horizon": 24, "period": 24,
    "d_model": 128, "d_ff": 128, "n_heads": 4,
    "dropout": 0, "attn_dropout": 0.15, "activation": "gelu",
    "stable_len": 2, "revin": 1,
    "ia_layers": 1, "ca_layers": 1,
    "attn_mode": "full", "layer_order": "int_coint",
    "use_future_exog": True, "use_history_exog": True,
    "infer_use_future": True,
    "num_epochs": 100, "patience": 10, "batch_size": 64,
    "lr": 0.001, "lradj": "type1", "loss": "MSE",
}


class DCASTformer(DeepForecastingModelBase):
    """
    DCASTformer: Dual-Channel Adaptive Spatio-Temporal Transformer.

    修复 TB_DualExogFusion 的 fusion 参数未训练问题：
    - 融合组件在 _init_optimizer 中创建（此时 enc_in 已确定）
    - optimizer 包含 fusion 参数，确保被训练
    """

    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "DCASTformer"

    def _adjust_lr(self, optimizer, epoch, config):
        if config.lradj == "type1":
            lr_adjust = {epoch: config.lr * (0.5 ** ((epoch - 1) // 1))}
        elif config.lradj == "type3":
            lr_adjust = {epoch: config.lr if epoch < 3 else config.lr * (0.9 ** ((epoch - 3) // 1))}
        else:
            lr_adjust = {}
        if epoch in lr_adjust:
            for pg in optimizer.param_groups:
                pg["lr"] = lr_adjust[epoch]

    def _init_criterion(self):
        return nn.MSELoss() if self.config.loss == "MSE" else nn.L1Loss()

    def _init_model(self):
        # 用默认值创建模型
        if not hasattr(self.config, "output_dim"):
            self.config.output_dim = 1
        if not hasattr(self.config, "input_dim"):
            self.config.input_dim = 1
        return DCASTformer_model(self.config)

    def _init_optimizer(self, CovariateFusion=None):
        # 此时 enc_in 已被框架设置，初始化融合组件
        enc_in = getattr(self.config, "enc_in", 1)
        series_dim = getattr(self.config, "series_dim", enc_in)
        exog_dim = max(enc_in - series_dim, 0)

        # 更新模型的 c_in 和 exog_dim
        self.model.c_in = enc_in
        self.model.exog_dim = exog_dim

        # 初始化融合组件（如果需要）
        if exog_dim > 0 and not self.model._fusion_initialized:
            alpha_init = getattr(self.config, "alpha_init", 0.0)
            self.model.init_fusion_components(exog_dim, alpha_init)

        # 创建 optimizer（包含 fusion 参数）
        if CovariateFusion is not None:
            optimizer = optim.Adam([
                {"params": self.model.parameters(), "lr": self.config.lr},
                {"params": CovariateFusion.parameters(), "lr": self.config.lr},
            ])
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        output = self.model(input, x_mark_enc=input_mark, exog_future=exog_future)
        return {"output": output}
