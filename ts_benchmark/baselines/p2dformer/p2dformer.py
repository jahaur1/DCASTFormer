import torch
import torch.nn as nn
from torch import optim

from ts_benchmark.baselines.p2dformer.models.p2dformer_model import Model as P2DFormer_model
from ts_benchmark.baselines.p2dformer.models.p2dformer_model import ExogenousEmbedding
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase


MODEL_HYPER_PARAMS = {
    "seq_len": 96, "horizon": 24, "pred_len": 24,
    "d_model": 128, "d_ff": 256, "n_heads": 4,
    "e_layers": 2, "dropout": 0.1,
    "patch_len": 16, "top_k": 3,
    "use_future_exog": True, "use_history_exog": True,
    "num_epochs": 100, "patience": 10, "batch_size": 64,
    "lr": 0.001, "lradj": "type1", "loss": "MSE",
}


class P2DFormer(DeepForecastingModelBase):
    """
    P2D-Former: Period-aware 2D Transformer for time series forecasting.
    """

    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "P2DFormer"

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
        # Create model with default values
        if not hasattr(self.config, "output_dim"):
            self.config.output_dim = 1
        if not hasattr(self.config, "input_dim"):
            self.config.input_dim = 1
        if not hasattr(self.config, "c_out"):
            self.config.c_out = self.config.enc_in
        return P2DFormer_model(self.config)

    def _init_optimizer(self, CovariateFusion=None):
        # enc_in has been set by the framework, initialize exog_dim
        enc_in = getattr(self.config, "enc_in", 1)
        series_dim = getattr(self.config, "series_dim", enc_in)
        exog_dim = max(enc_in - series_dim, 0)

        # Update model's exog_dim
        if exog_dim > 0:
            self.model.exog_dim = exog_dim
            # Re-initialize exog_embedding if needed
            if self.model.exog_embedding is None and self.model.use_future_exog:
                self.model.exog_embedding = ExogenousEmbedding(
                    pred_len=self.model.pred_len,
                    exog_dim=exog_dim,
                    d_model=self.model.d_model,
                    d_ff=self.model.d_ff
                ).to(next(self.model.parameters()).device)

        # Create optimizer
        if CovariateFusion is not None:
            optimizer = optim.Adam([
                {"params": self.model.parameters(), "lr": self.config.lr},
                {"params": CovariateFusion.parameters(), "lr": self.config.lr},
            ])
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        # P2D-Former uses exog_future as x_mark_dec for future exogenous
        output = self.model(input, x_mark_enc=input_mark, x_mark_dec=exog_future)
        return {"output": output}
