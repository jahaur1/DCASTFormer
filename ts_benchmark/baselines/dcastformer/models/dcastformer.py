"""
DCASTFormer: Dual-Channel Adaptive Spatio-Temporal Transformer
"""
import torch
import torch.nn as nn

from ts_benchmark.baselines.dcastformer.layers.Embed import PatchEmbed
from ts_benchmark.baselines.dcastformer.layers.SelfAttention_Family import TSMixer, ResAttention
from ts_benchmark.baselines.dcastformer.layers.Transformer_EncDec import TSEncoder, IntAttention, CointAttention


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin
        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.horizon
        self.d_model = configs.d_model

        self.use_future_exog = getattr(configs, "use_future_exog", True)
        self.use_history_exog = getattr(configs, "use_history_exog", True)
        self.configs = configs
        self.attn_mode = getattr(configs, "attn_mode", "full")
        self.layer_order = getattr(configs, "layer_order", "int_coint")
        self.infer_use_future = getattr(configs, "infer_use_future", False)

        # Patch setting
        assert self.seq_len % self.period == 0
        self.num_p = self.seq_len // self.period

        if getattr(configs, "num_p", None) is None:
            configs.num_p = self.num_p

        self.patch_embed = PatchEmbed(configs, num_p=self.num_p)

        # Encoder
        layers = self.layers_init(configs)
        if self.attn_mode == "none":
            self.encoder = nn.Identity()
        else:
            self.encoder = TSEncoder(layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_p * configs.d_model, self.pred_len, bias=False),
        )

        # Fusion components (lazy creation to ensure exog_dim is correct)
        self.future_gate = None
        self.future_exog_proj = None
        self.future_exog_gate = None
        self.fusion_alpha_logit = None
        self.exog_dim = 0
        self._fusion_initialized = False

    def init_fusion_components(self, exog_dim, alpha_init=0.0):
        """Called in _init_optimizer when enc_in is already determined."""
        if self._fusion_initialized or exog_dim <= 0:
            self._fusion_initialized = True
            return

        device = next(self.parameters()).device

        # gated_overwrite branch
        self.future_gate = nn.Sequential(
            nn.Linear(exog_dim * 2, exog_dim),
            nn.Sigmoid(),
        ).to(device)

        # embedding_concat branch
        self.future_exog_proj = nn.Sequential(
            nn.Linear(exog_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        ).to(device)
        self.future_exog_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid(),
        ).to(device)

        # Learnable fusion weight
        self.fusion_alpha_logit = nn.Parameter(torch.tensor(float(alpha_init), device=device))

        self.exog_dim = exog_dim
        self._fusion_initialized = True

    def build_temporal_encoder(self, configs):
        return IntAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff,
            dropout=configs.dropout, stable_len=configs.stable_len,
            activation=configs.activation, stable=False, enc_in=self.c_in,
        )

    def build_covariate_encoder(self, configs):
        return CointAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff,
            dropout=configs.dropout, activation=configs.activation,
            stable=False, enc_in=self.c_in, stable_len=configs.stable_len,
        )

    def layers_init(self, configs):
        layers = []
        ia_layers = getattr(configs, "ia_layers", 1)
        ca_layers = getattr(configs, "ca_layers", 1)

        if self.attn_mode == "none":
            return layers
        if self.attn_mode == "only_int":
            for _ in range(ia_layers):
                layers.append(self.build_temporal_encoder(configs))
            return layers
        if self.attn_mode == "only_coint":
            for _ in range(ca_layers):
                layers.append(self.build_covariate_encoder(configs))
            return layers
        if self.attn_mode == "full":
            if self.layer_order == "int_coint":
                for _ in range(ia_layers):
                    layers.append(self.build_temporal_encoder(configs))
                for _ in range(ca_layers):
                    layers.append(self.build_covariate_encoder(configs))
            elif self.layer_order == "coint_int":
                for _ in range(ca_layers):
                    layers.append(self.build_covariate_encoder(configs))
                for _ in range(ia_layers):
                    layers.append(self.build_temporal_encoder(configs))
            elif self.layer_order == "interleave":
                n = max(ia_layers, ca_layers)
                for i in range(n):
                    if i < ia_layers:
                        layers.append(self.build_temporal_encoder(configs))
                    if i < ca_layers:
                        layers.append(self.build_covariate_encoder(configs))
            return layers
        return layers

    def _remove_history_exog(self, x_enc):
        if self.use_history_exog:
            return x_enc
        series_dim = getattr(self.configs, "series_dim", None)
        if series_dim is None:
            series_dim = getattr(self.configs, "enc_in", 1)
        x_new = x_enc.clone()
        if x_new.shape[-1] > series_dim:
            x_new[:, :, series_dim:] = 0.0
        return x_new

    def _path1_gated_overwrite(self, x_enc, exog_future):
        """Branch 1: gate * future_exog + (1-gate) * history_exog_tail"""
        if exog_future is None or self.future_gate is None:
            return x_enc

        future_len = min(exog_future.shape[1], self.pred_len, self.seq_len)
        if future_len <= 0:
            return x_enc

        history_tail = x_enc[:, -future_len:, :].clone()
        use_dim = min(exog_future.shape[-1], history_tail.shape[-1])

        if use_dim <= 0:
            return x_enc

        hist_exog = history_tail[:, :, -use_dim:]
        fut_exog = exog_future[:, -future_len:, :use_dim]

        gate = self.future_gate(torch.cat([hist_exog, fut_exog], dim=-1))
        fused = gate * fut_exog + (1 - gate) * hist_exog

        history_tail_fixed = torch.cat([
            history_tail[:, :, :history_tail.shape[-1] - use_dim],
            fused
        ], dim=-1)

        x_enc = torch.cat([x_enc, history_tail_fixed], dim=1)
        x_enc = x_enc[:, -self.seq_len:, :]
        return x_enc

    def _path2_embedding_enhance(self, x_enc_emb, exog_future):
        """Branch 2: future_exog -> embedding, fusion after PatchEmbed"""
        if self.future_exog_proj is None or self.future_exog_gate is None:
            return x_enc_emb

        d_model = x_enc_emb.shape[-1]

        fut_emb = self.future_exog_proj(exog_future)
        fut_emb_mean = fut_emb.mean(dim=1)

        fut_exp = fut_emb_mean
        for _ in range(x_enc_emb.ndim - 2):
            fut_exp = fut_exp.unsqueeze(1)
        fut_exp = fut_exp.expand_as(x_enc_emb)

        gate = self.future_exog_gate(torch.cat([x_enc_emb, fut_exp], dim=-1))
        x_enc_emb = x_enc_emb + gate * fut_exp

        return x_enc_emb

    def forecast(self, x_enc, x_mark_enc=None, exog_future=None):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(
                (*x_enc.shape[:-1], 4),
                device=x_enc.device, dtype=x_enc.dtype,
            )

        original_c_in = x_enc.shape[-1]

        # 1. history exog ablation
        x_enc = self._remove_history_exog(x_enc)

        # 2. Instance normalization
        if self.revin:
            mean = x_enc.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc_norm = (x_enc - mean) / std
        else:
            mean = std = None
            x_enc_norm = x_enc

        # 3. Dual-branch fusion: gated_overwrite + embedding_concat
        x_enc_gated = self._path1_gated_overwrite(x_enc_norm.clone(), exog_future)
        x_enc_gated_emb = self.patch_embed(x_enc_gated, x_mark_enc)

        x_enc_base_emb = self.patch_embed(x_enc_norm, x_mark_enc)
        if exog_future is not None and self.exog_dim > 0:
            x_enc_concat_emb = self._path2_embedding_enhance(x_enc_base_emb, exog_future)
        else:
            x_enc_concat_emb = x_enc_base_emb

        if self.fusion_alpha_logit is not None:
            alpha = torch.sigmoid(self.fusion_alpha_logit)
            x_enc_emb = alpha * x_enc_gated_emb + (1 - alpha) * x_enc_concat_emb
        else:
            x_enc_emb = x_enc_gated_emb

        # 6. Encoder
        if self.attn_mode == "none":
            enc_out = x_enc_emb
        else:
            enc_out = self.encoder(x_enc_emb)[0]

        enc_out = enc_out[:, :self.c_in, ...]

        # 7. Decoder
        dec_out = self.decoder(enc_out).transpose(-1, -2)

        # 8. De-normalization
        if self.revin:
            dec_out = dec_out * std + mean

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, exog_future=None):
        dec_out = self.forecast(x_enc, x_mark_enc, exog_future)
        return dec_out[:, -self.pred_len:, :]

    def get_alpha(self):
        if self.fusion_alpha_logit is not None:
            return torch.sigmoid(self.fusion_alpha_logit).item()
        return None
