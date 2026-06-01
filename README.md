# DCASTFormer

DCASTFormer is a dynamic covariate-aware soft-sensing forecasting model designed for difficult-to-measure target prediction with easy-to-measure covariates.

In many industrial and environmental monitoring scenarios, the target variable is difficult to obtain continuously or in real time, while several auxiliary variables are easier to measure and may be available over the future prediction horizon. DCASTFormer is developed for this type of forecasting task, where historical target observations, historical covariates, and future available covariates need to be modeled jointly.

## Overview

DCASTFormer introduces a dynamic covariate-aware forecasting framework. Instead of simply concatenating future covariates with historical inputs, the model uses a dual-path fusion mechanism to incorporate future covariate information in a more structured way.

The model contains the following main components:

- Patch-based temporal representation
- Dual-path future covariate fusion
- Adaptive covariate fusion weighting
- Temporal relation encoding
- Covariate interaction encoding
- Linear prediction decoder

The overall objective is to improve soft-sensing prediction by capturing both historical dynamics and future covariate guidance.

## Model Motivation

In practical soft-sensing tasks, the target variable may be difficult to measure due to high cost, long sampling interval, complex laboratory analysis, or sensor limitations. However, some covariates can be measured more easily and may be available in advance during forecasting.

A direct way to use these future covariates is to concatenate them with the input sequence or overwrite the historical covariate channels. However, these simple strategies may suffer from several limitations:

- direct concatenation may introduce redundant channels;
- hard overwriting may discard useful historical covariate information;
- independent covariate encoding may not sufficiently interact with the target representation;
- fixed fusion strategies may not adapt well to different samples or monitoring sites.

DCASTFormer addresses these problems by using a dual-path adaptive fusion structure. One path injects future covariates into the historical covariate space, while the other path enhances the latent representation using future covariate embeddings. The two paths are then combined through a learnable adaptive fusion coefficient.

## Architecture

The model takes three types of information as input:

```text
Historical input sequence:
    x_enc: [B, seq_len, enc_in]

Historical time markers:
    x_mark_enc: [B, seq_len, time_dim]

Future available covariates:
    exog_future: [B, pred_len, exog_dim]
```

The output is the predicted target sequence:

```text
Prediction:
    y_pred: [B, pred_len, target_dim]
```

The overall data flow is:

```text
Input sequence
    │
    ├── Reversible normalization
    │
    ├── Dual-path future covariate fusion
    │       ├── Path 1: gated covariate overwrite
    │       └── Path 2: embedding-level covariate enhancement
    │
    ├── Adaptive path fusion
    │
    ├── Patch representation
    │
    ├── Temporal relation encoder
    │
    ├── Covariate interaction encoder
    │
    ├── Prediction decoder
    │
    └── Output forecast
```

## Data Flow

A simplified data flow is shown below:

```text
x_enc: [B, seq_len, enc_in]
x_mark_enc: [B, seq_len, time_dim]
exog_future: [B, pred_len, exog_dim]

    │
    ├── Normalization
    │
    ├── Path 1:
    │       history_exog + future_exog
    │       → gated covariate overwrite
    │       → patch embedding
    │
    ├── Path 2:
    │       historical input
    │       → patch embedding
    │       → future covariate projection
    │       → embedding-level enhancement
    │
    ├── Adaptive fusion:
    │       α * path_1 + (1 - α) * path_2
    │
    ├── Temporal relation encoder
    │
    ├── Covariate interaction encoder
    │
    ├── Flatten + Linear decoder
    │
    ├── De-normalization
    │
    └── y_pred: [B, pred_len, target_dim]
```

## Main Features

* Supports future available covariates.
* Uses dual-path covariate fusion instead of simple concatenation.
* Preserves both historical covariate information and future covariate guidance.
* Learns an adaptive balance between two covariate fusion paths.
* Uses patch-based temporal representation for efficient sequence modeling.
* Models both temporal relations and covariate interactions.
* Supports reversible normalization for non-stationary time series.

## Usage

Run the model with:

```bash
python Run_DCASTformer.py
```

Alternatively, the benchmark script can be executed through:

```bash
bash scripts/covariate_forecasting/DCASTformer.sh
```

## Requirements

* Python 3.8+
* PyTorch
* einops
* numpy
* pandas
* scikit-learn

Install the required packages according to your local environment.

Example:

```bash
pip install torch einops numpy pandas scikit-learn
```

## Project Structure

```text
DCASTformer/
├── README.md
├── Run_DCASTformer.py
├── config/
├── dataset/
│   └── forecasting/
├── scripts/
│   ├── run_benchmark.py
│   └── covariate_forecasting/
│       └── DCASTformer.sh
└── ts_benchmark/
    ├── baselines/
    │   ├── dcastformer/
    │   │   ├── __init__.py
    │   │   ├── dcastformer.py
    │   │   ├── layers/
    │   │   │   ├── Embed.py
    │   │   │   ├── SelfAttention_Family.py
    │   │   │   └── Transformer_EncDec.py
    │   │   └── models/
    │   │       └── dcastformer.py
    │   └── deep_forecasting_model_base.py
    ├── data/
    ├── evaluation/
    ├── models/
    ├── report/
    ├── utils/
    ├── common/
    └── pipeline.py
```

## Input Format

The historical input sequence should be organized as:

```text
x_enc: [batch_size, seq_len, enc_in]
```

The future covariates should be organized as:

```text
exog_future: [batch_size, pred_len, exog_dim]
```

The time marker input is optional. If it is not provided, the model will create a zero-valued time marker internally.

```text
x_mark_enc: [batch_size, seq_len, time_dim]
```
