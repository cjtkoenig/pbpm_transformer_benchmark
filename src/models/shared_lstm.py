"""
Shared LSTM for extended attribute mode (next-activity task)

Source inspiration:
- Wickramanayake et al. (2022) Shared & Specialised LSTM
- https://github.com/ZhipengHe/Shared-and-Specialised-Attention-based-Interpretable-Models

What changed vs the original pasted code?
- We removed training loops, plotting, metrics, and file I/O from the model file.
- This module now ONLY defines the Keras architecture function get_next_activity_model.
- All data shaping (activities/resources/delta_t), padding, and vocabulary usage happen in the adapter
  and loader to preserve the repository-wide canonical data pipeline and comparability.

Why this design?
- Benchmark-wide rule: models should be pure architectures; adapters handle inputs/outputs.
- Keeps minimal-mode models untouched and avoids duplicate preprocessing logic in each model file.
- Makes cross-validation, metrics, and logging uniform across models.

How to use this model now
- Do NOT call this directly in scripts. Use the registry and adapter via the CLI:
  uv run python -m src.cli task=next_activity model.name=shared_lstm \
    data.attribute_mode=extended data.datasets="[Helpdesk]"
- The registry builds the Keras model from here and the SharedLSTMAdapter provides input tensors:
  [ac_input, rl_input, t_input] aligned along time, with shapes:
    ac_input: [B, T] int32 activity tokens
    rl_input: [B, T] int32 resource tokens
    t_input:  [B, T, 1] float32 delta hours since case start

Minimal vs Extended
- Minimal (activities only): use model.name=activity_only_lstm
- Extended (activities, resource/group, delta time): use model.name=shared_lstm (this module)

Implementation notes
- The model applies timestep-attention (alpha) and feature-attention (beta) over the concatenated
  streams (activity embedding, resource embedding, delta_t). Context is a weighted sum over time and
  features, followed by a dense softmax head over next-activity classes.
- Data preparation moved to src/data/loader.py (prepare_next_activity_extended_data).
- Model selection/creation moved to src/models/model_registry.py (shared_lstm_factory) and
  SharedLSTMAdapter, which takes DataFrames and outputs tensors for Keras .fit().
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend, Model, callbacks
from tensorflow.keras.regularizers import l2


# New canonical API-compatible builder

def get_next_activity_model(
    max_case_length: int,
    ac_vocab_size: int,
    rl_vocab_size: int,
    output_dim: int,
    embed_dim_ac: int = 64,
    embed_dim_rl: int = 64,
    lstm_size_alpha: int = 50,
    lstm_size_beta: int = 50,
    dropout_input: float = 0.15,
    dropout_context: float = 0.15,
    l2reg: float = 1e-4,
) -> Model:
    """Create the Shared LSTM model for next-activity (extended attribute mode).

    Inputs:
      - ac_input: [B, T] integer activity tokens
      - rl_input: [B, T] integer resource tokens
      - t_input:  [B, T, 1] float delta time since case start (hours)
    Output:
      - softmax over next activity classes (output_dim)
    """
    # Inputs
    ac_input = layers.Input(shape=(max_case_length,), dtype="int32", name="ac_input")
    rl_input = layers.Input(shape=(max_case_length,), dtype="int32", name="rl_input")
    t_input = layers.Input(shape=(max_case_length, 1), dtype="float32", name="t_input")

    # Embeddings
    ac_embs = layers.Embedding(input_dim=ac_vocab_size, output_dim=embed_dim_ac, name="ac_embedding")(ac_input)
    rl_embs = layers.Embedding(input_dim=rl_vocab_size, output_dim=embed_dim_rl, name="rl_embedding")(rl_input)

    # Concatenate streams: [B, T, D_ac + D_rl + 1]
    full_embs = layers.Concatenate(name="full_embs")( [ac_embs, rl_embs, t_input] )
    full_embs = layers.Dropout(dropout_input, name="input_dropout")(full_embs)

    dim_total = int(embed_dim_ac + embed_dim_rl + 1)

    # Attention LSTMs
    alpha_bi = layers.Bidirectional(layers.LSTM(lstm_size_alpha, return_sequences=True), name="alpha")
    beta_bi = layers.Bidirectional(layers.LSTM(lstm_size_beta, return_sequences=True), name="beta")

    # Dense layers for attentions
    alpha_dense = layers.TimeDistributed(layers.Dense(1, kernel_regularizer=l2(l2reg)), name="alpha_dense")
    beta_dense = layers.TimeDistributed(layers.Dense(dim_total, activation="tanh", kernel_regularizer=l2(l2reg)), name="feature_attention")

    # Compute attentions
    a = alpha_bi(full_embs)
    e = alpha_dense(a)
    alpha = layers.Softmax(axis=1, name="timestep_attention")(e)  # [B, T, 1]

    b = beta_bi(full_embs)
    beta = beta_dense(b)  # [B, T, D]

    # Context: sum over time of alpha_t * beta_t * full_embs_t
    c_t = layers.Multiply(name="feature_importance")([alpha, beta, full_embs])
    context = layers.Lambda(lambda x: backend.sum(x, axis=1), name="context_sum")(c_t)  # [B, D]
    context = layers.Dropout(dropout_context, name="context_dropout")(context)

    # Prediction head
    out = layers.Dense(output_dim, activation="softmax", kernel_initializer="glorot_uniform", name="act_output")(context)

    model = Model(inputs=[ac_input, rl_input, t_input], outputs=out, name="shared_lstm_next_activity")
    return model
