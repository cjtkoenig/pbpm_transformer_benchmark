"""
Wickramanayake et al. (2022) Activities-Only LSTM (Shared/Specialised equivalent when only activities are present).
https://github.com/ZhipengHe/Shared-and-Specialised-Attention-based-Interpretable-Models

Rationale:
- In the activities-only setup (attribute_mode == "minimal"), the Shared and Specialised
  LSTM models from Wickramanayake et al. (2022) reduce to the same mechanism:
  a single feature-attention (beta) and a timestep-attention (alpha) over the activity
  embedding sequence. This module factors out that common implementation.

Usage constraints:
- Supports only task='next_activity'.
- Designed for attribute_mode='minimal' (activities only). Extended mode is not implemented here (see specialised/shared_lstm).
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class TimestepAndFeatureAttention(layers.Layer):
    """Combined per-timestep (alpha) and per-feature (beta) attention over a single stream.

    For activities-only:
    - beta: BiLSTM -> TimeDistributed(Dense(D, tanh)) produces per-timestep feature gates.
    - alpha: BiLSTM -> TimeDistributed(Dense(1)) -> softmax over time produces timestep weights.
    - context: sum_t (alpha_t * beta_t * x_t).
    """
    def __init__(self, hidden_alpha: int = 50, hidden_beta: int = 50, l2reg: float = 1e-4, dropout: float = 0.0, name: str = "attn", **kwargs):
        super().__init__(name=name, **kwargs)
        # Beta branch (feature attention)
        self.beta_lstm = layers.Bidirectional(layers.LSTM(hidden_beta, return_sequences=True), name="beta")
        self.beta_dense = layers.TimeDistributed(
            layers.Dense(units=None, activation="tanh", kernel_regularizer=regularizers.l2(l2reg)),
            name="feature_attention",
        )
        # Alpha branch (timestep attention)
        self.alpha_lstm = layers.Bidirectional(layers.LSTM(hidden_alpha, return_sequences=True), name="alpha")
        self.alpha_dense = layers.TimeDistributed(
            layers.Dense(1, kernel_regularizer=regularizers.l2(l2reg)), name="alpha_dense"
        )
        self.softmax_time = layers.Softmax(axis=1)
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        # input_shape: (B, T, D)
        d = int(input_shape[-1])
        # Set dense units to D so beta has same feature dimension as embeddings
        self.beta_dense.layer.units = d
        super().build(input_shape)

    def call(self, x: tf.Tensor, training=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # x: [B, T, D]
        # Feature attention (beta) over the same input stream
        b = self.beta_lstm(x, training=training)             # [B, T, H']
        beta = self.beta_dense(b, training=training)         # [B, T, D]
        # Timestep attention (alpha) over the same input stream
        a = self.alpha_lstm(x, training=training)            # [B, T, H']
        e = self.alpha_dense(a, training=training)           # [B, T, 1]
        alpha = self.softmax_time(e)                         # [B, T, 1]
        # Optional dropout on input stream
        x = self.dropout(x, training=training)
        # Combine attentions and inputs
        c_t = alpha * beta * x                               # [B, T, D]
        context = tf.reduce_sum(c_t, axis=1)                 # [B, D]
        return context, alpha, beta


def get_next_activity_model(
    max_case_length: int,
    vocab_size: int,
    output_dim: int,
    embed_dim: int = 64,
    lstm_size_alpha: int = 50,
    lstm_size_beta: int = 50,
    dropout_input: float = 0.15,
    dropout_context: float = 0.15,
    l2reg: float = 1e-4,
    attribute_mode: str = "minimal",
) -> Model:
    """Create the activities-only Wickramanayake LSTM model for next-activity.

    Returns a tf.keras.Model with input [ac_tokens] and softmax outputs over output_dim.
    """
    if attribute_mode != "minimal":
        # Guard for activities-only version
        raise NotImplementedError("activity_only_lstm supports only attribute_mode='minimal' (activities only)")

    # Input: integer activity tokens [B, T]
    ac_input = layers.Input(shape=(max_case_length,), dtype="int32", name="ac_input")

    # Activity embedding
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="ac_embedding")(ac_input)  # [B, T, D]

    # Attention over the single stream
    context, alpha, beta = TimestepAndFeatureAttention(
        hidden_alpha=lstm_size_alpha,
        hidden_beta=lstm_size_beta,
        l2reg=l2reg,
        dropout=dropout_input,
    )(x)

    # Context dropout
    context = layers.Dropout(dropout_context, name="context_dropout")(context)

    # Prediction head
    out = layers.Dense(output_dim, activation="softmax", kernel_initializer="glorot_uniform", name="act_output")(context)

    model = Model(inputs=[ac_input], outputs=out, name="activity_only_lstm_next_activity")
    return model
