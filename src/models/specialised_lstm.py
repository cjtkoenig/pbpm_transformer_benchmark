"""
Specialised LSTM with attention (activities-only variant).

This implementation adapts the "specialised" attention LSTM from:
- https://github.com/ZhipengHe/Shared-and-Specialised-Attention-based-Interpretable-Models

Key alignment decisions for this benchmark (canonical methodology):
- Inputs: integer activity tokens of shape [B, T]. No roles/per-step time streams yet.
- We keep the temporal order (no reverse-time processing) to match the benchmark.
- Adapters in this repo are content-preserving; therefore, we do not fabricate
  role tokens or per-step time arrays here. Those will arrive later via an
  optional extended attribute mode.
- We mirror the original model's structure as closely as possible for a single
  stream: an embedding, per-stream "beta" attention is degenerated to a single
  stream gating, and an "alpha" timestep attention is applied over the gated
  sequence to build a context vector, which is then fed to a softmax head.

for future extended mode:
- The attention blocks are written in a way that we can plug in multi-stream
  inputs later (activities, roles, per-step time) without changing canonical
  behavior. The exported function name remains focused on next-activity.

This model outputs class probabilities (softmax), and should be compiled
with sparse_categorical_crossentropy by the task runner.
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class TimestepAttention(layers.Layer):
    """Implements the "alpha" timestep attention over a sequence of hidden states.

    Given h_t (from an LSTM/BiLSTM), produce scalar e_t per timestep, then
    alpha_t = softmax(e_t) and a context vector c = sum_t alpha_t * h_t.

    We closely follow the original code idea (alpha LSTM + dense + softmax),
    but encapsulate the scorer as a layer.
    """
    def __init__(self, hidden_alpha: int = 50, l2reg: float = 1e-4, name: str = "timestep_attention", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha_lstm = layers.Bidirectional(
            layers.LSTM(hidden_alpha, return_sequences=True), name="alpha"
        )
        self.alpha_dense = layers.TimeDistributed(
            layers.Dense(1, kernel_regularizer=regularizers.l2(l2reg)), name="alpha_dense"
        )
        self.softmax_time = layers.Softmax(axis=1)

    def call(self, h: tf.Tensor, training=None) -> Tuple[tf.Tensor, tf.Tensor]:
        # h: [B, T, H]
        a = self.alpha_lstm(h, training=training)            # [B, T, H']
        e = self.alpha_dense(a, training=training)           # [B, T, 1]
        alpha = self.softmax_time(e)                         # [B, T, 1]
        context = tf.reduce_sum(alpha * h, axis=1)           # [B, H]
        return context, alpha


class FeatureGate(layers.Layer):
    """Implements a single-stream variant of the original "beta" feature attention.

    In the specialised model, the original repository uses separate beta LSTMs
    per modality (activities, roles, time) and then multiplies the resulting
    per-feature weights with each modality stream before concatenation.

    Here we only have activities (a single stream). We therefore apply a single
    beta LSTM + tanh projection to produce per-timestep, per-feature gates that
    modulate the activity embeddings.
    """
    def __init__(self, hidden_beta: int = 50, l2reg: float = 1e-4, name: str = "feature_attention", **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta_lstm = layers.Bidirectional(
            layers.LSTM(hidden_beta, return_sequences=True), name="beta_ac"
        )
        # units is set dynamically on build to match input feature dim
        self.beta_dense = layers.TimeDistributed(
            layers.Dense(units=None, activation="tanh", kernel_regularizer=regularizers.l2(l2reg)),
            name="feature_attention_ac",
        )

    def build(self, input_shape):
        # input_shape: (B, T, D)
        d = int(input_shape[-1])
        # Set dense units to D so beta has same feature dimension as embeddings
        self.beta_dense.layer.units = d
        super().build(input_shape)

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        # x: [B, T, D]
        b = self.beta_lstm(x, training=training)             # [B, T, H']
        beta = self.beta_dense(b, training=training)         # [B, T, D]
        # Multiply per-feature weights with the embeddings (gating)
        return layers.Multiply(name="ac_importance")([beta, x])  # [B, T, D]


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
    """Create the activities-only specialised LSTM model for next-activity.

    Parameters (kept close to original defaults where meaningful):
    - embed_dim: size of the activity embedding (replaces "ac_weights" in the original)
    - lstm_size_alpha/beta: hidden sizes for alpha/beta BiLSTMs
    - dropout_input/context: dropouts mirroring original hyperparameters
    - attribute_mode: reserved for future "extended" mode; currently unused here

    Returns: tf.keras.Model with inputs [ac_tokens] and softmax outputs over output_dim classes.
    """
    # Input: integer activity tokens [B, T]
    ac_input = layers.Input(shape=(max_case_length,), dtype="int32", name="ac_input")

    # Activity embedding (trainable). In the original code, they could initialize
    # with precomputed weights; here we keep a trainable embedding layer only.
    ac_embs = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="ac_embedding",
    )(ac_input)  # [B, T, D]

    x = layers.Dropout(dropout_input, name="dropout_input")(ac_embs)

    # Specialised per-stream feature gating (single stream here)
    gated = FeatureGate(hidden_beta=lstm_size_beta, l2reg=l2reg, name="feature_attention")(x)  # [B, T, D]

    # Timestep attention (alpha) over the gated sequence
    context, alpha = TimestepAttention(hidden_alpha=lstm_size_alpha, l2reg=l2reg)(gated)

    # Context dropout (similar to original dropout_context)
    context = layers.Dropout(dropout_context, name="context_dropout")(context)

    # Classification head: softmax over next_activity classes
    act_output = layers.Dense(
        output_dim,
        activation="softmax",
        kernel_initializer="glorot_uniform",
        name="act_output",
    )(context)

    model = Model(inputs=[ac_input], outputs=act_output, name="specialised_lstm_next_activity")
    return model
