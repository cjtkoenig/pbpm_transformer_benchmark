"""
Specialised LSTM for extended attribute mode (next-activity task)

Source inspiration:
- Wickramanayake et al. (2022) Specialised LSTM
- https://github.com/ZhipengHe/Shared-and-Specialised-Attention-based-Interpretable-Models

What changed vs the original pasted code?
- Removed all training loops, plotting, metrics, and file I/O that were embedded in the original script.
- This module now ONLY defines the Keras architecture function get_next_activity_model.
- All data shaping (activities/resources/delta_t), padding, and vocabulary usage happen in the adapter
  and loader to preserve the repository-wide canonical data pipeline and comparability.

Why this design?
- Benchmark-wide rule: models should be pure architectures; adapters handle inputs/outputs.
- Keeps minimal-mode models untouched and avoids duplicate preprocessing logic in each model file.
- Makes cross-validation, metrics, and logging uniform across models.

How to use this model now
- Do NOT call this directly in scripts. Use the registry and adapter via the CLI:
  uv run python -m src.cli task=next_activity model.name=specialised_lstm \
    data.attribute_mode=extended data.datasets="[Helpdesk]"
- The registry builds the Keras model from here and the SpecialisedLSTMAdapter provides input tensors:
  [ac_input, rl_input, t_input] aligned along time, with shapes:
    ac_input: [B, T] int32 activity tokens
    rl_input: [B, T] int32 resource tokens
    t_input:  [B, T, 1] float32 delta hours since case start

Minimal vs Extended
- Minimal (activities only): use model.name=activity_only_lstm
- Extended (activities, resource/group, delta time): use model.name=specialised_lstm (this module) or shared_lstm

Implementation notes
- The model applies timestep-attention (alpha) and specialised feature-attention (beta) with separate
  streams for activity, resource, and time. Context is a weighted sum over time and features,
  followed by a dense softmax head over next-activity classes.
- Data preparation lives in src/data/loader.py (prepare_next_activity_extended_data).
- Model selection/creation lives in src/models/model_registry.py (specialised_lstm_factory) and
  SpecialisedLSTMAdapter, which takes DataFrames and outputs tensors for Keras .fit().
"""
from tensorflow.keras import layers, backend, Model
from tensorflow.keras.regularizers import l2


# Canonical API-compatible builder used by the model registry

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
    """Create the Specialised LSTM model for next-activity (extended attribute mode).

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

    # Time feature already shaped as [B, T, 1]
    time_embs = t_input

    dim_ac = int(embed_dim_ac)
    dim_rl = int(embed_dim_rl)
    dim_t = 1

    # Specialised beta attention streams
    beta_ac_bi = layers.Bidirectional(layers.LSTM(lstm_size_beta, return_sequences=True), name="beta_ac")
    beta_rl_bi = layers.Bidirectional(layers.LSTM(lstm_size_beta, return_sequences=True), name="beta_rl")
    beta_t_bi = layers.Bidirectional(layers.LSTM(lstm_size_beta, return_sequences=True), name="beta_t")

    beta_dense_ac = layers.TimeDistributed(layers.Dense(dim_ac, activation="tanh", kernel_regularizer=l2(l2reg)), name="feature_attention_ac")
    beta_dense_rl = layers.TimeDistributed(layers.Dense(dim_rl, activation="tanh", kernel_regularizer=l2(l2reg)), name="feature_attention_rl")
    beta_dense_t = layers.TimeDistributed(layers.Dense(dim_t, activation="tanh", kernel_regularizer=l2(l2reg)), name="feature_attention_t")

    # Compute feature-level importances and weighted representations
    beta_out_ac = beta_dense_ac(beta_ac_bi(ac_embs))
    c_t_ac = layers.Multiply(name="ac_importance")([beta_out_ac, ac_embs])

    beta_out_rl = beta_dense_rl(beta_rl_bi(rl_embs))
    c_t_rl = layers.Multiply(name="rl_importance")([beta_out_rl, rl_embs])

    beta_out_t = beta_dense_t(beta_t_bi(time_embs))
    c_t_t = layers.Multiply(name="t_importance")([beta_out_t, time_embs])

    # Concatenate specialised contexts
    c_t = layers.Concatenate(name="concat")([c_t_ac, c_t_rl, c_t_t])
    c_t = layers.Dropout(dropout_input, name="input_dropout")(c_t)

    # Global timestep attention (alpha)
    alpha_bi = layers.Bidirectional(layers.LSTM(lstm_size_alpha, return_sequences=True), name="alpha")
    alpha_dense = layers.TimeDistributed(layers.Dense(1, kernel_regularizer=l2(l2reg)), name="alpha_dense")

    a = alpha_bi(c_t)
    e = alpha_dense(a)
    alpha = layers.Softmax(axis=1, name="timestep_attention")(e)  # [B, T, 1]

    # Context: sum over time of alpha_t * c_t
    weighted = layers.Multiply()([alpha, c_t])
    context = layers.Lambda(lambda x: backend.sum(x, axis=1), name="context_sum")(weighted)
    context = layers.Dropout(dropout_context, name="context_dropout")(context)

    # Prediction head
    out = layers.Dense(output_dim, activation="softmax", kernel_initializer="glorot_uniform", name="act_output")(context)

    model = Model(inputs=[ac_input, rl_input, t_input], outputs=out, name="specialised_lstm_next_activity")
    return model
