"""
Multi-task learning transformer model "MTLFormer"for process mining taken from 
https://github.com/Zaharah/processtransformer/blob/main/processtransformer/models/transformer.py
"""


import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_a = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_b = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_a = layers.Dropout(rate)
        self.dropout_b = layers.Dropout(rate)

    def call(self, inputs, training=None): # added training=None to match the signature of the parent class
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_a(attn_output, training=training)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, training=training)
        return self.layernorm_b(out_a + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def get_predict_model(max_case_length, vocab_size, output_dim, 
                     embed_dim=36, num_heads=4, ff_dim=64):
    """
    Create a true multi-task learning model with three outputs:
    - out1: Next activity prediction (classification)
    - out2: Next time prediction (regression) 
    - out3: Remaining time prediction (regression)
    
    This is the actual MTLFormer implementation from the main.py code.
    """
    # Input for activity sequences
    activity_inputs = layers.Input(shape=(max_case_length,), name="activity_input")
    
    # Input for time sequences (for time-based tasks)
    time_inputs = layers.Input(shape=(max_case_length,), name="time_input")
    
    # Input for time features (3 features for time tasks)
    time_features = layers.Input(shape=(3,), name="time_features")
    
    # Shared transformer backbone for activity sequences
    x_act = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(activity_inputs)
    x_act = TransformerBlock(embed_dim, num_heads, ff_dim)(x_act)
    x_act = layers.GlobalAveragePooling1D()(x_act)
    
    # Shared transformer backbone for time sequences
    x_time = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(time_inputs)
    x_time = TransformerBlock(embed_dim, num_heads, ff_dim)(x_time)
    x_time = layers.GlobalAveragePooling1D()(x_time)
    
    # Time features processing
    x_tf = layers.Dense(32, activation="relu")(time_features)
    
    # Task-specific outputs
    # Output 1: Next Activity (Classification)
    out1 = layers.Dropout(0.1)(x_act)
    out1 = layers.Dense(64, activation="relu")(out1)
    out1 = layers.Dropout(0.1)(out1)
    out1 = layers.Dense(output_dim, activation="linear", name="out1")(out1)
    
    # Output 2: Next Time (Regression)
    x_next_time = layers.Concatenate()([x_time, x_tf])
    out2 = layers.Dropout(0.1)(x_next_time)
    out2 = layers.Dense(128, activation="relu")(out2)
    out2 = layers.Dropout(0.1)(out2)
    out2 = layers.Dense(1, activation="linear", name="out2")(out2)
    
    # Output 3: Remaining Time (Regression)
    x_remaining_time = layers.Concatenate()([x_time, x_tf])
    out3 = layers.Dropout(0.1)(x_remaining_time)
    out3 = layers.Dense(128, activation="relu")(out3)
    out3 = layers.Dropout(0.1)(out3)
    out3 = layers.Dense(1, activation="linear", name="out3")(out3)
    
    # Create model with multiple outputs
    model = tf.keras.Model(
        inputs=[activity_inputs, time_inputs, time_features],
        outputs=[out1, out2, out3],
        name="mtlformer_multi_task"
    )
    
    return model

def get_next_activity_model(max_case_length, vocab_size, output_dim, 
    embed_dim = 36, num_heads = 4, ff_dim = 64):
    inputs = layers.Input(shape=(max_case_length,))
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    transformer = tf.keras.Model(inputs=inputs, outputs=outputs,
        name = "next_activity_transformer")
    return transformer

def get_next_time_model(max_case_length, vocab_size, output_dim = 1, 
    embed_dim = 36, num_heads = 4, ff_dim = 64):

    inputs = layers.Input(shape=(max_case_length,))
    # Three time-based features
    time_inputs = layers.Input(shape=(3,)) 
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_t = layers.Dense(32, activation="relu")(time_inputs)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    transformer = tf.keras.Model(inputs=[inputs, time_inputs], outputs=outputs,
        name = "next_time_transformer")
    return transformer

def get_remaining_time_model(max_case_length, vocab_size, output_dim = 1, 
    embed_dim = 36, num_heads = 4, ff_dim = 64):

    inputs = layers.Input(shape=(max_case_length,))
    # Three time-based features
    time_inputs = layers.Input(shape=(3,)) 
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_t = layers.Dense(32, activation="relu")(time_inputs)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    transformer = tf.keras.Model(inputs=[inputs, time_inputs], outputs=outputs,
        name = "remaining_time_transformer")
    return transformer