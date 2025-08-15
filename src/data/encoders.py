class Vocabulary:
    """Simple vocabulary with pad and end-of-case tokens."""
    def __init__(self, tokens):
        unique_tokens = sorted(set(tokens))
        self.index_to_token = ["<pad>"] + unique_tokens + ["<eoc>"]
        self.token_to_index = {token: index for index, token in enumerate(self.index_to_token)}

    def encode_sequence(self, token_sequence):
        return [self.token_to_index[token] for token in token_sequence]

    def decode_sequence(self, index_sequence):
        return [self.index_to_token[index] for index in index_sequence]
