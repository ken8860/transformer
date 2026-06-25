import math
import torch.nn as nn

# Input(embedding)
class TokenEmbedding(nn.Module):
    """
    token id reflect to d_model dimension stochastic trainable word vector
    the weight initial some random value, update while training
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)
        self.d_model = d_model

    def forward(self, x):
        """
        x: (batch, seq_len)  -> token ids
        return: (batch, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)

