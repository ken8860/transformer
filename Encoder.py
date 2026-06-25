import parameters as para
import torch.nn as nn
import Multi_Head_Attention as MHA
import Add_and_Norm as AN
import Feed_Forward_Network as FFN
import Input as In
import PositionalEncoding as Po

# single Encoder layer
class EncoderLayer(nn.Module):
    """
    Encoder layer contain:
      self-attention
      Add & Norm(for self-attention)
      FFN
      Add & Norm(for FFN)
    """
    def __init__(self, d_model: int = para.D_MODEL, h: int = para.H, d_k: int = para.D_K,
                 d_ff: int = para.D_FF, dropout: float = para.DROPOUT):
        super().__init__()
        self.self_attn = MHA.MultiHeadAttention(d_model, h, d_k)
        self.add_norm1 = AN.AddAndNorm(d_model, dropout)
        self.ffn = FFN.FeedForwardNetwork(d_model, d_ff, dropout)
        self.add_norm2 = AN.AddAndNorm(d_model, dropout)

    def forward(self, x, src_mask=None):
        """
        x:(batch, src_len, d_model)
        src_mask:(batch, 1, 1, src_len)
        """
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.add_norm1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.add_norm2(x, ffn_out)
        return x


# k layers Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = para.D_MODEL, h: int = para.H,
                 d_k: int = para.D_K, d_ff: int = para.D_FF, k: int = para.NUM_ENCODER_LAYERS,
                 max_len: int = 5000, dropout: float = para.DROPOUT):
        super().__init__()

        # Input
        self.token_embedding  = In.TokenEmbedding(vocab_size, d_model)
        self.pos_encoding     = Po.PositionalEncoding(d_model, max_len, dropout)

        # k Encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_k, d_ff, dropout) for _ in range(k)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        """
        parameters:
           src:(batch, src_len)
           src_mask:(batch, 1, 1, src_len)
        return: (batch, src_len, d_model)
        """

        # embedding + positional encoding
        x = self.token_embedding(src)
        x = self.pos_encoding(x)

        # pass k Encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)   
