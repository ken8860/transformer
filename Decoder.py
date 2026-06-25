import parameters as para
import torch.nn as nn
import torch
import Multi_Head_Attention as MHA
import Add_and_Norm as AN
import Feed_Forward_Network as FFN
import Input as In
import PositionalEncoding as Po

# Subsequent Mask
def make_subsequent_mask(seq_len: int, device):
    """
    generate upper triangular mask
    return shape: (1, seq_len, seq_len)，True, which means this position is masked
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0)


# single Decoder
class DecoderLayer(nn.Module):
    """
    Decoder layer contain:
      Masked multi-head self-attention
      Add & Norm(for Masked mutil-head self-attention)
      Cross Multi-Head Attention，Q from Decoder，K/V from Encoder）
      Add & Norm(for Cross Mutil-Head Attention)
      FFN
      Add & Norm(for FFN)
    """
    def __init__(self, d_model: int = para.D_MODEL, h: int = para.H, d_k: int = para.D_K,
                 d_ff: int = para.D_FF, dropout: float = para.DROPOUT):
        super().__init__()

        # Masked multi-head self-attention
        self.masked_self_attn = MHA.MultiHeadAttention(d_model, h, d_k)
        self.add_norm1        = AN.AddAndNorm(d_model, dropout)

        # Cross Multi-Head Attention
        self.cross_attn       = MHA.MultiHeadAttention(d_model, h, d_k)
        self.add_norm2        = AN.AddAndNorm(d_model, dropout)

        # FFN
        self.ffn              = FFN.FeedForwardNetwork(d_model, d_ff, dropout)
        self.add_norm3        = AN.AddAndNorm(d_model, dropout)

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """
        tgt:        (batch, tgt_len, d_model)  — Decoder input
        enc_output: (batch, src_len, d_model)  — Encoder final output
        tgt_mask:   (batch, 1, tgt_len, tgt_len) — subsequent mask
        src_mask:   (batch, 1, 1, src_len)       — masked src padding
        """
        # # Masked multi-head self-attention
        masked_attn_out = self.masked_self_attn(tgt, tgt, tgt, mask=tgt_mask)
        x = self.add_norm1(tgt, masked_attn_out)

        # Cross Multi-Head Attention
        cross_attn_out = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.add_norm2(x, cross_attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.add_norm3(x, ffn_out)

        return x


# p layers Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = para.D_MODEL, h: int = para.H,
                 d_k: int = para.D_K, d_ff: int = para.D_FF, p: int = para.NUM_DECODER_LAYERS,
                 max_len: int = 5000, dropout: float = para.DROPOUT):
        super().__init__()

        # Output layer
        self.token_embedding = In.TokenEmbedding(vocab_size, d_model)
        self.pos_encoding    = Po.PositionalEncoding(d_model, max_len, dropout)

        # p Decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_k, d_ff, dropout) for _ in range(p)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """
        tgt:        (batch, tgt_len)  — translation token ids (shifted-right)
        enc_output: (batch, src_len, d_model)
        """
        x = self.token_embedding(tgt)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        return self.norm(x)