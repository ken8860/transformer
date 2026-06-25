import parameters as para
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# single head attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    calculate one head output
    parameters:
      Q: (batch, seq_q, d_k)
      K: (batch, seq_k, d_k)
      V: (batch, seq_k, d_v)
      mask: (batch, 1, seq_q, seq_k) or None
    return:
      Z_n:     (batch, seq_q, d_v)
      attn_w:  (batch, seq_q, seq_k)
    """
    d_k = Q.size(-1)

    # QK^T
    raw_scores = torch.matmul(Q, K.transpose(-2, -1))
    # print(f"[Attention] raw_scores (QK^T) shape: {raw_scores.shape}")

    # QK^T / sqrt(d_k)
    scaled_scores = raw_scores / math.sqrt(d_k)

    # if mask or not
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask, float("-inf"))

    # softmax(QK^T / sqrt(d_k)) --> score
    attn_w = F.softmax(scaled_scores, dim=-1)

    # output for one head
    Z_n = torch.matmul(attn_w, V)

    return Z_n, attn_w

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """
    step for multi-head attention:
      1. for each head, there are different Q/K/V matrix, the weight matrix are W_q^i, W_k^i, W_v^i
      2. for each head, calculate scaled_dot_product_attention --> Z_1,...,Z_h independency
      3. connect Z_concat = Concat(Z_1,...,Z_h)
      4. Z_concat multiply W_o --> final output Z
    parameters:
      d_model: model dimensions
      h:       head
      d_k:     head dimension, which equal to d_model/h, d_k = d_v = d_q
    """
    def __init__(self, d_model: int = para.D_MODEL, h: int = para.H, d_k: int = para.D_K):
        super().__init__()
        assert d_model == h * d_k, f"d_model({d_model}) != h({h}) * d_k({d_k})"
        self.h = h
        self.d_k = d_k
        self.d_model = d_model

        # for every head, one W_q, W_k, W_v corresponding, put every W_q, W_k, W_v into one big matrix
        # the big matrix shape will be (d_model * d_model) since for one head W_q, W_k, W_v would be (d_model * d_k)
        self.W_q = nn.Linear(d_model, h * d_k, bias=False)
        self.W_k = nn.Linear(d_model, h * d_k, bias=False)
        self.W_v = nn.Linear(d_model, h * d_k, bias=False)

        # the same as W_q, W_k, W_v
        self.W_o = nn.Linear(h * d_k, d_model, bias=False)
        self.dropout = nn.Dropout(p = para.DROPOUT)

    def forward(self, query, key, value, mask=None):
        """
        parameters:
           query: (batch, seq_q, d_model)
           key:   (batch, seq_k, d_model)
           value: (batch, seq_k, d_model)
           mask:  (batch, 1, seq_q, seq_k) or None
        return: (batch, seq_q, d_model)
        """
        batch = query.size(0)
        seq_q = query.size(1)

        # W_q(query): (batch, seq_q, d_model) → (batch, seq_q, h*d_k)
        # W_k(key): (batch, seq_k, d_model) → (batch, seq_k, h*d_k)
        # W_v(value): (batch, seq_v, d_model) → (batch, seq_v, h*d_k)
        Q_proj = self.W_q(query)
        K_proj = self.W_k(key)
        V_proj = self.W_v(value)

        # reshape to (batch, seq, h, d_k)，and transpose → (batch, h, seq, d_k)
        Q_heads = Q_proj.view(batch, -1, self.h, self.d_k).transpose(1, 2)
        K_heads = K_proj.view(batch, -1, self.h, self.d_k).transpose(1, 2)
        V_heads = V_proj.view(batch, -1, self.h, self.d_k).transpose(1, 2)

        # compute every head,
        Z_heads, attn_weights = scaled_dot_product_attention(
            Q_heads, K_heads, V_heads, mask=mask
        )

        # transpose back: (batch, seq_q, h, d_k) → contiguous → (batch, seq_q, h*d_k)
        Z_concat = Z_heads.transpose(1, 2).contiguous().view(batch, seq_q, self.h * self.d_k)

        # Z = Z_concat · W_o, shape: (batch, seq_q, d_model)
        Z = self.W_o(Z_concat)
        Z = self.dropout(Z)

        return Z