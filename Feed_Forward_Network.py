import parameters as para
import torch.nn as nn
import torch.nn.functional as F

# Feed-Forward Network
class FeedForwardNetwork(nn.Module):
    """
    two layer network, first layer upgrade dimension, second reduce dimension
       FFN(x) = ReLU( x · W_1 + b_1 ) · W_2 + b_2
       d_model → d_ff (dimension = 2048) → d_model
    """
    def __init__(self, d_model: int = para.D_MODEL, d_ff: int = para.D_FF, dropout: float = para.DROPOUT):
        super().__init__()
        # upgrade dimension: d_model --> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        # reduce dimension: d_ff --> d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x:(batch, seq, d_model)
        return:(batch, seq, d_model)
        """
        # layer 1, shape:(batch, seq, d_ff)
        h = F.relu(self.linear1(x))
        h = self.dropout(h)
        # layer 2, shape:(batch, seq, d_model)
        out = self.linear2(h)
        return out