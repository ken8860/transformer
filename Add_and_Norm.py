import parameters as para
import torch.nn as nn

# Add & Norm
class AddAndNorm(nn.Module):
    """
    resnet + Layer Normalization：
      output = LayerNorm( x + sublayer(x) )
    """
    def __init__(self, d_model: int = para.D_MODEL, dropout: float = para.DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        """
        x:shape(batch, seq, d_model)
        sublayer_output: same shape as FFN
        """
        return self.norm(x + self.dropout(sublayer_output))