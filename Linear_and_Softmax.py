import torch.nn as nn
import torch.nn.functional as F

# Linear and Softmax
class LinearSoftmax(nn.Module):
    """
    final output layer:
      Linear:  d_model → v1 (the size of vocabulary)
      Softmax: --> Probability distribution
    """
    def __init__(self, d_model: int, v1: int):
        super().__init__()
        self.linear = nn.Linear(d_model, v1)

    def forward(self, x, apply_softmax: bool = False):
        """
        parameters:
          x:  (batch, tgt_len, d_model)
        return: (batch, tgt_len, v1)
          return logits while training(for CrossEntropy)
          return Probability distribution while reasoning(apply_softmax=True)
        """
        logits = self.linear(x)
        if apply_softmax:
            return F.softmax(logits, dim=-1)
        return logits
