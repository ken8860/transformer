"""
Parameters:
  - n = 512       d_model
  - h = 8         head
  - d_k = d_q = d_v = 64   dimension of each head
  - k = 6         Encoder layer
  - p = 6         Decoder layer
  - FFN           Dimensional Elevation: d_ff = 2048
  - v1            Vocabulary
"""
# hyper parameters
D_MODEL = 512
H       = 8
D_K     = D_MODEL // H
assert D_MODEL % H == 0,"D_MODEL % H must be equal 0"  # assert D_MODEL % H == 0

NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
D_FF = 2048
DROPOUT = 0.1