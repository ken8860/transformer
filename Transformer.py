import parameters as para
import torch.nn as nn
import Encoder as En
import Decoder as De
import Linear_and_Softmax as LS

# Transformer
class Transformer(nn.Module):
    """
    Transformer(Encoder-Decoder)

    parameters:
      src_vocab_size: source vocabulary size
      tgt_vocab_size: translational vocabulary size
      d_model:        d_model
      h:              head
      d_k:            d_k = d_model/head
      d_ff:           FFN upgrade dimension
      k:              Encoder layers
      p:              Decoder layers
      max_len:        the max sequent length
      dropout:        Dropout rate
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int   = para.D_MODEL,
        h: int         = para.H,
        d_k: int       = para.D_K,
        d_ff: int      = para.D_FF,
        k: int         = para.NUM_ENCODER_LAYERS,
        p: int         = para.NUM_DECODER_LAYERS,
        max_len: int   = 5000,
        dropout: float = para.DROPOUT,
    ):
        super().__init__()
        self.encoder        = En.Encoder(src_vocab_size, d_model, h, d_k, d_ff, k, max_len, dropout)
        self.decoder        = De.Decoder(tgt_vocab_size, d_model, h, d_k, d_ff, p, max_len, dropout)
        self.output_layer   = LS.LinearSoftmax(d_model, tgt_vocab_size)
        self.d_model        = d_model
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx: int = 0):
        """
        parameters:
           masked padding token in source。
           src: (batch, src_len)
        return: (batch, 1, 1, src_len)  True = masked
        """
        return (src == pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt, pad_idx: int = 0):
        """
        combine padding mask and subsequent mask.
        parameters:
           tgt: (batch, tgt_len)
        return: (batch, 1, tgt_len, tgt_len)
        """
        tgt_len = tgt.size(1)
        device  = tgt.device

        pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

        sub_mask = De.make_subsequent_mask(tgt_len, device)
        sub_mask = sub_mask.unsqueeze(1)

        tgt_mask = pad_mask | sub_mask
        return tgt_mask

    def forward(self, src, tgt, src_pad_idx: int = 0, tgt_pad_idx: int = 0):
        """
        parameters:
           src: (batch, src_len)  source text
           tgt: (batch, tgt_len)  translation text
        return: logits (batch, tgt_len, v1)
        """
        src_mask = self.make_src_mask(src, src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, tgt_pad_idx)

        # Encoder
        enc_output = self.encoder(src, src_mask)

        # Decoder
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)

        # Linear and (log)softmax
        logits = self.output_layer(dec_output)

        return logits
