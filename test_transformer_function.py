import parameters as para
import torch
import torch.nn as nn
import torch.nn.functional as F
import Transformer as tr

# ─────────────────────────────────────────────
# 14. 简单演示（Toy Example）
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── 语料库设定 ──────────────────────────────────────
    SRC_VOCAB_SIZE = 10000   # 原文词表（传入语料库词表大小）
    TGT_VOCAB_SIZE = 8000    # 译文字符/词表大小 v1（传入译文语料库词表大小）
    PAD_IDX        = 0       # padding token id
    BOS_IDX        = 1       # begin-of-sentence token id
    EOS_IDX        = 2       # end-of-sentence token id

    BATCH_SIZE = 2
    SRC_SEQ_LEN = 12
    TGT_SEQ_LEN = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # ── 构建模型 ──────────────────────────────────────
    model = tr.Transformer(
        src_vocab_size = SRC_VOCAB_SIZE,
        tgt_vocab_size = TGT_VOCAB_SIZE,  # v1
        d_model        = para.D_MODEL,    # n = 512
        h              = para.H,          # 8 头
        d_k            = para.D_K,        # 64，满足 h * d_k = d_model
        d_ff           = para.D_FF,       # FFN 升维 m = 2048
        k              = para.NUM_ENCODER_LAYERS,  # k=6 层 Encoder
        p              = para.NUM_DECODER_LAYERS,  # p=6 层 Decoder
        dropout        = para.DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数总量: {total_params:,}")
    print(f"架构: d_model={para.D_MODEL}, h={para.H}, d_k={para.D_K}, d_ff={para.D_FF}, "
          f"k={para.NUM_ENCODER_LAYERS}, p={para.NUM_DECODER_LAYERS}\n")

    # ── 构造假输入（随机 token ids）────────────────────
    src = torch.randint(3, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN)).to(device)
    tgt = torch.randint(3, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_SEQ_LEN)).to(device)

    # ── 前向传播 ──────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt, PAD_IDX, PAD_IDX)

    print(f"输入 src shape:    {src.shape}      (batch={BATCH_SIZE}, src_len={SRC_SEQ_LEN})")
    print(f"输入 tgt shape:    {tgt.shape}      (batch={BATCH_SIZE}, tgt_len={TGT_SEQ_LEN})")
    print(f"输出 logits shape: {logits.shape}  (batch, tgt_len, v1={TGT_VOCAB_SIZE})")

    # ── 查看某一 token 的概率分布 ─────────────────────
    probs = F.softmax(logits[0, 0, :], dim=-1)   # 第 0 个样本、第 0 个位置的预测概率
    top5_probs, top5_ids = probs.topk(5)
    print(f"\n第 0 个样本、第 0 个 token 的 Top-5 预测:")
    for rank, (p, idx) in enumerate(zip(top5_probs.tolist(), top5_ids.tolist()), 1):
        print(f"  Top-{rank}: token_id={idx:5d}  prob={p:.6f}")

    # ── 训练示例（一个 step）────────────────────────
    print("\n── 训练一步示例 ──")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 标签：tgt 右移一位（即预测下一个 token）
    tgt_input  = tgt[:, :-1]                        # (batch, tgt_len-1)
    tgt_output = tgt[:, 1:].contiguous().view(-1)   # (batch*(tgt_len-1),)

    logits_train = model(src, tgt_input, PAD_IDX, PAD_IDX)
    logits_flat  = logits_train.contiguous().view(-1, TGT_VOCAB_SIZE)

    loss = criterion(logits_flat, tgt_output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"训练 Loss: {loss.item():.4f}")
    print("\n✓ Transformer 完整前向 + 反向传播验证通过！")
