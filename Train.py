"""
train.py — Transformer

public interface or class:
    Config                        parameter setting
    Vocabulary                    Vocabulary
    TranslationDataset            dataset
    collate_fn                    DataLoader batch function
    NoamScheduler                 Noam learning rate
    run_epoch()                   epoch
    greedy_translate()            greedy translator
    save_checkpoint()             save check point
    load_checkpoint()             load check point
    build_and_train()             built and train model
    tokenize_en(text)             english tokenize
    tokenize_zh(text)             chinese tokenize
"""

import os
import re
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import Transformer as tr
import parameters as para


# parameter setting
class Config:
    SRC_FILE  = None          # source.txt path
    TGT_FILE  = None          # target.txt path
    SAVE_DIR  = "."

    @classmethod
    def best_model_path(cls):
        return os.path.join(cls.SAVE_DIR, "best_model.pt")

    @classmethod
    def checkpoint_path(cls):
        return os.path.join(cls.SAVE_DIR, "checkpoint.pt")

    # the maxsize of vocabulary
    SRC_VOCAB_SIZE = 30000
    TGT_VOCAB_SIZE = 30000
    MIN_FREQ       = 1
    MAX_SRC_LEN    = 100
    MAX_TGT_LEN    = 100

    # special token
    PAD = "<pad>";  PAD_IDX = 0
    BOS = "<bos>";  BOS_IDX = 1
    EOS = "<eos>";  EOS_IDX = 2
    UNK = "<unk>";  UNK_IDX = 3
    SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # training parameters
    BATCH_SIZE   = 32
    EPOCHS       = 30
    WARMUP_STEPS = 4000
    CLIP_GRAD    = 1.0
    VAL_RATIO    = 0.05
    SEED         = 42

    # parameters from parameters.py
    D_MODEL = para.D_MODEL
    H       = para.H
    D_K     = para.D_K
    D_FF    = para.D_FF
    K       = para.NUM_ENCODER_LAYERS
    P       = para.NUM_DECODER_LAYERS
    DROPOUT = para.DROPOUT
    MAX_LEN = 512


# tokenizer
def tokenize_en(text: str):
    return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower().strip())


def tokenize_zh(text: str):
    return list(text.strip())


# Vocabulary
class Vocabulary:
    def __init__(self, specials=None):
        self.specials = specials or Config.SPECIALS
        self.token2id = {tok: i for i, tok in enumerate(self.specials)}
        self.id2token = {i: tok for i, tok in enumerate(self.specials)}

    def build(self, token_lists, max_size=10000, min_freq=1):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        sorted_tokens = [
            tok for tok, freq in counter.most_common()
            if freq >= min_freq and tok not in self.token2id
        ]
        sorted_tokens = sorted_tokens[: max_size - len(self.specials)]

        for tok in sorted_tokens:
            idx = len(self.token2id)
            self.token2id[tok] = idx
            self.id2token[idx] = tok

        return self

    # encode
    def encode(self, tokens, add_bos=True, add_eos=True):
        ids = [self.token2id.get(t, Config.UNK_IDX) for t in tokens]
        if add_bos:
            ids = [Config.BOS_IDX] + ids
        if add_eos:
            ids = ids + [Config.EOS_IDX]
        return ids

    # decode
    def decode(self, ids, skip_special=True):
        special_set = {Config.PAD_IDX, Config.BOS_IDX,
                       Config.EOS_IDX, Config.UNK_IDX}
        out = []
        for i in ids:
            if skip_special and i in special_set:
                if i == Config.EOS_IDX:
                    break
                continue
            out.append(self.id2token.get(i, Config.UNK))
        return "".join(out)

    def __len__(self):
        return len(self.token2id)


# dataset
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file,
                 src_vocab=None, tgt_vocab=None,
                 build_vocab=True,
                 log_fn=print):

        with open(src_file, "r", encoding="utf-8") as f:
            src_raw = [l.strip() for l in f if l.strip()]
        with open(tgt_file, "r", encoding="utf-8") as f:
            tgt_raw = [l.strip() for l in f if l.strip()]

        if len(src_raw) != len(tgt_raw):
            raise ValueError(
                f"inconsistent! source row: {len(src_raw)}，target row: {len(tgt_raw)}"
            )
        log_fn(f"original sentence pair: {len(src_raw)}")

        src_tokens = [tokenize_en(s) for s in src_raw]
        tgt_tokens = [tokenize_zh(s) for s in tgt_raw]

        pairs = [
            (s, t) for s, t in zip(src_tokens, tgt_tokens)
            if 1 <= len(s) <= Config.MAX_SRC_LEN
            and 1 <= len(t) <= Config.MAX_TGT_LEN
        ]
        log_fn(f"filter sentence pair: {len(pairs)}")
        src_tokens, tgt_tokens = zip(*pairs)

        if build_vocab:
            log_fn("build source vocabulary...")
            self.src_vocab = Vocabulary().build(
                src_tokens,
                max_size=Config.SRC_VOCAB_SIZE,
                min_freq=Config.MIN_FREQ,
            )
            log_fn(f"  source vocabulary size: {len(self.src_vocab)}")

            log_fn("build target vocabulary...")
            self.tgt_vocab = Vocabulary().build(
                tgt_tokens,
                max_size=Config.TGT_VOCAB_SIZE,
                min_freq=Config.MIN_FREQ,
            )
            log_fn(f"  target vocabulary size: {len(self.tgt_vocab)}")
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # encode the word as id
        self.src_ids = [self.src_vocab.encode(s) for s in src_tokens]
        self.tgt_ids = [self.tgt_vocab.encode(t) for t in tgt_tokens]

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_ids[idx], dtype=torch.long),
            torch.tensor(self.tgt_ids[idx], dtype=torch.long),
        )


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    return (
        pad_sequence(src_batch, padding_value=Config.PAD_IDX, batch_first=True),
        pad_sequence(tgt_batch, padding_value=Config.PAD_IDX, batch_first=True),
    )


# Noam learning rate
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self._step        = 0
        self._rate        = 0.0

    def step(self):
        self._step += 1
        rate = (self.d_model ** -0.5) * min(
            self._step ** -0.5,
            self._step * (self.warmup_steps ** -1.5),
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def current_lr(self):
        return self._rate


# epoch
def run_epoch(model, dataloader, criterion, optimizer,
              device, is_train, tgt_vocab_size,
              progress_fn=None):

    model.train() if is_train else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0
    n_batches     = len(dataloader)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]                       # remove <eos>
            tgt_label = tgt[:, 1:].contiguous().view(-1)  # remove <bos>

            logits = model(src, tgt_input,
                           src_pad_idx=Config.PAD_IDX,
                           tgt_pad_idx=Config.PAD_IDX)
            logits_flat = logits.contiguous().view(-1, tgt_vocab_size)

            loss = criterion(logits_flat, tgt_label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)
                optimizer.step()

            total_loss += loss.item()

            non_pad = tgt_label != Config.PAD_IDX
            pred    = logits_flat.argmax(dim=-1)
            total_correct += (pred[non_pad] == tgt_label[non_pad]).sum().item()
            total_tokens  += non_pad.sum().item()

            if progress_fn is not None:
                progress_fn(batch_idx + 1, n_batches)

    avg_loss = total_loss / n_batches
    accuracy = total_correct / max(total_tokens, 1)
    return avg_loss, accuracy


# greedy translator
@torch.no_grad()
def greedy_translate(model, sentence, src_vocab, tgt_vocab, device,
                     max_len=None):

    if max_len is None:
        max_len = Config.MAX_TGT_LEN

    model.eval()

    tokens     = tokenize_en(sentence)
    src_ids    = src_vocab.encode(tokens, add_bos=True, add_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

    src_mask   = model.make_src_mask(src_tensor, Config.PAD_IDX)
    enc_output = model.encoder(src_tensor, src_mask)

    tgt_ids = [Config.BOS_IDX]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
        tgt_mask   = model.make_tgt_mask(tgt_tensor, Config.PAD_IDX)
        dec_out    = model.decoder(tgt_tensor, enc_output, tgt_mask, src_mask)
        logits     = model.output_layer(dec_out)
        next_id    = logits[0, -1, :].argmax(dim=-1).item()
        tgt_ids.append(next_id)
        if next_id == Config.EOS_IDX:
            break

    return tgt_vocab.decode(tgt_ids)


# save check point
def save_checkpoint(path, model, optimizer, epoch,
                    best_val_loss, src_vocab, tgt_vocab):
    torch.save({
        "epoch":         epoch,
        "model_state":   model.state_dict(),
        "optim_state":   optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "src_vocab":     src_vocab,
        "tgt_vocab":     tgt_vocab,
    }, path)


# load check point
def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt


# build and train
def build_and_train(
    src_file: str,
    tgt_file: str,
    save_dir: str = ".",
    log_fn=print,
    epoch_end_fn=None,
    progress_fn=None,
    stop_flag=None,
):
    # Config
    Config.SRC_FILE = src_file
    Config.TGT_FILE = tgt_file
    Config.SAVE_DIR = save_dir
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_fn(f"the device is: {device}")

    # 1: dataset
    log_fn("\nread corpus and build vocabulary")
    full_dataset = TranslationDataset(
        src_file, tgt_file, build_vocab=True, log_fn=log_fn
    )
    src_vocab = full_dataset.src_vocab
    tgt_vocab = full_dataset.tgt_vocab

    # 2: divide train/validate
    total    = len(full_dataset)
    val_size = max(1, int(total * Config.VAL_RATIO))
    trn_size = total - val_size
    gen      = torch.Generator().manual_seed(Config.SEED)
    train_ds, val_ds = random_split(full_dataset, [trn_size, val_size],
                                    generator=gen)
    log_fn(f"number of train set sentences: {trn_size}, number of validate set sentences: {val_size}")

    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn,
                              pin_memory=use_pin)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn,
                              pin_memory=use_pin)

    # 3: model
    log_fn("\ninitial model")
    model = tr.Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = Config.D_MODEL,
        h              = Config.H,
        d_k            = Config.D_K,
        d_ff           = Config.D_FF,
        k              = Config.K,
        p              = Config.P,
        max_len        = Config.MAX_LEN,
        dropout        = Config.DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_fn(f"number of trainable parameters: {n_params:,}")

    # 4: optimizer & scheduling & loss ──────────────────────────
    base_opt  = torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(base_opt, Config.D_MODEL, Config.WARMUP_STEPS)
    criterion = nn.CrossEntropyLoss(
        ignore_index=Config.PAD_IDX, label_smoothing=0.1
    )

    start_epoch   = 0
    best_val_loss = float("inf")
    ckpt_path     = Config.checkpoint_path()

    if os.path.exists(ckpt_path):
        log_fn(f"find checkpoint {ckpt_path}，continue training...")
        ckpt          = load_checkpoint(ckpt_path, model, base_opt)
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        src_vocab     = ckpt["src_vocab"]
        tgt_vocab     = ckpt["tgt_vocab"]
        log_fn(f"  from Epoch {start_epoch} continue training，history best val_loss={best_val_loss:.4f}")

    tgt_vocab_size = len(tgt_vocab)

    # 5: training loop
    log_fn(f"\nstart training( totally {Config.EPOCHS} Epochs)")
    for epoch in range(start_epoch, Config.EPOCHS):

        if stop_flag is not None and stop_flag.is_set():
            log_fn("the training has been stopped by user")
            break

        t0 = time.time()

        trn_loss, trn_acc = run_epoch(
            model, train_loader, criterion, scheduler, device,
            is_train=True,  tgt_vocab_size=tgt_vocab_size,
            progress_fn=progress_fn,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, scheduler, device,
            is_train=False, tgt_vocab_size=tgt_vocab_size,
        )

        elapsed = time.time() - t0
        msg = (
            f"Epoch [{epoch+1:3d}/{Config.EPOCHS}] | "
            f"lr={scheduler.current_lr:.2e} | "
            f"Train Loss={trn_loss:.4f}  Acc={trn_acc*100:.1f}% | "
            f"Val Loss={val_loss:.4f}  Acc={val_acc*100:.1f}% | "
            f"Time={elapsed:.1f}s"
        )
        log_fn(msg)

        if epoch_end_fn is not None:
            epoch_end_fn(epoch + 1, Config.EPOCHS,
                         trn_loss, val_loss,
                         trn_acc, val_acc,
                         scheduler.current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(Config.best_model_path(), model, base_opt,
                            epoch, best_val_loss, src_vocab, tgt_vocab)
            log_fn(f"  ✅ the best model has been kept to {Config.best_model_path()}")

        save_checkpoint(ckpt_path, model, base_opt,
                        epoch, best_val_loss, src_vocab, tgt_vocab)

    log_fn(f"\n✅ the training finished! the best Val Loss = {best_val_loss:.4f}")

    best_path = Config.best_model_path()
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)
        log_fn("already load the best weight")

    return model, src_vocab, tgt_vocab