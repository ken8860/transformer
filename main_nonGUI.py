import os
import torch
import Train as tr
from tqdm import tqdm


def main():
    # file path
    SRC_FILE = "english_20000.txt"
    TGT_FILE = "chinese_20000.txt"
    SAVE_DIR = "model_weights"

    # check if the file exist
    if not os.path.exists(SRC_FILE) or not os.path.exists(TGT_FILE):
        print(f"❌ error: could not find corpus file! please make sure{SRC_FILE} and {TGT_FILE} exist")
        return


    # configuration
    tr.Config.EPOCHS = 150  # epochs
    tr.Config.BATCH_SIZE = 64  # batch size
    tr.Config.D_MODEL = 512
    tr.Config.D_FF = 2048
    tr.Config.H = 8
    tr.Config.K = 6  # Encoder layers
    tr.Config.P = 6  # Decoder layers
    tr.Config.WARMUP_STEPS = 4000
    tr.Config.MAX_SRC_LEN = 100
    tr.Config.MAX_TGT_LEN = 100


    pbar = None

    def progress_callback(cur_batch, total_batches):
        nonlocal pbar
        if cur_batch == 1:
            pbar = tqdm(total=total_batches, desc="Training", unit="batch", leave=False)

        pbar.update(1)

        if cur_batch == total_batches:
            pbar.close()

    def epoch_log_callback(epoch, total_epochs, trn_loss, val_loss, trn_acc, val_acc, lr):
        print(f"✨ Epoch [{epoch}/{total_epochs}] | "
              f"Loss: {trn_loss:.4f}/{val_loss:.4f} | "
              f"Acc: {trn_acc * 100:.1f}%/{val_acc * 100:.1f}% | "
              f"LR: {lr:.2e}")

    # training
    print("🚀 start training...")
    try:
        model, src_vocab, tgt_vocab = tr.build_and_train(
            src_file=SRC_FILE,
            tgt_file=TGT_FILE,
            save_dir=SAVE_DIR,
            log_fn=print,
            epoch_end_fn=epoch_log_callback,
            progress_fn=progress_callback
        )
        print("\n🎉 training finish")


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_sent = "The world is changing fast."
        print(f"\n🔍 test translation: {test_sent}")
        translation = tr.greedy_translate(model, test_sent, src_vocab, tgt_vocab, device)
        print(f"📖 result: {translation}")

    except Exception as e:
        print(f"💥 interruption: {e}")


if __name__ == "__main__":
    main()