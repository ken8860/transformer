"""
main.py — Transformer GUI
the function of GUI:
    1,choose source .txt file
    2,choose target .txt file
    3,choose model weight save directory
    4,changeable parameters(epochs / batch size / D_model)
    5,click start training
    6,click stop training
    7,after training, test the translator
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import Train as tr


def _safe(root, fn):
    def wrapper(*args, **kwargs):
        root.after(0, fn, *args, **kwargs)
    return wrapper


# the main window
class TransformerGUI:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Transformer translation trainer")
        self.root.resizable(True, True)
        self.root.minsize(760, 680)

        # training state
        self._stop_flag  = threading.Event()
        self._model      = None
        self._src_vocab  = None
        self._tgt_vocab  = None
        self._is_training = False

        # Tk
        self._src_file  = tk.StringVar()
        self._tgt_file  = tk.StringVar()
        self._save_dir  = tk.StringVar(value=os.getcwd())
        self._epochs    = tk.IntVar(value=tr.Config.EPOCHS)
        self._batch     = tk.IntVar(value=tr.Config.BATCH_SIZE)
        self._d_model   = tk.IntVar(value=tr.Config.D_MODEL)
        self._d_ff      = tk.IntVar(value=tr.Config.D_FF)
        self._heads     = tk.IntVar(value=tr.Config.H)
        self._enc_layers= tk.IntVar(value=tr.Config.K)
        self._dec_layers= tk.IntVar(value=tr.Config.P)
        self._warmup    = tk.IntVar(value=tr.Config.WARMUP_STEPS)
        self._max_src   = tk.IntVar(value=tr.Config.MAX_SRC_LEN)
        self._max_tgt   = tk.IntVar(value=tr.Config.MAX_TGT_LEN)

        self._build_ui()


    # UI
    def _build_ui(self):
        pad = dict(padx=8, pady=4)
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=6)

        tab_train   = ttk.Frame(nb)
        tab_infer   = ttk.Frame(nb)
        nb.add(tab_train, text="  training configuration  ")
        nb.add(tab_infer, text="  translate test  ")

        # choose the file
        file_frame = ttk.LabelFrame(tab_train, text="corpus file & save directory", padding=8)
        file_frame.pack(fill="x", **pad)
        file_frame.columnconfigure(1, weight=1)

        # source file
        ttk.Label(file_frame, text="source (.txt):").grid(
            row=0, column=0, sticky="w", pady=3)
        ttk.Entry(file_frame, textvariable=self._src_file,
                  state="readonly").grid(
            row=0, column=1, sticky="ew", padx=6)
        ttk.Button(file_frame, text="choose file",
                   command=self._pick_src).grid(row=0, column=2)

        # target file
        ttk.Label(file_frame, text="target (.txt):").grid(
            row=1, column=0, sticky="w", pady=3)
        ttk.Entry(file_frame, textvariable=self._tgt_file,
                  state="readonly").grid(
            row=1, column=1, sticky="ew", padx=6)
        ttk.Button(file_frame, text="choose file",
                   command=self._pick_tgt).grid(row=1, column=2)

        # save directory
        ttk.Label(file_frame, text="save directory:").grid(
            row=2, column=0, sticky="w", pady=3)
        ttk.Entry(file_frame, textvariable=self._save_dir,
                  state="readonly").grid(
            row=2, column=1, sticky="ew", padx=6)
        ttk.Button(file_frame, text="choose directory",
                   command=self._pick_dir).grid(row=2, column=2)


        # choose parameters
        hp_frame = ttk.LabelFrame(tab_train, text="parameters", padding=8)
        hp_frame.pack(fill="x", **pad)

        params = [
            ("Epochs",    self._epochs,     0, 0),
            ("Batch Size",           self._batch,      0, 2),
            ("D_Model", self._d_model,    1, 0),
            ("D_FF",       self._d_ff,       1, 2),
            ("Head H",          self._heads,      2, 0),
            ("Encoder layer K",        self._enc_layers, 2, 2),
            ("Decoder layer P",        self._dec_layers, 3, 0),
            ("Warmup Steps",          self._warmup,     3, 2),
            ("the longest of source",           self._max_src,    4, 0),
            ("the longest of target",           self._max_tgt,    4, 2),
        ]
        for label, var, row, col in params:
            ttk.Label(hp_frame, text=label + ":").grid(
                row=row, column=col,     sticky="w", padx=4, pady=3)
            ttk.Entry(hp_frame, textvariable=var, width=10).grid(
                row=row, column=col + 1, sticky="w", padx=(0, 16))


        # button
        btn_frame = ttk.Frame(tab_train)
        btn_frame.pack(fill="x", **pad)

        self._start_btn = ttk.Button(
            btn_frame, text="▶  start training",
            command=self._start_training, style="Accent.TButton")
        self._start_btn.pack(side="left", ipadx=10, ipady=4, padx=(0, 8))

        self._stop_btn = ttk.Button(
            btn_frame, text="■  stop training",
            command=self._stop_training, state="disabled")
        self._stop_btn.pack(side="left", ipadx=10, ipady=4)


        # progress bar
        prog_frame = ttk.Frame(tab_train)
        prog_frame.pack(fill="x", padx=8, pady=(0, 4))

        ttk.Label(prog_frame, text="Epoch progress:").pack(side="left")
        self._epoch_bar = ttk.Progressbar(prog_frame, mode="determinate",
                                          length=300)
        self._epoch_bar.pack(side="left", padx=6, fill="x", expand=True)
        self._epoch_label = ttk.Label(prog_frame, text="0 / 0")
        self._epoch_label.pack(side="left")

        prog_frame2 = ttk.Frame(tab_train)
        prog_frame2.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(prog_frame2, text="Batch progress:").pack(side="left")
        self._batch_bar = ttk.Progressbar(prog_frame2, mode="determinate",
                                          length=300)
        self._batch_bar.pack(side="left", padx=6, fill="x", expand=True)
        self._batch_label = ttk.Label(prog_frame2, text="0 / 0")
        self._batch_label.pack(side="left")


        # log
        log_frame = ttk.LabelFrame(tab_train, text="training log", padding=4)
        log_frame.pack(fill="both", expand=True, **pad)

        self._log_box = scrolledtext.ScrolledText(
            log_frame, state="disabled", height=14,
            font=("Consolas", 9), wrap="word",
            background="#1e1e1e", foreground="#d4d4d4",
            insertbackground="white",
        )
        self._log_box.pack(fill="both", expand=True)
        self._log_box.tag_config("info",    foreground="#9cdcfe")
        self._log_box.tag_config("success", foreground="#4ec9b0")
        self._log_box.tag_config("warn",    foreground="#dcdcaa")
        self._log_box.tag_config("error",   foreground="#f44747")


        # translation test
        ttk.Label(tab_infer,
                  text="the training has been finished, input the the sentence to test the translation:",
                  font=("", 10)).pack(anchor="w", padx=10, pady=(12, 4))

        in_frame = ttk.LabelFrame(tab_infer, text="input sentence", padding=8)
        in_frame.pack(fill="x", padx=10, pady=4)

        self._infer_input = tk.Text(in_frame, height=4, font=("", 11),
                                    wrap="word")
        self._infer_input.pack(fill="x")
        self._infer_input.insert("1.0", "Please enter an English sentence here.")
        self._infer_input.bind("<FocusIn>", self._clear_placeholder)

        ttk.Button(tab_infer, text="translation",
                   command=self._do_translate).pack(pady=6)

        out_frame = ttk.LabelFrame(tab_infer, text="translation result", padding=8)
        out_frame.pack(fill="both", expand=True, padx=10, pady=4)

        self._infer_output = scrolledtext.ScrolledText(
            out_frame, height=8, font=("", 11), state="disabled",
            wrap="word", background="#f5f5f5"
        )
        self._infer_output.pack(fill="both", expand=True)


    # choose file
    def _pick_src(self):
        path = filedialog.askopenfilename(
            title="choose source file",
            filetypes=[("txt file", "*.txt"), ("all file", "*.*")]
        )
        if path:
            self._src_file.set(path)

    def _pick_tgt(self):
        path = filedialog.askopenfilename(
            title="choose target file",
            filetypes=[("txt file", "*.txt"), ("all file", "*.*")]
        )
        if path:
            self._tgt_file.set(path)

    def _pick_dir(self):
        path = filedialog.askdirectory(title="choose the directory of model weight")
        if path:
            self._save_dir.set(path)


    # log writing
    def _log(self, msg: str, tag="info"):
        def _write():
            self._log_box.configure(state="normal")
            self._log_box.insert("end", msg + "\n", tag)
            self._log_box.see("end")
            self._log_box.configure(state="disabled")
        self.root.after(0, _write)

    def _log_success(self, msg): self._log(msg, "success")
    def _log_warn(self, msg):    self._log(msg, "warn")
    def _log_error(self, msg):   self._log(msg, "error")

    def _dispatch_log(self, msg: str):
        if "✓" in msg or "✅" in msg or "finish" in msg:
            self._log_success(msg)
        elif "error" in msg or "Error" in msg or "fail" in msg:
            self._log_error(msg)
        elif "warn" in msg or "Warning" in msg:
            self._log_warn(msg)
        else:
            self._log(msg, "info")


    # update progress
    def _update_epoch_progress(self, cur_epoch, total_epochs,
                                trn_loss, val_loss,
                                trn_acc, val_acc, lr):
        def _update():
            pct = int(cur_epoch / total_epochs * 100)
            self._epoch_bar["value"] = pct
            self._epoch_label.config(text=f"{cur_epoch} / {total_epochs}")
        self.root.after(0, _update)

    def _update_batch_progress(self, cur_batch, total_batches):
        def _update():
            pct = int(cur_batch / total_batches * 100)
            self._batch_bar["value"] = pct
            self._batch_label.config(text=f"{cur_batch} / {total_batches}")
        self.root.after(0, _update)


    # start/stop
    def _start_training(self):
        src = self._src_file.get().strip()
        tgt = self._tgt_file.get().strip()
        sav = self._save_dir.get().strip()

        if not src:
            messagebox.showwarning("missing file", "please choose source file (.txt)")
            return
        if not tgt:
            messagebox.showwarning("missing file", "please choose target file (.txt)")
            return
        if not os.path.isfile(src):
            messagebox.showerror("file not exist", f"could not find source file: \n{src}")
            return
        if not os.path.isfile(tgt):
            messagebox.showerror("file not exist", f"could not find target file: \n{tgt}")
            return

        # import parameters
        try:
            tr.Config.EPOCHS       = self._epochs.get()
            tr.Config.BATCH_SIZE   = self._batch.get()
            tr.Config.D_MODEL      = self._d_model.get()
            tr.Config.D_FF         = self._d_ff.get()
            tr.Config.H            = self._heads.get()
            tr.Config.K            = self._enc_layers.get()
            tr.Config.P            = self._dec_layers.get()
            tr.Config.WARMUP_STEPS = self._warmup.get()
            tr.Config.MAX_SRC_LEN  = self._max_src.get()
            tr.Config.MAX_TGT_LEN  = self._max_tgt.get()
            tr.Config.D_K          = tr.Config.D_MODEL // tr.Config.H
        except Exception as e:
            messagebox.showerror("parameters error", f"the parameters are filled in incorrectly: {e}")
            return

        if tr.Config.D_MODEL % tr.Config.H != 0:
            messagebox.showerror(
                "parameters error",
                f"D_Model ({tr.Config.D_MODEL}) must be divided of H ({tr.Config.H})"
            )
            return


        self._stop_flag.clear()
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._epoch_bar["value"]  = 0
        self._batch_bar["value"]  = 0
        self._epoch_label.config(text=f"0 / {tr.Config.EPOCHS}")
        self._batch_label.config(text="0 / 0")
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")
        self._model = None

        self._log("═" * 55, "info")
        self._log("  Transformer start training", "success")
        self._log("═" * 55, "info")
        self._log(f"source file: {src}")
        self._log(f"target file: {tgt}")
        self._log(f"save directory: {sav}")

        def _train_thread():
            try:
                model, sv, tv = tr.build_and_train(
                    src_file     = src,
                    tgt_file     = tgt,
                    save_dir     = sav,
                    log_fn       = self._dispatch_log,
                    epoch_end_fn = self._update_epoch_progress,
                    progress_fn  = self._update_batch_progress,
                    stop_flag    = self._stop_flag,
                )
                self._model     = model
                self._src_vocab = sv
                self._tgt_vocab = tv
                self.root.after(0, self._on_train_done)
            except Exception as e:
                self._log_error(f"training error: {e}")
                import traceback
                self._log_error(traceback.format_exc())
                self.root.after(0, self._on_train_done)

        t = threading.Thread(target=_train_thread, daemon=True)
        t.start()

    def _stop_training(self):
        self._stop_flag.set()
        self._log_warn("sending stop signal, waiting for batch finish...")

    def _on_train_done(self):
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._epoch_bar["value"] = 100
        self._batch_bar["value"] = 100
        if self._model is not None:
            self._log_success("model has been ready, switch to <translation test> to validate")
        messagebox.showinfo("training end", "train progress has been finished")



    # translation test
    def _clear_placeholder(self, event=None):
        cur = self._infer_input.get("1.0", "end-1c").strip()
        if cur == "Please enter an English sentence here.":
            self._infer_input.delete("1.0", "end")

    def _do_translate(self):
        if self._model is None:
            messagebox.showwarning("model are not ready", "please finish training first")
            return

        sentence = self._infer_input.get("1.0", "end-1c").strip()
        if not sentence or sentence == "Please enter an English sentence here.":
            messagebox.showwarning("empty input", "please input source sentence in the input box。")
            return

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            result = tr.greedy_translate(
                self._model, sentence,
                self._src_vocab, self._tgt_vocab,
                device, max_len=tr.Config.MAX_TGT_LEN
            )
        except Exception as e:
            messagebox.showerror("fail translate", str(e))
            return

        self._infer_output.configure(state="normal")
        self._infer_output.delete("1.0", "end")
        self._infer_output.insert("end", f"source: {sentence}\n\n译文：{result}\n")
        self._infer_output.configure(state="disabled")



# entrance
def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        available = style.theme_names()
        for preferred in ("vista", "aqua", "clam"):
            if preferred in available:
                style.theme_use(preferred)
                break
    except Exception:
        pass

    app = TransformerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()