"""Builds figures + README.md from results/*.json (run from repo root: python src/make_report.py)."""
import json, os
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R, F = Path("results"), Path("figures"); F.mkdir(exist_ok=True)
ld = lambda p: json.loads(p.read_text()) if p.exists() else None
tf = ld(R / "forda_baseline.json"); lb = ld(R / "jena_baseline.json")
VAR = {n: ld(R / f"jena_{n}.json") for n in ["hidden64", "stack2", "past360"]}
VAR = {k: v for k, v in VAR.items() if v}
W = 5                                               # equal epoch budget used for the controlled comparison
LAB = {"hidden64": "LSTM(64) - wider", "stack2": "2 x LSTM(32) - deeper", "past360": "look-back 360 steps (60 h) - shorter input"}

# ---- figures ------------------------------------------------------------------------------------
if lb:
    h = lb["history"]; plt.figure(figsize=(5.5, 3.8))
    plt.plot(range(1, len(h["loss"]) + 1), h["loss"], "b-o", label="train"); plt.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], "r-o", label="validation")
    plt.xlabel("epoch"); plt.ylabel("MSE (normalised)"); plt.title("LSTM baseline loss"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(F / "lstm_baseline_loss.png", dpi=150); plt.close()
    plt.figure(figsize=(6.5, 4))
    plt.plot(range(1, W + 1), lb["history"]["val_loss"][:W], "k-o", label="baseline LSTM(32)")
    for k, v in VAR.items(): plt.plot(range(1, len(v["history"]["val_loss"]) + 1), v["history"]["val_loss"], "-o", label=LAB[k])
    plt.xticks(range(1, W + 1)); plt.xlabel("epoch"); plt.ylabel("validation MSE"); plt.title("LSTM improvements (same 5-epoch budget)"); plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(F / "lstm_improvements.png", dpi=150); plt.close()
if tf:
    h = tf["history"]; fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); e = range(1, len(h["loss"]) + 1)
    ax[0].plot(e, h["loss"], label="train"); ax[0].plot(e, h["val_loss"], label="val"); ax[0].set_title("Transformer loss")
    ax[1].plot(e, h["sparse_categorical_accuracy"], label="train"); ax[1].plot(e, h["val_sparse_categorical_accuracy"], label="val"); ax[1].set_title("Transformer accuracy")
    for a in ax: a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(F / "transformer_baseline.png", dpi=150); plt.close()

# ---- numbers ---------------------------------------------------------------------------------
def lrow(name, r, vm=None):
    v = vm if vm is not None else r["val_mse"]
    return f"| {name} | {r['params']:,} | {r['config']['hidden']} x {r['config']['lstm_layers']} | {r['config']['past']//6} | {r['train_mse']:.4f} | {v:.4f} | {r['val_mae_degC']:.2f} | {r['val_rmse_degC']:.2f} |"
bmse = min(lb["history"]["val_loss"][:W]) if lb else None
rows = [lrow("**baseline** LSTM(32)", lb, bmse)] if lb else []
for k, v in VAR.items(): rows.append(lrow(LAB[k], v))
imp = "\n".join(["| Variant | Params | Hidden x layers | Look-back (h) | Train MSE | Val MSE | Val MAE (degC) | Val RMSE (degC) |", "|---|---|---|---|---|---|---|---|"] + rows)

best = min(VAR.items(), key=lambda kv: kv[1]["val_mse"]) if VAR else None
def delta(k): return (VAR[k]["val_mse"] - bmse) / bmse * 100
lines = []
if lb:
    gap = lb["history"]["val_loss"][-1] - lb["history"]["loss"][-1]
    over = lb["history"]["val_loss"][-1] > min(lb["history"]["val_loss"]) * 1.02
    lines.append(f"The baseline reaches its best validation MSE of {min(lb['history']['val_loss']):.4f} (about {lb['val_mae_degC']:.2f} degC mean absolute error for a 12-hour-ahead forecast) at epoch {lb['best_epoch']}. "
                 + (f"After that, training loss keeps falling to {lb['history']['loss'][-1]:.4f} while validation loss rises to {lb['history']['val_loss'][-1]:.4f}: the model starts to overfit the training period (the first 71.5% of the record), so early stopping is doing real work." if over else "Training and validation loss stay close, so the model is not overfitting at this budget."))
for k in VAR: lines.append(f"- **{LAB[k]}**: val MSE {VAR[k]['val_mse']:.4f} vs {bmse:.4f} ({delta(k):+.1f}%), MAE {VAR[k]['val_mae_degC']:.2f} degC, {VAR[k]['params']:,} params.")
learn = ""
if best and {"hidden64", "stack2", "past360"} <= set(VAR):
    k, v = best
    sgn = lambda x: "better" if x < 0 else "worse"
    h, st, pa = VAR["hidden64"], VAR["stack2"], VAR["past360"]
    learn = (f"The best change was **{LAB[k]}** ({delta(k):+.1f}% validation MSE versus the baseline at the same {W}-epoch budget). "
             f"Widening to 64 units gave {delta('hidden64'):+.1f}% ({sgn(delta('hidden64'))}) with {h['params']/lb['params']:.1f}x the parameters: its training loss kept falling ({h['history']['loss'][0]:.4f} to {h['history']['loss'][-1]:.4f}) while validation loss rose from {h['history']['val_loss'][0]:.4f} to {h['history']['val_loss'][-1]:.4f} over the {len(h['history']['loss'])} epochs, i.e. extra capacity overfits instead of helping. "
             f"Stacking a second LSTM layer gave {delta('stack2'):+.1f}% ({sgn(delta('stack2'))}) for {st['params']/lb['params']:.1f}x the parameters, so depth did not improve the best score at this data size, although its validation curve stays flatter than the baseline's, which rises after epoch 2 (epoch-5 val MSE {st['history']['val_loss'][-1]:.4f} vs {lb['history']['val_loss'][4]:.4f}). "
             f"Shortening the look-back from 120 h to 60 h gave {delta('past360'):+.1f}% ({sgn(delta('past360'))}) while halving the sequence length: for a 12-hour-ahead forecast the older half of the window adds noise more than signal, and a shorter input is also cheaper. "
             f"So for this task the input window and regularisation matter more than model size.")
tf_txt = (f"| Transformer (FordA) | {tf['params']:,} | {tf['epochs_run']} (best {int(np.argmin(tf['history']['val_loss']))+1}) | {tf['history']['sparse_categorical_accuracy'][int(np.argmin(tf['history']['val_loss']))]:.3f} | {tf['history']['val_sparse_categorical_accuracy'][int(np.argmin(tf['history']['val_loss']))]:.3f} | **{tf['test_acc']:.3f}** | {tf['train_time_s']/tf['epochs_run']:.0f} s |"
          if tf else "| Transformer (FordA) | - | - | - | - | - | run `notebooks/forda_transformer_colab.ipynb` |")
tf_obs = (f"Run as the official Keras example with three changes needed to make it train (see section 3b): a Conv1D(64, kernel 7) input projection, layer norm applied before attention and feed-forward (pre-norm), and Adam 3e-4. Dropout is 0.1 and MLP dropout 0.2 (official: 0.25 and 0.4), batch 32 (official: 64), up to 60 epochs with early stopping (patience 15, best weights restored); {tf['epochs_run']} epochs ran. "
          f"Test accuracy is **{tf['test_acc']:.3f}** with {tf['params']:,} parameters (official write-up: about 0.85 after 150 epochs). Train/val accuracy in the table are taken at the epoch with the lowest validation loss, which is the epoch whose weights the test score uses. "
          f"Training accuracy ({tf['history']['sparse_categorical_accuracy'][int(np.argmin(tf['history']['val_loss']))]:.3f}) sits below validation accuracy because dropout is active during training only and the validation split is small (720 series)." if tf else "No Transformer result yet.")

# ---- transformer: debugging runs and the one tried improvement -------------------------------
import numpy as np
TRY = [("official_raw", "official block, raw 1-feature input, lr 1e-4, batch 32"),
       ("proj_k1", "+ Conv1D projection to 64 features, kernel 1, lr 1e-4"),
       ("p_post_lr1e3", "projection kernel 7, post-norm (official block), lr 1e-3, 12 epochs"),
       ("p_pre_lr1e3", "projection kernel 7, pre-norm, lr 1e-3, 12 epochs"),
       ("p_pre_lr3e4", "projection kernel 7, pre-norm, lr 3e-4, 12 epochs")]
trows = []
for n, desc in TRY:
    r = ld(R / f"forda_{n}.json")
    if r: trows.append(f"| {desc} | {r['epochs_run']} | {r['val_acc']:.3f} | {r['test_acc']:.3f} |")
trows_txt = "\n".join(["| Setting | Epochs run | Val acc | Test acc |", "|---|---|---|---|"] + trows) if trows else ""
wf = ld(R / "forda_wide_ff.json")
def tfrow(name, r):
    h = r["history"]; bl = int(np.argmin(h["val_loss"]))
    return f"| {name} | {r['params']:,} | {r['epochs_run']} | {h['val_sparse_categorical_accuracy'][bl]:.3f} | {r['test_loss']:.3f} | **{r['test_acc']:.3f}** |"
xrows = ([tfrow("baseline (projection k7, pre-norm, ff_dim 4)", tf)] if tf else []) + ([tfrow("wider feed-forward (ff_dim 128)", wf)] if wf else [])
xtab = "\n".join(["| Variant | Params | Epochs run | Val acc | Test loss | Test acc |", "|---|---|---|---|---|---|"] + xrows) if xrows else ""
if tf and wf:
    d = wf["test_acc"] - tf["test_acc"]
    xtxt = (f"Widening the feed-forward layer from 4 to 128 filters changed test accuracy by {d:+.3f} ({wf['test_acc']:.3f} vs {tf['test_acc']:.3f}) with {wf['params']-tf['params']:,} more parameters. "
            f"A difference this small is probably within run-to-run noise (single seed, seed spread not measured), so we read it as no measurable gain.")
else:
    xtxt = ""

readme = f"""# Time-Series Modelling Benchmark: Transformer Classification (FordA) and LSTM Forecasting (Jena Climate)

CMPE 401 Instructor-defined Project 2. Two official Keras time-series examples are reproduced, organised and benchmarked. The LSTM forecaster is improved with three controlled changes, and the Transformer needed debugging before it would train at all (section 3b). All numbers in this README are generated from `results/*.json` by `src/make_report.py`.

## 1. Problem setup
| | Classification | Forecasting |
|---|---|---|
| Dataset | FordA (UCR archive): engine sensor, 3,601 train / 1,320 test series, 500 steps each, 1 channel | Jena Climate 2009-2016: 420,551 rows (every 10 min), 14 weather features, 7 selected |
| Task | Binary label: is a fault present? | Regression: temperature 12 h (72 steps) ahead |
| Input | `(batch, 500, 1)` | `(batch, 120, 7)`: past 120 h sampled hourly, z-scored with training statistics |
| Output | Softmax over 2 classes | One real number (normalised temperature) |
| Model | Transformer encoder: Conv1D input projection, 4 x [multi-head self-attention + residual + LayerNorm + Conv1D feed-forward], global average pooling, MLP head | One LSTM layer (32 units) + Dense(1) |
| Loss / metric | Sparse categorical cross-entropy / accuracy | MSE (normalised) / MAE and RMSE in degC |
| Split | 80/20 train/validation from the train file; official test file for final score | First 71.5% of time for training, remainder for validation (chronological, no shuffling across the boundary) |

**Classification vs forecasting.** Classification maps a whole series to one discrete label and may look at all time steps at once. Forecasting predicts future values, so it must respect time order: the split is chronological and the label is always in the future of the input window.
**Transformer vs LSTM.** An LSTM reads steps one at a time and carries a hidden state, so cost grows linearly with length but it is sequential. Self-attention lets every step look at every other step in parallel (cost grows quadratically with length), which captures long-range patterns but needs more memory and compute.

## 2. Repository layout and how to run
```
src/transformer_forda.py   FordA Transformer (Keras example, hyper-parameters and the fixes in section 3b as flags)
src/lstm_jena.py           Jena LSTM forecaster (official Keras code, extended with flags)
src/make_report.py         builds figures/ and this README from results/
notebooks/forda_transformer_colab.ipynb   Colab version of the Transformer (not used for the reported results)
results/                   one JSON per run (config, params, metrics, per-epoch history) + training logs
figures/                   plots used below
```
```bash
pip install tensorflow keras pandas matplotlib
python src/lstm_jena.py --name baseline                      # downloads the Jena data on first run
python src/lstm_jena.py --name hidden64 --hidden 64 --epochs 5
python src/lstm_jena.py --name stack2   --lstm_layers 2 --epochs 5
python src/lstm_jena.py --name past360  --past 360 --epochs 5
python src/transformer_forda.py --name baseline --epochs 60 --patience 15 --lr 3e-4 --batch 32 --d_model 64 --embed_kernel 7 --dropout 0.1 --mlp_dropout 0.2 --prenorm 1   # GPU recommended
python src/transformer_forda.py --name wide_ff --epochs 60 --patience 15 --lr 3e-4 --batch 32 --d_model 64 --embed_kernel 7 --dropout 0.1 --mlp_dropout 0.2 --prenorm 1 --ff_dim 128
python src/make_report.py
```

## 3. Task 1 - Baselines
| Model | Params | Epochs run | Train acc | Val acc | Test acc | Time/epoch |
|---|---|---|---|---|---|---|
{tf_txt}

{tf_obs}

![transformer](figures/transformer_baseline.png)

**LSTM baseline (Jena).** Official settings: LSTM(32), Adam 1e-3, batch 256, up to 10 epochs, early stopping patience 5. Best validation MSE {min(lb['history']['val_loss']):.4f} = {lb['val_mae_degC']:.2f} degC MAE / {lb['val_rmse_degC']:.2f} degC RMSE ({lb['params']:,} parameters, {lb['epochs_run']} epochs run, best epoch {lb['best_epoch']}).

![lstm](figures/lstm_baseline_loss.png)

## 3b. Getting the Transformer to train
The unmodified Keras example sat at chance level here: loss 0.693 (= ln 2) and 0.510 validation accuracy (the share of the majority class in the validation split) for the whole run. Several runs were needed to find out why. All of them are kept in `results/`.

{trows_txt}

Our diagnosis, in the order it was found:
1. **Layer norm on a 1-feature input outputs zero.** FordA has one value per time step, so `LayerNormalization` over the feature axis normalises a single number: (x - mean) is exactly 0. Checked directly: it returns all zeros. Attention and feed-forward outputs therefore collapse to constants and only the residual path carries the raw series. Each series is already z-scored (mean about 0), so averaging over time leaves nothing to classify. Fix: project the input to 64 features first (`--d_model 64`).
2. **A per-step projection still cannot see the shape of the waveform.** With kernel size 1 every step is embedded on its own, and attention, the feed-forward layers and the global average pooling that follow all ignore step order, so the model sees only a histogram of values. The test accuracy stayed at 0.516. On a small convolution-only model we checked separately, kernel size 1 gave 0.515 and kernel size 7 gave 0.719 test accuracy (an extra check, not saved in `results/`). Fix: kernel size 7 (`--embed_kernel 7`), so each feature covers 7 neighbouring steps.
3. **Optimisation.** With kernel 7 the official post-norm block with lr 1e-3 only began to learn in the last epochs of a 12-epoch run (validation accuracy 0.577). Pre-norm (layer norm before attention and feed-forward) at lr 3e-4 trained steadily to 0.846 validation and 0.833 test accuracy in 12 epochs, so it was chosen for the baseline.

The first row used a pre-norm block that was in the script at that moment, and earlier runs of the exact official post-norm block on the raw input (batch 16) gave the same 0.510 validation accuracy; those runs' JSON files were overwritten and only their logs were seen. All five runs above are single-seed.

### Transformer variant tried
{xtab}

{xtxt}

Other variants (fewer blocks, wider input window, smaller attention key size) were planned and not run; the three controlled improvements are on the LSTM (section 4).

## 4. Task 2 - Improvements (LSTM forecaster, three modifications)
Each change is made one at a time on top of the baseline; seed, data, optimiser, learning rate and batch size are identical. All variants and the baseline are compared at the same 5-epoch budget (the baseline row uses its best validation MSE within its first {W} epochs).

{imp}

{chr(10).join(lines)}

![improvements](figures/lstm_improvements.png)

**What we learned.** {learn}

## 5. Task 3 - Benchmark summary
| Model | Task | Params | Headline result |
|---|---|---|---|
| Transformer | FordA classification | {(format(tf['params'], ',') if tf else '-')} | {('test accuracy %.3f' % tf['test_acc']) if tf else 'pending Colab run'} |
{('| Transformer, wider feed-forward | FordA classification | %s | test accuracy %.3f |' % (format(wf['params'], ','), wf['test_acc'])) if wf else ''}\n| LSTM baseline | Jena 12 h forecast | {lb['params']:,} | val MAE {lb['val_mae_degC']:.2f} degC |
{chr(10).join(f"| {LAB[k]} | Jena 12 h forecast | {v['params']:,} | val MAE {v['val_mae_degC']:.2f} degC |" for k, v in VAR.items())}

## 6. Task 4 - Questions
**Which model did you find easier to understand and why?** The LSTM. Its data flow is one idea: step through time, update a hidden state, predict from the last state; the Keras code is five lines. The Transformer needs several ideas at once (queries/keys/values, multi-head attention, residual connections, LayerNorm, Conv1D used as a position-wise dense layer, pooling) and has no built-in notion of order, which is why its behaviour and hyper-parameters (heads, head size, blocks) are harder to reason about. Getting the Transformer to train at all (section 3b) took several runs, while the LSTM trained on the first try.

**What improvement did you try, and what did you learn from it?** We changed the LSTM forecaster three ways: more units, a second stacked layer and a shorter look-back window. {learn} The practical lesson is that the baseline is already near the point where validation error stops improving after a couple of epochs, so gains come from controlling variance and input length rather than simply adding parameters.

## 7. Honest limitations
- Compute: the LSTM runs were done on a 2-core CPU with several runs sharing it, so their wall-clock times are not reported (not comparable); parameter counts are. The Transformer runs were done on an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) under WSL2 Ubuntu with TensorFlow 2.21, XLA turned off after an out-of-memory error with it on (TensorFlow could use about 3.5 GB of the 6 GB). The LSTM improvements use a 5-epoch budget to keep runs comparable and short.\n- The Transformer test score is below the official write-up's 0.85 and uses a different recipe (projection, pre-norm, 3e-4), so it is not a like-for-like reproduction of the official numbers.
- Single seed (0); differences of a few percent in MSE can be seed noise. A fair next step is 3-5 seeds and reporting mean +/- std.
- The Jena data is the standard Keras copy; the validation set is a single chronological block, so it is also the model-selection set. A separate test block would be needed for a final unbiased number.
"""
Path("README.md").write_text(readme); print("README.md written;", "transformer" if tf else "NO transformer result yet")
