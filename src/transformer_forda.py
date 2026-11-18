"""FordA time-series classification with a Transformer encoder.
Baseline = official Keras example (keras.io/examples/timeseries/timeseries_classification_transformer).
All hyper-parameters exposed as flags so each experiment is one controlled change."""
import argparse, json, time, os
import numpy as np, keras
from keras import layers

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--blocks", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--head_size", type=int, default=256)
p.add_argument("--ff_dim", type=int, default=4)
p.add_argument("--mlp_units", type=int, nargs="*", default=[128])
p.add_argument("--dropout", type=float, default=0.25)
p.add_argument("--mlp_dropout", type=float, default=0.4)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--d_model", type=int, default=0, help="0 = official (raw 1-feature input); >0 = Conv1D input projection to d_model features")
p.add_argument("--embed_kernel", type=int, default=1)
p.add_argument("--prenorm", type=int, default=0, help="1 = layer norm before attention/feed-forward (pre-norm) instead of the official post-norm")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
keras.utils.set_random_seed(a.seed)
D = os.path.join(os.path.dirname(__file__), "..", "data")

def readucr(f):
    d = np.loadtxt(f, delimiter="\t"); return d[:, 1:], d[:, 0].astype(int)
ROOT = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
import urllib.request
os.makedirs(D, exist_ok=True)
for f in ("FordA_TRAIN.tsv", "FordA_TEST.tsv"):          # fetch the official Keras-example data if not present
    if not os.path.exists(f"{D}/{f}"): urllib.request.urlretrieve(ROOT + f, f"{D}/{f}")
x_train, y_train = readucr(f"{D}/FordA_TRAIN.tsv")
x_test, y_test = readucr(f"{D}/FordA_TEST.tsv")
x_train = x_train[..., None]; x_test = x_test[..., None]
n_classes = len(np.unique(y_train))
idx = np.random.RandomState(a.seed).permutation(len(x_train))
x_train, y_train = x_train[idx], y_train[idx]
y_train[y_train == -1] = 0; y_test[y_test == -1] = 0

def encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    if a.prenorm:   # pre-norm variant (normalise first, then sub-layer, then residual add)
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
    # default: identical to the official Keras example's transformer_encoder (post-norm)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build(input_shape):
    inp = keras.Input(shape=input_shape); x = inp
    if a.d_model: x = layers.Conv1D(a.d_model, a.embed_kernel, padding="same")(x)
    for _ in range(a.blocks):
        x = encoder(x, a.head_size, a.heads, a.ff_dim, a.dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for d in a.mlp_units:
        x = layers.Dense(d, activation="relu")(x); x = layers.Dropout(a.mlp_dropout)(x)
    return keras.Model(inp, layers.Dense(n_classes, activation="softmax")(x))

model = build(x_train.shape[1:])
model.compile(jit_compile=False, loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(a.lr),
              metrics=["sparse_categorical_accuracy"])
t0 = time.time()
h = model.fit(x_train, y_train, validation_split=0.2, epochs=a.epochs, batch_size=a.batch, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(patience=a.patience, restore_best_weights=True)])
train_time = time.time() - t0
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
hist = h.history; best = int(np.argmax(hist["val_sparse_categorical_accuracy"]))
res = dict(name=a.name, config=vars(a), params=int(model.count_params()), epochs_run=len(hist["loss"]),
           best_epoch=best + 1, train_acc=hist["sparse_categorical_accuracy"][best],
           val_acc=hist["val_sparse_categorical_accuracy"][best], test_acc=float(test_acc),
           test_loss=float(test_loss), train_time_s=round(train_time, 1), history=hist)
os.makedirs(f"{D}/../results", exist_ok=True)
json.dump(res, open(f"{D}/../results/forda_{a.name}.json", "w"), default=float)
print("RESULT", a.name, "params", res["params"], "val", round(res["val_acc"], 4), "test", round(res["test_acc"], 4), "time", res["train_time_s"])
