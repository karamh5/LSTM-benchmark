"""Jena climate temperature forecasting with an LSTM.
Baseline = official Keras example (keras.io/examples/timeseries/timeseries_weather_forecasting).
Predicts temperature 72 steps (12 h) ahead from the previous 720 steps (120 h), sampled hourly."""
import argparse, json, time, os
import numpy as np, pandas as pd, keras

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--hidden", type=int, default=32)
p.add_argument("--lstm_layers", type=int, default=1)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--past", type=int, default=720)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
keras.utils.set_random_seed(a.seed)
D = os.path.join(os.path.dirname(__file__), "..", "data")

CSV = f"{D}/jena_climate_2009_2016.csv"
if not os.path.exists(CSV):   # first run on a fresh machine (e.g. Colab): fetch the official Keras copy
    from zipfile import ZipFile
    z = keras.utils.get_file(origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip", fname="jena_climate_2009_2016.csv.zip")
    ZipFile(z).extractall(D)
df = pd.read_csv(CSV)
df = df.rename(columns={"Date Time": "Date Time"})
keys = ["p (mbar)","T (degC)","Tpot (K)","Tdew (degC)","rh (%)","VPmax (mbar)","VPact (mbar)","VPdef (mbar)","sh (g/kg)","H2OC (mmol/mol)","rho (g/m**3)","wv (m/s)","max. wv (m/s)","wd (deg)"]
sel = [keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]       # same 7 features as the Keras example
split = int(0.715 * df.shape[0]); step = 6; future = 72
raw = df[sel].values
mean, std = raw[:split].mean(0), raw[:split].std(0)
feat = pd.DataFrame((raw - mean) / std)
train, val = feat.loc[0:split - 1], feat.loc[split:]
start = a.past + future
seq_len = int(a.past / step)
ds_train = keras.preprocessing.timeseries_dataset_from_array(
    train.values, feat.iloc[start:start + split][[1]], sequence_length=seq_len, sampling_rate=step, batch_size=a.batch)
x_end = len(val) - a.past - future
ds_val = keras.preprocessing.timeseries_dataset_from_array(
    val.iloc[:x_end].values, feat.iloc[split + a.past + future:][[1]], sequence_length=seq_len, sampling_rate=step, batch_size=a.batch)
for xb, yb in ds_train.take(1): shape = xb.shape[1:]

inp = keras.layers.Input(shape=shape); x = inp
for i in range(a.lstm_layers):
    x = keras.layers.LSTM(a.hidden, return_sequences=i < a.lstm_layers - 1, dropout=a.dropout)(x)
model = keras.Model(inp, keras.layers.Dense(1)(x))
model.compile(optimizer=keras.optimizers.Adam(a.lr), loss="mse")
t0 = time.time()
h = model.fit(ds_train, epochs=a.epochs, validation_data=ds_val, verbose=2, callbacks=[
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
tt = time.time() - t0
# metrics in real units (degC) on the validation set
pred = model.predict(ds_val, verbose=0).flatten()
true = np.concatenate([y.numpy().flatten() for _, y in ds_val])
t_std = std[1]
mae_c = float(np.mean(np.abs(pred - true)) * t_std); rmse_c = float(np.sqrt(np.mean((pred - true) ** 2)) * t_std)
hist = h.history; best = int(np.argmin(hist["val_loss"]))
res = dict(name=a.name, config=vars(a), params=int(model.count_params()), epochs_run=len(hist["loss"]),
           best_epoch=best + 1, train_mse=hist["loss"][best], val_mse=hist["val_loss"][best],
           val_mae_degC=mae_c, val_rmse_degC=rmse_c, train_time_s=round(tt, 1), history=hist)
json.dump(res, open(f"{D}/../results/jena_{a.name}.json", "w"), default=float)
print("RESULT", a.name, "params", res["params"], "val_mse", round(res["val_mse"], 4), "MAE degC", round(mae_c, 3), "time", res["train_time_s"])
