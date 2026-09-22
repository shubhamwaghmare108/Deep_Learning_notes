"""
End-to-end RNN (LSTM) pipeline for next-day stock price forecasting,
using the previous N days' OHLC (Open, High, Low, Close) as input.

Pipeline stages
----------------
1. Data ingestion   -> load_data()      (Yahoo Finance / local CSV / synthetic demo data)
2. Preprocessing    -> make_sequences() (per-feature scaling + sliding-window OHLC sequences)
3. Model            -> StockRNN         (configurable LSTM / vanilla RNN / GRU, multivariate input)
4. Training          -> train_model()
5. Evaluation        -> evaluate()      (RMSE, MAE, MAPE, R^2 on next-day Close)
6. Visualisation     -> plot_results()  (loss curve + predicted vs actual)
7. Persistence       -> saves model weights (.pt) and the fitted scaler (.pkl)

Usage
-----
# Demo run on synthetic OHLC data (no internet needed, works out of the box):
    python stock_rnn.py --demo

# Real data via Yahoo Finance (requires `pip install yfinance` + internet):
    python stock_rnn.py --ticker AAPL --start 2015-01-01 --end 2025-01-01

# From your own CSV (must contain Date, Open, High, Low, Close columns):
    python stock_rnn.py --csv my_stock.csv
"""

import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FEATURE_COLS = ["Open", "High", "Low", "Close"]
TARGET_COL = "Close"
TARGET_IDX = FEATURE_COLS.index(TARGET_COL)


# --------------------------------------------------------------------------- #
# 1. Data ingestion
# --------------------------------------------------------------------------- #
def _synthetic_stock_series(n_days: int = 2000, start_price: float = 100.0) -> pd.DataFrame:
    """Generates a plausible-looking OHLC series via geometric Brownian motion
    plus a slow seasonal wave, so the pipeline is runnable with zero external
    dependencies / network access."""
    rng = np.random.default_rng(RANDOM_SEED)
    mu, sigma = 0.0003, 0.018  # daily drift / volatility
    shocks = rng.normal(mu, sigma, n_days)
    seasonal = 0.05 * np.sin(np.linspace(0, 8 * np.pi, n_days))
    log_returns = shocks + np.diff(np.concatenate(([0], seasonal)))
    close = start_price * np.exp(np.cumsum(log_returns))

    # Build Open/High/Low around each day's close with independent noise.
    prev_close = np.concatenate(([start_price], close[:-1]))
    open_ = prev_close * (1 + rng.normal(0, 0.003, n_days))
    intraday_range = np.abs(rng.normal(0, 0.01, n_days)) * close
    high = np.maximum(open_, close) + intraday_range
    low = np.minimum(open_, close) - intraday_range

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame({"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close})


def load_data(ticker: str = None, csv_path: str = None, start: str = None,
              end: str = None, demo: bool = False) -> pd.DataFrame:
    """Returns a DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close'],
    sorted ascending."""
    if demo or (ticker is None and csv_path is None):
        print("[data] Using synthetic demo OHLC data (no ticker/csv supplied).")
        return _synthetic_stock_series()

    if csv_path is not None:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"[data] CSV missing {missing} — filling from 'Close' (close-only fallback).")
            for c in missing:
                df[c] = df["Close"]
        df = df[["Date"] + FEATURE_COLS].sort_values("Date").reset_index(drop=True)
        print(f"[data] Loaded {len(df)} rows from {csv_path}")
        return df

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("Empty download")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()[["Date"] + FEATURE_COLS]
        print(f"[data] Downloaded {len(df)} rows for {ticker} from Yahoo Finance.")
        return df
    except Exception as e:
        print(f"[data] Falling back to synthetic data — yfinance download failed ({e}).")
        return _synthetic_stock_series()


# --------------------------------------------------------------------------- #
# 2. Preprocessing
# --------------------------------------------------------------------------- #
def make_sequences(features: np.ndarray, target: np.ndarray, lookback: int):
    """features: (n, n_features) scaled OHLC. target: (n,) scaled Close.
    X[i] = features[i : i+lookback]  (past `lookback` days' OHLC)
    y[i] = target[i+lookback]        (next day's Close)"""
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)


def prepare_data(df: pd.DataFrame, lookback: int, train_split: float):
    ohlc = df[FEATURE_COLS].values  # (n, 4)
    n_train_raw = int(len(ohlc) * train_split)

    # Fit the scaler on the training portion only (per-feature min/max),
    # to avoid leaking test-set statistics into the transform.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(ohlc[:n_train_raw])
    scaled = scaler.transform(ohlc)  # (n, 4)

    target_scaled = scaled[:, TARGET_IDX]
    X, y = make_sequences(scaled, target_scaled, lookback)
    n_train = int(len(X) * train_split)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    return (to_tensor(X_train), torch.tensor(y_train, dtype=torch.float32),
            to_tensor(X_test), torch.tensor(y_test, dtype=torch.float32),
            scaler)


def inverse_target(scaled_values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transforms a 1-D array of scaled Close values back to price
    units, using the fitted per-feature scaler (Close is one of its columns)."""
    dummy = np.zeros((len(scaled_values), len(FEATURE_COLS)))
    dummy[:, TARGET_IDX] = scaled_values
    return scaler.inverse_transform(dummy)[:, TARGET_IDX]


# --------------------------------------------------------------------------- #
# 3. Model
# --------------------------------------------------------------------------- #
class StockRNN(nn.Module):
    """Configurable recurrent regressor over multivariate (OHLC) sequences.
    cell_type: 'lstm' (default), 'gru', or 'rnn'."""

    def __init__(self, input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2,
                 dropout=0.2, cell_type="lstm"):
        super().__init__()
        cell_type = cell_type.lower()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[cell_type]
        rnn_kwargs = dict(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.rnn = rnn_cls(**rnn_kwargs)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.rnn(x)          # out: (batch, seq_len, hidden_size)
        last_step = out[:, -1, :]     # take the final time step's hidden state
        return self.head(last_step).squeeze(-1)


# --------------------------------------------------------------------------- #
# 4. Training
# --------------------------------------------------------------------------- #
def train_model(model, X_train, y_train, X_val, y_val, epochs=60,
                 batch_size=32, lr=1e-3, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []
    best_val, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(dataset)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[train] epoch {epoch:3d}/{epochs}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

        if bad_epochs >= patience:
            print(f"[train] early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


# --------------------------------------------------------------------------- #
# 5. Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test).numpy()
    true_scaled = y_test.numpy()

    pred = inverse_target(pred_scaled, scaler)
    true = inverse_target(true_scaled, scaler)

    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    r2 = r2_score(true, pred)

    print(f"\n[eval] RMSE : {rmse:.4f}")
    print(f"[eval] MAE  : {mae:.4f}")
    print(f"[eval] MAPE : {mape:.2f}%")
    print(f"[eval] R^2  : {r2:.4f}")
    return true, pred, dict(rmse=rmse, mae=mae, mape=mape, r2=r2)


# --------------------------------------------------------------------------- #
# 6. Visualisation
# --------------------------------------------------------------------------- #
def plot_results(train_losses, val_losses, true, pred, out_path="stock_rnn_results.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="train MSE")
    axes[0].plot(val_losses, label="val MSE")
    axes[0].set_title("Training Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (scaled)")
    axes[0].legend()

    axes[1].plot(true, label="Actual", linewidth=1.5)
    axes[1].plot(pred, label="Predicted", linewidth=1.5, alpha=0.8)
    axes[1].set_title("Predicted vs Actual Next-Day Close (test set)")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Price")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot] saved -> {out_path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="End-to-end RNN stock price forecaster (OHLC -> next-day Close)")
    p.add_argument("--ticker", type=str, default=None, help="e.g. AAPL, MSFT (needs yfinance + internet)")
    p.add_argument("--csv", type=str, default=None, help="Local CSV with Date, Open, High, Low, Close columns")
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--demo", action="store_true", help="Force synthetic demo data")
    p.add_argument("--lookback", type=int, default=30, help="Number of past days' OHLC used as input")
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--cell", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--out-prefix", type=str, default="stock_rnn")
    args = p.parse_args()

    df = load_data(ticker=args.ticker, csv_path=args.csv, start=args.start,
                    end=args.end, demo=args.demo)

    X_train, y_train, X_test, y_test, scaler = prepare_data(
        df, lookback=args.lookback, train_split=args.train_split
    )
    print(f"[data] train sequences: {X_train.shape[0]}  test sequences: {X_test.shape[0]}  "
          f"lookback: {args.lookback}  features/day: {X_train.shape[-1]} (OHLC)")

    model = StockRNN(hidden_size=args.hidden_size, num_layers=args.layers,
                      dropout=args.dropout, cell_type=args.cell)

    model, train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
    )

    true, pred, metrics = evaluate(model, X_test, y_test, scaler)
    plot_results(train_losses, val_losses, true, pred, out_path=f"{args.out_prefix}_results.png")

    torch.save(model.state_dict(), f"{args.out_prefix}_model.pt")
    with open(f"{args.out_prefix}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"[save] model weights -> {args.out_prefix}_model.pt")
    print(f"[save] scaler        -> {args.out_prefix}_scaler.pkl")

    return metrics


if __name__ == "__main__":
    main()
