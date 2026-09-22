"""
End-to-end RNN (LSTM) pipeline for stock closing-price forecasting.

Pipeline stages
----------------
1. Data ingestion   -> load_data()      (Yahoo Finance / local CSV / synthetic demo data)
2. Preprocessing    -> make_sequences() (scaling + sliding-window sequence construction)
3. Model            -> StockRNN         (configurable LSTM / vanilla RNN / GRU)
4. Training          -> train_model()
5. Evaluation        -> evaluate()      (RMSE, MAE, MAPE, R^2)
6. Visualisation     -> plot_results()  (loss curve + predicted vs actual)
7. Persistence       -> saves model weights (.pt) and the fitted scaler (.pkl)

Usage
-----
# Demo run on synthetic data (no internet needed, works out of the box):
    python stock_rnn.py --demo

# Real data via Yahoo Finance (requires `pip install yfinance` + internet):
    python stock_rnn.py --ticker AAPL --start 2015-01-01 --end 2025-01-01

# From your own CSV (must contain a 'Date' and 'Close' column):
    python stock_rnn.py --csv my_stock.csv
"""

import argparse
import pickle
import sys

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


# --------------------------------------------------------------------------- #
# 1. Data ingestion
# --------------------------------------------------------------------------- #
def _synthetic_stock_series(n_days: int = 2000, start_price: float = 100.0) -> pd.DataFrame:
    """Generates a plausible-looking stock price series via geometric Brownian
    motion plus a slow seasonal wave, so the pipeline is runnable with zero
    external dependencies / network access."""
    rng = np.random.default_rng(RANDOM_SEED)
    dt = 1.0
    mu, sigma = 0.0003, 0.018  # daily drift / volatility
    shocks = rng.normal(mu * dt, sigma * np.sqrt(dt), n_days)
    seasonal = 0.05 * np.sin(np.linspace(0, 8 * np.pi, n_days))
    log_returns = shocks + np.diff(np.concatenate(([0], seasonal)))
    prices = start_price * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame({"Date": dates, "Close": prices})


def load_data(ticker: str = None, csv_path: str = None, start: str = None,
              end: str = None, demo: bool = False) -> pd.DataFrame:
    """Returns a DataFrame with columns ['Date', 'Close'], sorted ascending."""
    if demo or (ticker is None and csv_path is None):
        print("[data] Using synthetic demo data (no ticker/csv supplied).")
        return _synthetic_stock_series()

    if csv_path is not None:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df[["Date", "Close"]].sort_values("Date").reset_index(drop=True)
        print(f"[data] Loaded {len(df)} rows from {csv_path}")
        return df

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("Empty download")
        df = df.reset_index()[["Date", "Close"]]
        print(f"[data] Downloaded {len(df)} rows for {ticker} from Yahoo Finance.")
        return df
    except Exception as e:
        print(f"[data] Falling back to synthetic data — yfinance download failed ({e}).")
        return _synthetic_stock_series()


# --------------------------------------------------------------------------- #
# 2. Preprocessing
# --------------------------------------------------------------------------- #
def make_sequences(series: np.ndarray, lookback: int):
    """Turns a 1-D price array into overlapping (X, y) windows for
    next-step-ahead prediction: X[i] = series[i : i+lookback], y[i] = series[i+lookback]."""
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


def prepare_data(df: pd.DataFrame, lookback: int, train_split: float):
    close = df["Close"].values.reshape(-1, 1)
    n_train_raw = int(len(close) * train_split)

    # Fit the scaler on the training portion only, to avoid leaking test-set
    # statistics into the transform.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close[:n_train_raw])
    scaled = scaler.transform(close).flatten()

    X, y = make_sequences(scaled, lookback)
    n_train = int(len(X) * train_split)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
    return (to_tensor(X_train), torch.tensor(y_train, dtype=torch.float32),
            to_tensor(X_test), torch.tensor(y_test, dtype=torch.float32),
            scaler)


# --------------------------------------------------------------------------- #
# 3. Model
# --------------------------------------------------------------------------- #
class StockRNN(nn.Module):
    """Configurable recurrent regressor: 'lstm' (default), 'gru', or 'rnn'."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
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
        pred_scaled = model(X_test).numpy().reshape(-1, 1)
    true_scaled = y_test.numpy().reshape(-1, 1)

    pred = scaler.inverse_transform(pred_scaled).flatten()
    true = scaler.inverse_transform(true_scaled).flatten()

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
    axes[1].set_title("Predicted vs Actual Close Price (test set)")
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
    p = argparse.ArgumentParser(description="End-to-end RNN stock price forecaster")
    p.add_argument("--ticker", type=str, default=None, help="e.g. AAPL, MSFT (needs yfinance + internet)")
    p.add_argument("--csv", type=str, default=None, help="Local CSV with Date, Close columns")
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--demo", action="store_true", help="Force synthetic demo data")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--cell", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-prefix", type=str, default="stock_rnn")
    args = p.parse_args()

    df = load_data(ticker=args.ticker, csv_path=args.csv, start=args.start,
                    end=args.end, demo=args.demo)

    X_train, y_train, X_test, y_test, scaler = prepare_data(
        df, lookback=args.lookback, train_split=args.train_split
    )
    print(f"[data] train sequences: {X_train.shape[0]}  test sequences: {X_test.shape[0]}  "
          f"lookback: {args.lookback}")

    model = StockRNN(hidden_size=args.hidden_size, num_layers=args.layers,
                      dropout=args.dropout, cell_type=args.cell)

    model, train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
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
