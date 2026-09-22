STOCK RNN (LSTM) FORECASTER — README
=====================================

WHAT THIS IS
------------
stock_rnn.py is a self-contained, end-to-end pipeline that trains a
recurrent neural network (LSTM by default, GRU/vanilla RNN optional)
to predict next-day stock closing prices from historical price
sequences.

It handles: data loading -> preprocessing -> model training ->
evaluation -> plotting -> saving the trained model.


1. REQUIREMENTS
----------------
- Python 3.9+
- pip packages:
    torch
    scikit-learn
    pandas
    matplotlib
    yfinance      (optional — only needed if you fetch live data)

Install everything with:

    pip install torch scikit-learn pandas matplotlib yfinance

(If you're on a machine with a GPU and want faster training, install
the CUDA build of torch instead — see https://pytorch.org/get-started)


2. FILES
--------
    stock_rnn.py                Main script (run this)
    stock_rnn_results.png       Output plot (created after running)
    stock_rnn_model.pt          Saved trained model weights (created after running)
    stock_rnn_scaler.pkl        Saved fitted MinMaxScaler (created after running)


3. HOW TO RUN
--------------

Option A — Quick demo, no internet or ticker needed
(uses synthetic generated price data so you can test the pipeline
immediately):

    python stock_rnn.py --demo


Option B — Real stock data from Yahoo Finance
(requires internet access and the yfinance package):

    python stock_rnn.py --ticker AAPL --start 2015-01-01 --end 2025-01-01

Replace AAPL with any valid ticker symbol, e.g. MSFT, TSLA, GOOGL.


Option C — Your own CSV file
CSV must contain at least two columns named exactly "Date" and "Close":

    Date,Close
    2020-01-02,74.33
    2020-01-03,73.88
    ...

Run with:

    python stock_rnn.py --csv path/to/your_file.csv


4. USEFUL OPTIONAL FLAGS
-------------------------
    --lookback 60        Number of past days used to predict the next day
    --train-split 0.8    Fraction of data used for training (rest = test)
    --cell lstm           Choose recurrent cell: lstm | gru | rnn
    --hidden-size 64      Size of the RNN hidden state
    --layers 2             Number of stacked RNN layers
    --dropout 0.2          Dropout between stacked layers
    --epochs 60             Max training epochs (early stopping may end sooner)
    --batch-size 32         Mini-batch size
    --lr 0.001              Learning rate
    --out-prefix stock_rnn  Prefix used for saved plot/model/scaler filenames

Example, combining several:

    python stock_rnn.py --ticker TSLA --start 2018-01-01 --cell gru \
        --lookback 90 --epochs 100 --hidden-size 128


5. WHAT HAPPENS WHEN YOU RUN IT
---------------------------------
1. Loads price data (Yahoo Finance, your CSV, or synthetic demo data).
2. Scales prices to [0, 1] using only the training portion (no data
   leakage into the test set).
3. Builds sliding-window sequences (default: 60 past days -> next day).
4. Trains the RNN with Adam + MSE loss, gradient clipping, a
   learning-rate scheduler, and early stopping.
5. Evaluates on the held-out test set, printing RMSE, MAE, MAPE, R^2.
6. Saves a PNG with the training curve and predicted-vs-actual prices.
7. Saves the trained model weights and the fitted scaler so you can
   reload them later for inference without retraining.


6. TROUBLESHOOTING
-------------------
- "No module named yfinance": only needed for --ticker mode. Either
  `pip install yfinance` or use --demo / --csv instead.
- yfinance download fails / returns empty: check your internet
  connection and that the ticker symbol is correct. The script will
  automatically fall back to synthetic demo data if the download
  fails, so it won't crash.
- Training is slow: reduce --epochs, --hidden-size, or --layers, or
  install a GPU-enabled build of torch.
- Poor prediction accuracy: this is expected — next-day stock prices
  are close to a random walk, so a plain price-history LSTM will
  mostly track recent momentum rather than truly predicting direction.
  For stronger results, feed in extra features (volume, technical
  indicators, sentiment, etc.) — the script is a good base to extend.


7. RELOADING A SAVED MODEL LATER (for reference)
--------------------------------------------------
    import torch, pickle
    from stock_rnn import StockRNN

    model = StockRNN()  # use same hyperparameters as when you trained it
    model.load_state_dict(torch.load("stock_rnn_model.pt"))
    model.eval()

    with open("stock_rnn_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # then scale new input sequences with `scaler` before feeding them
    # into `model`, and inverse_transform the output to get real prices.
