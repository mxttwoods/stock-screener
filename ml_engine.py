import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class AlphaPredictor:
    def __init__(self, data_dict: dict, benchmark_symbol: str = "SPY"):
        """
        Initializes the AlphaPredictor (Regressor).
        Target: Predict Future 20-Day Return %.
        """
        self.data_dict = data_dict
        self.benchmark_symbol = benchmark_symbol
        self.model = RandomForestRegressor(
            n_estimators=100, min_samples_split=10, random_state=42
        )

    def _compute_rsa(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_features(self, ticker: str):
        if ticker not in self.data_dict or self.benchmark_symbol not in self.data_dict:
            return None

        df = pd.DataFrame({"Close": self.data_dict[ticker]})
        spy = pd.DataFrame({"Close": self.data_dict[self.benchmark_symbol]})

        # Align dates
        df = df.join(spy, lsuffix="_stock", rsuffix="_spy").dropna()

        # Feature Engineering
        # 1. RSI (Momentum)
        df["RSI"] = self._compute_rsa(df["Close_stock"])

        # 2. Relative Strength (Price / SPY)
        df["Rel_Strength"] = df["Close_stock"] / df["Close_spy"]

        # 3. Volatility (20-day rolling std dev of returns)
        df["Returns"] = df["Close_stock"].pct_change()
        df["Volatility"] = df["Returns"].rolling(20).std()

        # 4. SMA Distance ((Price - SMA50) / SMA50)
        sma50 = df["Close_stock"].rolling(50).mean()
        df["SMA_Dist"] = (df["Close_stock"] - sma50) / sma50

        # Target: Future 20-Day Return (Magnitude)
        # Shift -20 means look 20 days ahead.
        df["Target"] = df["Close_stock"].shift(-20) / df["Close_stock"] - 1

        # Drop inputs that are NaN (due to rolling) OR targets that are NaN (last 20 days)
        # For training, we need valid targets.
        return df

    def train_model(self):
        """
        Trains a global model on all available tickers.
        """
        all_features = []
        all_targets = []

        print("  Training ML Regressor (Future Returns) on batch history...", end=" ")

        for ticker in self.data_dict.keys():
            if ticker == self.benchmark_symbol:
                continue

            df = self.prepare_features(ticker)
            if df is None or df.empty:
                continue

            # Drop rows with NaN in features or target
            df_train = df.dropna()

            if df_train.empty:
                continue

            # Features
            feat_cols = ["RSI", "Rel_Strength", "Volatility", "SMA_Dist"]
            X = df_train[feat_cols].values
            y = df_train["Target"].values

            all_features.append(X)
            all_targets.append(y)

        if not all_features:
            print("No data to train.")
            return

        X_train = np.vstack(all_features)
        y_train = np.concatenate(all_targets)

        # Fit model
        self.model.fit(X_train, y_train)
        print(f"Done. (Trained on {len(X_train)} samples)")

    def predict_alpha_probability(self, ticker: str) -> float:
        """
        Returns PREDICTED FUTURE RETURN % (20-day).
        Scaled to 0-100 logic for compatibility?
        No, let's return the raw % forecast, and handle scaling in main.py.
        """
        if ticker not in self.data_dict:
            return 0.0

        # For prediction, we don't need the Target column (which requires future data)
        # We just need the latest features.

        df_raw = pd.DataFrame({"Close": self.data_dict[ticker]})
        spy_raw = pd.DataFrame({"Close": self.data_dict[self.benchmark_symbol]})
        df = df_raw.join(spy_raw, lsuffix="_stock", rsuffix="_spy")

        df["RSI"] = self._compute_rsa(df["Close_stock"])
        df["Rel_Strength"] = df["Close_stock"] / df["Close_spy"]
        df["Returns"] = df["Close_stock"].pct_change()
        df["Volatility"] = df["Returns"].rolling(20).std()
        sma50 = df["Close_stock"].rolling(50).mean()
        df["SMA_Dist"] = (df["Close_stock"] - sma50) / sma50

        last_row = df.iloc[-1]

        # specific check for NaN in features
        if pd.isna(last_row["SMA_Dist"]) or pd.isna(last_row["RSI"]):
            return 0.0

        features = [
            [
                last_row["RSI"],
                last_row["Rel_Strength"],
                last_row["Volatility"],
                last_row["SMA_Dist"],
            ]
        ]

        # Predict Return
        pred_return = self.model.predict(features)[0]
        return float(pred_return * 100)  # Return as % (e.g., 5.0 for 5%)
