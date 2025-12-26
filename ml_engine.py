import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


class AlphaPredictor:
    def __init__(self, data_dict: dict, benchmark_symbol: str = "SPY", volume_dict: dict = None):
        """
        Enhanced AlphaPredictor with 15+ features for better return prediction.
        Target: Predict Future 20-Day Return %.

        Features include:
        - Momentum: RSI, multi-timeframe returns, relative strength
        - Volume: OBV trend, volume momentum
        - Volatility: Historical vol, vol regime
        - Technical: SMA distances, Bollinger position
        """
        self.data_dict = data_dict
        self.volume_dict = volume_dict or {}
        self.benchmark_symbol = benchmark_symbol
        self.scaler = StandardScaler()

        # Use Gradient Boosting for better prediction (handles non-linear relationships)
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            min_samples_split=15,
            learning_rate=0.05,
            random_state=42,
            subsample=0.8
        )
        self.feature_names = []
        self.feature_importances = {}

    def _compute_rsi(self, series, period=14):
        """Calculate RSI with proper handling of edge cases."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Handle division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI for undefined cases

    def _compute_obv_trend(self, close, volume, window=20):
        """Calculate On-Balance Volume trend (normalized)."""
        if volume is None or len(volume) < window:
            return pd.Series([0] * len(close), index=close.index)

        price_direction = np.sign(close.diff())
        obv = (volume * price_direction).cumsum()

        # Normalize OBV by its rolling mean to get trend direction
        obv_ma = obv.rolling(window).mean()
        obv_trend = (obv - obv_ma) / (obv_ma.abs() + 1e-10)
        return obv_trend.fillna(0)

    def _compute_bollinger_position(self, close, window=20):
        """Calculate position within Bollinger Bands (0-1 scale)."""
        sma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)

        # Position: 0 = at lower band, 1 = at upper band
        position = (close - lower) / (upper - lower + 1e-10)
        return position.clip(0, 1).fillna(0.5)

    def _compute_momentum(self, close, periods):
        """Calculate momentum over multiple periods."""
        return {f"mom_{p}d": close.pct_change(p) for p in periods}

    def prepare_features(self, ticker: str):
        """
        Prepare enhanced feature set for a ticker.
        Returns DataFrame with 15+ features.
        """
        if ticker not in self.data_dict or self.benchmark_symbol not in self.data_dict:
            return None

        df = pd.DataFrame({"Close": self.data_dict[ticker]})
        spy = pd.DataFrame({"Close": self.data_dict[self.benchmark_symbol]})

        # Get volume if available
        volume = None
        if ticker in self.volume_dict:
            volume = pd.Series(self.volume_dict[ticker], index=df.index)

        # Align dates
        df = df.join(spy, lsuffix="_stock", rsuffix="_spy").dropna()

        if len(df) < 60:  # Need minimum history
            return None

        # ==================================================================
        # MOMENTUM FEATURES
        # ==================================================================
        # RSI (14-day)
        df["RSI"] = self._compute_rsi(df["Close_stock"])

        # RSI regime (oversold/neutral/overbought as numeric)
        df["RSI_regime"] = pd.cut(df["RSI"], bins=[0, 30, 70, 100], labels=[1, 0, -1]).astype(float)

        # Multi-timeframe momentum
        df["Returns"] = df["Close_stock"].pct_change()
        df["Mom_5d"] = df["Close_stock"].pct_change(5)
        df["Mom_20d"] = df["Close_stock"].pct_change(20)
        df["Mom_60d"] = df["Close_stock"].pct_change(60)

        # Momentum consistency (are all timeframes aligned?)
        df["Mom_consistency"] = (
            np.sign(df["Mom_5d"]) +
            np.sign(df["Mom_20d"]) +
            np.sign(df["Mom_60d"])
        ) / 3

        # ==================================================================
        # RELATIVE STRENGTH FEATURES
        # ==================================================================
        # Relative strength vs SPY
        df["Rel_Strength"] = df["Close_stock"] / df["Close_spy"]
        df["Rel_Strength_20d_chg"] = df["Rel_Strength"].pct_change(20)

        # Outperformance streak
        spy_returns = df["Close_spy"].pct_change()
        stock_returns = df["Returns"]
        df["Outperform"] = (stock_returns > spy_returns).astype(int)
        df["Outperform_streak"] = df["Outperform"].rolling(10).sum() / 10

        # ==================================================================
        # VOLATILITY FEATURES
        # ==================================================================
        # Historical volatility (20-day)
        df["Volatility"] = df["Returns"].rolling(20).std()

        # Volatility regime (high/normal/low relative to history)
        vol_90d_avg = df["Volatility"].rolling(90).mean()
        df["Vol_regime"] = (df["Volatility"] - vol_90d_avg) / (vol_90d_avg + 1e-10)

        # Volatility trend (expanding or contracting)
        df["Vol_trend"] = df["Volatility"].pct_change(10)

        # ==================================================================
        # TECHNICAL FEATURES
        # ==================================================================
        # SMA Distances
        sma20 = df["Close_stock"].rolling(20).mean()
        sma50 = df["Close_stock"].rolling(50).mean()
        sma200 = df["Close_stock"].rolling(200).mean()

        df["SMA20_dist"] = (df["Close_stock"] - sma20) / sma20
        df["SMA50_dist"] = (df["Close_stock"] - sma50) / sma50
        df["SMA200_dist"] = (df["Close_stock"] - sma200) / sma200

        # Golden/Death Cross signal
        df["SMA_cross"] = (sma50 > sma200).astype(int)

        # Bollinger Band position
        df["BB_position"] = self._compute_bollinger_position(df["Close_stock"])

        # Distance from 52-week high/low
        high_52w = df["Close_stock"].rolling(252).max()
        low_52w = df["Close_stock"].rolling(252).min()
        df["Dist_from_high"] = (df["Close_stock"] - high_52w) / high_52w
        df["Dist_from_low"] = (df["Close_stock"] - low_52w) / low_52w

        # ==================================================================
        # VOLUME FEATURES (if available)
        # ==================================================================
        if volume is not None and len(volume) == len(df):
            # OBV trend
            df["OBV_trend"] = self._compute_obv_trend(df["Close_stock"], volume)

            # Volume momentum
            vol_ma = volume.rolling(20).mean()
            df["Vol_momentum"] = (volume - vol_ma) / (vol_ma + 1e-10)
        else:
            df["OBV_trend"] = 0
            df["Vol_momentum"] = 0

        # ==================================================================
        # TARGET
        # ==================================================================
        # Target: Future 20-Day Return
        df["Target"] = df["Close_stock"].shift(-20) / df["Close_stock"] - 1

        return df

    def get_feature_columns(self):
        """Return list of feature column names."""
        return [
            "RSI", "RSI_regime",
            "Mom_5d", "Mom_20d", "Mom_60d", "Mom_consistency",
            "Rel_Strength", "Rel_Strength_20d_chg", "Outperform_streak",
            "Volatility", "Vol_regime", "Vol_trend",
            "SMA20_dist", "SMA50_dist", "SMA200_dist", "SMA_cross",
            "BB_position", "Dist_from_high", "Dist_from_low",
            "OBV_trend", "Vol_momentum"
        ]

    def train_model(self):
        """
        Train an enhanced model on all available tickers with cross-validation.
        """
        all_features = []
        all_targets = []

        print("  Training Enhanced ML Predictor on batch history...", end=" ")

        feature_cols = self.get_feature_columns()
        self.feature_names = feature_cols

        for ticker in self.data_dict.keys():
            if ticker == self.benchmark_symbol:
                continue

            df = self.prepare_features(ticker)
            if df is None or df.empty:
                continue

            # Drop rows with NaN in features or target
            df_train = df[feature_cols + ["Target"]].dropna()

            if len(df_train) < 50:  # Minimum samples per ticker
                continue

            X = df_train[feature_cols].values
            y = df_train["Target"].values

            all_features.append(X)
            all_targets.append(y)

        if not all_features:
            print("No data to train.")
            return

        X_train = np.vstack(all_features)
        y_train = np.concatenate(all_targets)

        # Remove any remaining NaN/inf values
        mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]

        if len(X_train) < 100:
            print(f"Insufficient clean data ({len(X_train)} samples).")
            return

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train with cross-validation
        try:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=3, scoring='r2')
            print(f"CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})", end=" ")
        except Exception:
            pass  # Skip CV if it fails

        # Fit final model
        self.model.fit(X_train_scaled, y_train)

        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = dict(zip(feature_cols, self.model.feature_importances_))

        print(f"Done. (Trained on {len(X_train)} samples, {len(feature_cols)} features)")

    def predict_alpha_probability(self, ticker: str) -> float:
        """
        Returns PREDICTED FUTURE RETURN % (20-day) using enhanced features.
        """
        if ticker not in self.data_dict:
            return 0.0

        df = self.prepare_features(ticker)
        if df is None or df.empty:
            return 0.0

        feature_cols = self.get_feature_columns()
        last_row = df[feature_cols].iloc[-1]

        # Check for any NaN in features
        if last_row.isna().any():
            return 0.0

        features = last_row.values.reshape(1, -1)

        # Handle case where scaler hasn't been fit
        try:
            features_scaled = self.scaler.transform(features)
        except Exception:
            return 0.0

        # Predict Return
        pred_return = self.model.predict(features_scaled)[0]
        return float(pred_return * 100)  # Return as % (e.g., 5.0 for 5%)

    def get_prediction_confidence(self, ticker: str) -> dict:
        """
        Get prediction with confidence metrics.
        Returns dict with prediction, feature contributions, and confidence.
        """
        if ticker not in self.data_dict:
            return {"prediction": 0.0, "confidence": "low", "top_factors": []}

        prediction = self.predict_alpha_probability(ticker)

        # Calculate confidence based on feature quality
        df = self.prepare_features(ticker)
        if df is None or df.empty:
            return {"prediction": prediction, "confidence": "low", "top_factors": []}

        feature_cols = self.get_feature_columns()
        last_row = df[feature_cols].iloc[-1]
        nan_count = last_row.isna().sum()

        # Confidence levels
        if nan_count == 0:
            confidence = "high"
        elif nan_count <= 3:
            confidence = "medium"
        else:
            confidence = "low"

        # Top contributing factors (if we have feature importances)
        top_factors = []
        if self.feature_importances:
            sorted_features = sorted(
                self.feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            top_factors = [f"{name}: {imp:.2f}" for name, imp in sorted_features]

        return {
            "prediction": prediction,
            "confidence": confidence,
            "top_factors": top_factors
        }
