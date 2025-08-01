"""
Data Loading Utilities for Integrated Gradients Trading Models

This module provides data loading and preprocessing for financial data
from various sources including Yahoo Finance and Bybit.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union
from datetime import datetime, timedelta
import warnings

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class BybitDataFetcher:
    """
    Fetch cryptocurrency data from Bybit API.

    Parameters
    ----------
    base_url : str
        Bybit API base URL
    """

    BASE_URL = "https://api.bybit.com/v5"

    def __init__(self):
        if not HAS_REQUESTS:
            raise ImportError("requests library required for Bybit data fetching")

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch candlestick data from Bybit.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g., "BTCUSDT")
        interval : str
            Candlestick interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        limit : int
            Number of candles to fetch (max 1000)
        start_time : int, optional
            Start timestamp in milliseconds
        end_time : int, optional
            End timestamp in milliseconds

        Returns
        -------
        df : pd.DataFrame
            OHLCV data
        """
        url = f"{self.BASE_URL}/market/kline"

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] != 0:
            raise ValueError(f"API error: {data['retMsg']}")

        klines = data["result"]["list"]

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker information."""
        url = f"{self.BASE_URL}/market/tickers"
        params = {"category": "linear", "symbol": symbol}

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] != 0:
            raise ValueError(f"API error: {data['retMsg']}")

        return data["result"]["list"][0]


class FeatureEngineering:
    """
    Technical indicator and feature calculation for trading models.
    """

    @staticmethod
    def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate log returns."""
        return np.log(prices / prices.shift(periods))

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, sma, lower

    @staticmethod
    def calculate_bb_position(prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands (0-1 scale)."""
        upper, _, lower = FeatureEngineering.calculate_bollinger_bands(prices, period)
        return (prices - lower) / (upper - lower + 1e-10)

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        sign = np.sign(close.diff())
        return (sign * volume).cumsum()

    @staticmethod
    def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate price momentum."""
        return prices / prices.shift(period) - 1

    @staticmethod
    def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculate volume ratio relative to moving average."""
        return volume / volume.rolling(window=period).mean()


class IGDataLoader:
    """
    Data loader for integrated gradients trading models.

    Loads and preprocesses financial data with technical indicators
    for use with trading models and IG explanations.

    Parameters
    ----------
    symbols : list of str
        Trading symbols to load
    source : str
        Data source: "yfinance" or "bybit"
    features : list of str, optional
        Features to calculate
    seq_length : int
        Sequence length for time series
    """

    DEFAULT_FEATURES = [
        "returns", "returns_5", "returns_20",
        "volume_ratio", "rsi", "macd_hist",
        "bb_position", "atr_normalized", "momentum_5", "momentum_20"
    ]

    def __init__(
        self,
        symbols: List[str],
        source: str = "yfinance",
        features: Optional[List[str]] = None,
        seq_length: int = 50,
    ):
        self.symbols = symbols
        self.source = source
        self.features = features or self.DEFAULT_FEATURES
        self.seq_length = seq_length

        self.scaler_params: Dict[str, Tuple[float, float]] = {}

    def _fetch_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        if not HAS_YFINANCE:
            raise ImportError("yfinance library required for Yahoo Finance data")

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        df.columns = df.columns.str.lower()
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

        if 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        elif 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})

        return df

    def _fetch_bybit(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "60"
    ) -> pd.DataFrame:
        """Fetch data from Bybit."""
        fetcher = BybitDataFetcher()

        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        all_data = []
        current_end = end_ts

        while current_end > start_ts:
            df = fetcher.fetch_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                end_time=current_end
            )

            if df.empty:
                break

            all_data.append(df)
            current_end = int(df["timestamp"].min().timestamp() * 1000) - 1

            if len(df) < 1000:
                break

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"])
        result = result.sort_values("timestamp").reset_index(drop=True)

        # Filter to date range
        result = result[
            (result["timestamp"] >= pd.Timestamp(start_date)) &
            (result["timestamp"] <= pd.Timestamp(end_date))
        ]

        return result

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and features."""
        fe = FeatureEngineering()

        # Ensure required columns exist
        required_cols = ["close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        result = df.copy()

        # Returns
        if "returns" in self.features:
            result["returns"] = fe.calculate_returns(df["close"])
        if "returns_5" in self.features:
            result["returns_5"] = fe.calculate_returns(df["close"], 5)
        if "returns_20" in self.features:
            result["returns_20"] = fe.calculate_returns(df["close"], 20)

        # Volume
        if "volume_ratio" in self.features:
            result["volume_ratio"] = fe.calculate_volume_ratio(df["volume"])

        # RSI
        if "rsi" in self.features:
            result["rsi"] = fe.calculate_rsi(df["close"]) / 100  # Normalize to 0-1

        # MACD
        if "macd_hist" in self.features:
            _, _, hist = fe.calculate_macd(df["close"])
            result["macd_hist"] = hist

        # Bollinger Bands
        if "bb_position" in self.features:
            result["bb_position"] = fe.calculate_bb_position(df["close"])

        # ATR
        if "atr_normalized" in self.features and "high" in df.columns and "low" in df.columns:
            atr = fe.calculate_atr(df["high"], df["low"], df["close"])
            result["atr_normalized"] = atr / df["close"]

        # Momentum
        if "momentum_5" in self.features:
            result["momentum_5"] = fe.calculate_momentum(df["close"], 5)
        if "momentum_20" in self.features:
            result["momentum_20"] = fe.calculate_momentum(df["close"], 20)

        # OBV
        if "obv" in self.features:
            obv = fe.calculate_obv(df["close"], df["volume"])
            result["obv"] = obv / obv.abs().max()  # Normalize

        return result

    def _create_target(
        self,
        df: pd.DataFrame,
        target_type: str = "direction",
        horizon: int = 1
    ) -> pd.Series:
        """Create target variable."""
        future_returns = df["close"].shift(-horizon) / df["close"] - 1

        if target_type == "direction":
            return (future_returns > 0).astype(float)
        elif target_type == "returns":
            return future_returns
        elif target_type == "magnitude":
            return future_returns.abs()
        else:
            return (future_returns > 0).astype(float)

    def _normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """Normalize features using z-score."""
        result = df.copy()

        for col in feature_cols:
            if col not in result.columns:
                continue

            if fit:
                mean = result[col].mean()
                std = result[col].std()
                self.scaler_params[col] = (mean, std)
            else:
                mean, std = self.scaler_params.get(col, (0, 1))

            if std > 0:
                result[col] = (result[col] - mean) / std
            else:
                result[col] = 0

        return result

    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        target_type: str = "direction",
        target_horizon: int = 1,
        train_ratio: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess data for training.

        Parameters
        ----------
        start_date : str, optional
            Start date (default: 2 years ago)
        end_date : str, optional
            End date (default: today)
        interval : str
            Data interval
        target_type : str
            Target type: "direction", "returns", "magnitude"
        target_horizon : int
            Prediction horizon in bars
        train_ratio : float
            Training data ratio

        Returns
        -------
        X_train : np.ndarray
            Training features
        X_test : np.ndarray
            Test features
        y_train : np.ndarray
            Training targets
        y_test : np.ndarray
            Test targets
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        all_data = []

        for symbol in self.symbols:
            if self.source == "yfinance":
                df = self._fetch_yfinance(symbol, start_date, end_date, interval)
            elif self.source == "bybit":
                bybit_interval = "D" if interval == "1d" else interval.replace("h", "")
                df = self._fetch_bybit(symbol, start_date, end_date, bybit_interval)
            else:
                raise ValueError(f"Unknown source: {self.source}")

            if df.empty:
                warnings.warn(f"No data for {symbol}")
                continue

            df = self._calculate_features(df)
            df["target"] = self._create_target(df, target_type, target_horizon)
            df["symbol"] = symbol

            all_data.append(df)

        if not all_data:
            raise ValueError("No data loaded for any symbol")

        combined = pd.concat(all_data, ignore_index=True)

        # Get feature columns
        feature_cols = [f for f in self.features if f in combined.columns]

        # Drop NaN rows
        combined = combined.dropna(subset=feature_cols + ["target"])

        # Split train/test
        n_train = int(len(combined) * train_ratio)
        train_df = combined.iloc[:n_train]
        test_df = combined.iloc[n_train:]

        # Normalize
        train_df = self._normalize_features(train_df, feature_cols, fit=True)
        test_df = self._normalize_features(test_df, feature_cols, fit=False)

        # Extract arrays
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        y_train = train_df["target"].values
        y_test = test_df["target"].values

        return X_train, X_test, y_train, y_test

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [f for f in self.features if f in self.DEFAULT_FEATURES or f in self.features]


def create_sample_data(
    n_samples: int = 1000,
    n_features: int = 10,
    noise_level: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data for testing.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    noise_level : float
        Noise level
    seed : int
        Random seed

    Returns
    -------
    X : np.ndarray
        Features
    y : np.ndarray
        Targets
    """
    np.random.seed(seed)

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create non-linear target with known important features
    # Features 0, 2, 5 are important
    y = (
        0.5 * X[:, 0] +
        0.3 * np.sin(X[:, 2]) +
        0.2 * X[:, 5] ** 2 +
        noise_level * np.random.randn(n_samples)
    )

    # Convert to binary classification
    y = (y > y.mean()).astype(float)

    return X, y
