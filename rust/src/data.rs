//! Data Processing and Feature Engineering
//!
//! This module provides utilities for loading and preprocessing
//! financial data for use with trading models.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Market data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamps: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl MarketData {
    /// Create new MarketData from vectors
    pub fn new(
        timestamps: Vec<i64>,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    ) -> Self {
        Self {
            timestamps,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Get number of data points
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    /// Get close prices as array
    pub fn close_array(&self) -> Array1<f64> {
        Array1::from_vec(self.close.clone())
    }

    /// Get volume as array
    pub fn volume_array(&self) -> Array1<f64> {
        Array1::from_vec(self.volume.clone())
    }
}

/// Feature engineering utilities
pub struct FeatureEngineering;

impl FeatureEngineering {
    /// Calculate log returns
    pub fn returns(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() <= period {
            return vec![0.0; prices.len()];
        }

        let mut result = vec![0.0; period];
        for i in period..prices.len() {
            if prices[i - period] > 0.0 {
                result.push((prices[i] / prices[i - period]).ln());
            } else {
                result.push(0.0);
            }
        }
        result
    }

    /// Calculate Relative Strength Index (RSI)
    pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() <= period {
            return vec![50.0; prices.len()];
        }

        let mut result = vec![50.0; period];
        let mut gains = Vec::new();
        let mut losses = Vec::new();

        // Calculate price changes
        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        // First RSI value using SMA
        let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

        for i in period..gains.len() {
            // Smoothed averages
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            let rs = if avg_loss > 0.0 {
                avg_gain / avg_loss
            } else {
                100.0
            };

            let rsi = 100.0 - (100.0 / (1.0 + rs));
            result.push(rsi);
        }

        result
    }

    /// Calculate MACD histogram
    pub fn macd_histogram(prices: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
        let ema_fast = Self::ema(prices, fast);
        let ema_slow = Self::ema(prices, slow);

        let macd_line: Vec<f64> = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(f, s)| f - s)
            .collect();

        let signal_line = Self::ema(&macd_line, signal);

        macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(m, s)| m - s)
            .collect()
    }

    /// Calculate Exponential Moving Average
    pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(prices.len());
        let multiplier = 2.0 / (period + 1) as f64;

        // First value is SMA
        let sma: f64 = prices.iter().take(period).sum::<f64>() / period as f64;
        result.push(sma);

        for i in 1..prices.len() {
            let prev = result[i - 1];
            let ema = (prices[i] - prev) * multiplier + prev;
            result.push(ema);
        }

        result
    }

    /// Calculate Bollinger Band position (0-1 scale)
    pub fn bollinger_position(prices: &[f64], period: usize, std_dev: f64) -> Vec<f64> {
        if prices.len() < period {
            return vec![0.5; prices.len()];
        }

        let mut result = vec![0.5; period - 1];

        for i in (period - 1)..prices.len() {
            let window: Vec<f64> = prices[(i + 1 - period)..=i].to_vec();
            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            let upper = mean + std_dev * std;
            let lower = mean - std_dev * std;

            let position = if (upper - lower).abs() > 1e-10 {
                (prices[i] - lower) / (upper - lower)
            } else {
                0.5
            };

            result.push(position.clamp(0.0, 1.0));
        }

        result
    }

    /// Calculate Average True Range (normalized by price)
    pub fn atr_normalized(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
        if high.len() < 2 {
            return vec![0.0; high.len()];
        }

        let mut tr = vec![high[0] - low[0]];

        for i in 1..high.len() {
            let tr1 = high[i] - low[i];
            let tr2 = (high[i] - close[i - 1]).abs();
            let tr3 = (low[i] - close[i - 1]).abs();
            tr.push(tr1.max(tr2).max(tr3));
        }

        // Calculate ATR using EMA
        let atr = Self::ema(&tr, period);

        // Normalize by close price
        atr.iter()
            .zip(close.iter())
            .map(|(a, c)| if *c > 0.0 { a / c } else { 0.0 })
            .collect()
    }

    /// Calculate momentum
    pub fn momentum(prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() <= period {
            return vec![0.0; prices.len()];
        }

        let mut result = vec![0.0; period];
        for i in period..prices.len() {
            if prices[i - period] > 0.0 {
                result.push(prices[i] / prices[i - period] - 1.0);
            } else {
                result.push(0.0);
            }
        }
        result
    }

    /// Calculate volume ratio (relative to moving average)
    pub fn volume_ratio(volume: &[f64], period: usize) -> Vec<f64> {
        let ma = Self::ema(volume, period);

        volume
            .iter()
            .zip(ma.iter())
            .map(|(v, m)| if *m > 0.0 { v / m } else { 1.0 })
            .collect()
    }

    /// Generate all features from market data
    pub fn generate_features(data: &MarketData) -> Array2<f64> {
        let n = data.len();

        // Calculate all features
        let returns_1 = Self::returns(&data.close, 1);
        let returns_5 = Self::returns(&data.close, 5);
        let returns_20 = Self::returns(&data.close, 20);
        let rsi = Self::rsi(&data.close, 14);
        let macd = Self::macd_histogram(&data.close, 12, 26, 9);
        let bb_pos = Self::bollinger_position(&data.close, 20, 2.0);
        let atr = Self::atr_normalized(&data.high, &data.low, &data.close, 14);
        let momentum_5 = Self::momentum(&data.close, 5);
        let momentum_20 = Self::momentum(&data.close, 20);
        let vol_ratio = Self::volume_ratio(&data.volume, 20);

        // Normalize RSI to 0-1
        let rsi_norm: Vec<f64> = rsi.iter().map(|x| x / 100.0).collect();

        // Build feature matrix
        let n_features = 10;
        let mut features = Array2::zeros((n, n_features));

        for i in 0..n {
            features[[i, 0]] = returns_1[i];
            features[[i, 1]] = returns_5[i];
            features[[i, 2]] = returns_20[i];
            features[[i, 3]] = rsi_norm[i];
            features[[i, 4]] = macd[i];
            features[[i, 5]] = bb_pos[i];
            features[[i, 6]] = atr[i];
            features[[i, 7]] = momentum_5[i];
            features[[i, 8]] = momentum_20[i];
            features[[i, 9]] = vol_ratio[i];
        }

        features
    }

    /// Get feature names
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "returns_1",
            "returns_5",
            "returns_20",
            "rsi",
            "macd_hist",
            "bb_position",
            "atr_norm",
            "momentum_5",
            "momentum_20",
            "volume_ratio",
        ]
    }

    /// Z-score normalization
    pub fn normalize(features: &Array2<f64>) -> (Array2<f64>, Vec<(f64, f64)>) {
        let n_features = features.ncols();
        let mut params = Vec::with_capacity(n_features);
        let mut normalized = features.clone();

        for j in 0..n_features {
            let col = features.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.std(0.0);

            params.push((mean, std));

            if std > 1e-10 {
                for i in 0..features.nrows() {
                    normalized[[i, j]] = (features[[i, j]] - mean) / std;
                }
            }
        }

        (normalized, params)
    }

    /// Apply normalization with pre-computed parameters
    pub fn apply_normalization(features: &Array2<f64>, params: &[(f64, f64)]) -> Array2<f64> {
        let mut normalized = features.clone();

        for (j, (mean, std)) in params.iter().enumerate() {
            if *std > 1e-10 {
                for i in 0..features.nrows() {
                    normalized[[i, j]] = (features[[i, j]] - mean) / std;
                }
            }
        }

        normalized
    }
}

/// Create target variable (direction)
pub fn create_target(prices: &[f64], horizon: usize) -> Vec<f64> {
    if prices.len() <= horizon {
        return vec![0.5; prices.len()];
    }

    let mut target = Vec::with_capacity(prices.len());

    for i in 0..(prices.len() - horizon) {
        let future_return = prices[i + horizon] / prices[i] - 1.0;
        target.push(if future_return > 0.0 { 1.0 } else { 0.0 });
    }

    // Pad with 0.5 for last horizon values
    target.extend(vec![0.5; horizon]);

    target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 101.0, 102.0, 101.0, 103.0];
        let returns = FeatureEngineering::returns(&prices, 1);

        assert_eq!(returns.len(), 5);
        assert!(returns[0].abs() < 1e-10); // First return is 0
    }

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1)).collect();
        let rsi = FeatureEngineering::rsi(&prices, 14);

        assert_eq!(rsi.len(), 50);
        // Uptrend should have RSI > 50
        assert!(rsi.last().unwrap() > &50.0);
    }

    #[test]
    fn test_ema() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = FeatureEngineering::ema(&prices, 3);

        assert_eq!(ema.len(), 5);
    }

    #[test]
    fn test_feature_generation() {
        let data = MarketData {
            timestamps: (0..100).collect(),
            open: (0..100).map(|i| 100.0 + i as f64 * 0.1).collect(),
            high: (0..100).map(|i| 101.0 + i as f64 * 0.1).collect(),
            low: (0..100).map(|i| 99.0 + i as f64 * 0.1).collect(),
            close: (0..100).map(|i| 100.0 + i as f64 * 0.1).collect(),
            volume: vec![1000.0; 100],
        };

        let features = FeatureEngineering::generate_features(&data);
        assert_eq!(features.shape(), &[100, 10]);
    }
}
