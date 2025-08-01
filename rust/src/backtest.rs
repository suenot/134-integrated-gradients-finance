//! Backtesting Framework with Attribution Support
//!
//! This module provides backtesting capabilities for trading strategies
//! with integrated gradients attribution tracking.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::integrated_gradients::IntegratedGradients;
use crate::model::NeuralNetwork;

/// Single trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub entry_index: usize,
    pub exit_index: usize,
    pub direction: i32,  // 1 for long, -1 for short
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl: f64,
    pub pnl_percent: f64,
    pub confidence: f64,
    pub attributions: Option<Vec<f64>>,
}

/// Backtest performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub avg_trade_pnl: f64,
    pub equity_curve: Vec<f64>,
    pub trades: Vec<Trade>,
    pub winning_attr_mean: Option<Vec<f64>>,
    pub losing_attr_mean: Option<Vec<f64>>,
}

/// Backtester with integrated gradients support
pub struct Backtester<M: NeuralNetwork> {
    model: M,
    ig: IntegratedGradients,
    initial_capital: f64,
    transaction_cost: f64,
    position_size: f64,
    log_attributions: bool,
}

impl<M: NeuralNetwork> Backtester<M> {
    /// Create a new backtester
    ///
    /// # Arguments
    ///
    /// * `model` - Trading model
    /// * `initial_capital` - Starting capital
    /// * `transaction_cost` - Cost per trade as fraction
    /// * `position_size` - Position size as fraction of capital
    pub fn new(
        model: M,
        initial_capital: f64,
        transaction_cost: f64,
        position_size: f64,
    ) -> Self {
        Self {
            model,
            ig: IntegratedGradients::new(100),
            initial_capital,
            transaction_cost,
            position_size,
            log_attributions: true,
        }
    }

    /// Set whether to log attributions
    pub fn with_attributions(mut self, log: bool) -> Self {
        self.log_attributions = log;
        self
    }

    /// Set IG integration steps
    pub fn with_ig_steps(mut self, steps: usize) -> Self {
        self.ig = IntegratedGradients::new(steps);
        self
    }

    /// Generate trading signal from prediction
    fn generate_signal(&self, features: &Array1<f64>, threshold: f64) -> (i32, f64) {
        let output = self.model.forward(features);
        let prob = 1.0 / (1.0 + (-output[0]).exp());  // Sigmoid

        let signal = if prob > threshold {
            1  // Long
        } else if prob < (1.0 - threshold) {
            -1  // Short
        } else {
            0  // Neutral
        };

        (signal, prob)
    }

    /// Run backtest
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (n_samples x n_features)
    /// * `prices` - Price series
    /// * `signal_threshold` - Threshold for generating signals
    /// * `holding_period` - Number of bars to hold position
    ///
    /// # Returns
    ///
    /// Backtest results including performance metrics and trades
    pub fn run(
        &self,
        features: &ndarray::Array2<f64>,
        prices: &[f64],
        signal_threshold: f64,
        holding_period: usize,
    ) -> BacktestResults {
        let n = features.nrows();
        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve = vec![self.initial_capital];
        let mut capital = self.initial_capital;

        let mut position: Option<(usize, i32, f64, f64, Option<Vec<f64>>)> = None;
        let mut bars_held = 0;

        for i in 0..(n - holding_period) {
            let features_i = features.row(i).to_owned();
            let price = prices[i];

            // Check if we need to close position
            if let Some((entry_idx, direction, entry_price, confidence, attrs)) = position.clone() {
                bars_held += 1;
                if bars_held >= holding_period {
                    // Close position
                    let pnl_percent = direction as f64 * (price / entry_price - 1.0) - self.transaction_cost;
                    let pnl = pnl_percent * self.position_size * capital;

                    capital += pnl;

                    trades.push(Trade {
                        entry_index: entry_idx,
                        exit_index: i,
                        direction,
                        entry_price,
                        exit_price: price,
                        pnl,
                        pnl_percent,
                        confidence,
                        attributions: attrs,
                    });

                    position = None;
                    bars_held = 0;
                }
            }

            // Generate signal if no position
            if position.is_none() {
                let (signal, confidence) = self.generate_signal(&features_i, signal_threshold);

                if signal != 0 {
                    // Compute attributions if logging
                    let attrs = if self.log_attributions {
                        Some(self.ig.explain(&self.model, &features_i, None).to_vec())
                    } else {
                        None
                    };

                    capital -= capital * self.position_size * self.transaction_cost;
                    position = Some((i, signal, price, confidence, attrs));
                }
            }

            equity_curve.push(capital);
        }

        // Close any remaining position
        if let Some((entry_idx, direction, entry_price, confidence, attrs)) = position {
            let exit_price = prices[n - 1];
            let pnl_percent = direction as f64 * (exit_price / entry_price - 1.0) - self.transaction_cost;
            let pnl = pnl_percent * self.position_size * capital;

            capital += pnl;

            trades.push(Trade {
                entry_index: entry_idx,
                exit_index: n - 1,
                direction,
                entry_price,
                exit_price,
                pnl,
                pnl_percent,
                confidence,
                attributions: attrs,
            });
        }

        equity_curve.push(capital);

        // Calculate metrics
        self.calculate_results(equity_curve, trades)
    }

    /// Calculate performance metrics
    fn calculate_results(&self, equity_curve: Vec<f64>, trades: Vec<Trade>) -> BacktestResults {
        let n = equity_curve.len();

        // Calculate returns
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| w[1] / w[0] - 1.0)
            .collect();

        // Total return
        let total_return = equity_curve.last().unwrap() / self.initial_capital - 1.0;

        // Sharpe ratio (annualized, assuming daily data)
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();
        let sharpe_ratio = if std_dev > 0.0 {
            mean_return / std_dev * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if !downside_returns.is_empty() {
            let var: f64 = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64;
            var.sqrt()
        } else {
            std_dev
        };
        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * (252.0_f64).sqrt()
        } else {
            sharpe_ratio
        };

        // Maximum drawdown
        let mut peak = self.initial_capital;
        let mut max_drawdown = 0.0;
        for &equity in &equity_curve {
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Trade statistics
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64
        } else {
            0.0
        };

        let total_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let total_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if total_loss > 0.0 {
            total_profit / total_loss
        } else {
            f64::INFINITY
        };

        let avg_trade_pnl = if !trades.is_empty() {
            trades.iter().map(|t| t.pnl).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };

        // Attribution analysis
        let (winning_attr_mean, losing_attr_mean) = self.analyze_attributions(&trades);

        BacktestResults {
            total_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: -max_drawdown,
            win_rate,
            profit_factor,
            total_trades: trades.len(),
            avg_trade_pnl,
            equity_curve,
            trades,
            winning_attr_mean,
            losing_attr_mean,
        }
    }

    /// Analyze attributions for winning vs losing trades
    fn analyze_attributions(&self, trades: &[Trade]) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        let winning: Vec<&Vec<f64>> = trades
            .iter()
            .filter(|t| t.pnl > 0.0 && t.attributions.is_some())
            .map(|t| t.attributions.as_ref().unwrap())
            .collect();

        let losing: Vec<&Vec<f64>> = trades
            .iter()
            .filter(|t| t.pnl <= 0.0 && t.attributions.is_some())
            .map(|t| t.attributions.as_ref().unwrap())
            .collect();

        let winning_mean = if !winning.is_empty() {
            let n_features = winning[0].len();
            let mut mean = vec![0.0; n_features];
            for attrs in &winning {
                for (i, &a) in attrs.iter().enumerate() {
                    mean[i] += a;
                }
            }
            for m in &mut mean {
                *m /= winning.len() as f64;
            }
            Some(mean)
        } else {
            None
        };

        let losing_mean = if !losing.is_empty() {
            let n_features = losing[0].len();
            let mut mean = vec![0.0; n_features];
            for attrs in &losing {
                for (i, &a) in attrs.iter().enumerate() {
                    mean[i] += a;
                }
            }
            for m in &mut mean {
                *m /= losing.len() as f64;
            }
            Some(mean)
        } else {
            None
        };

        (winning_mean, losing_mean)
    }
}

/// Calculate common trading metrics from returns
pub fn calculate_metrics(returns: &[f64]) -> std::collections::HashMap<String, f64> {
    let mut metrics = std::collections::HashMap::new();

    if returns.is_empty() {
        return metrics;
    }

    let n = returns.len() as f64;

    // Total return
    let total_return: f64 = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;
    metrics.insert("total_return".to_string(), total_return);

    // Mean return
    let mean_return: f64 = returns.iter().sum::<f64>() / n;
    metrics.insert("annualized_return".to_string(), mean_return * 252.0);

    // Volatility
    let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n;
    let volatility = variance.sqrt() * (252.0_f64).sqrt();
    metrics.insert("annualized_volatility".to_string(), volatility);

    // Sharpe ratio
    let sharpe = if volatility > 0.0 {
        mean_return * 252.0 / volatility
    } else {
        0.0
    };
    metrics.insert("sharpe_ratio".to_string(), sharpe);

    // Maximum drawdown
    let mut equity = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;
    for r in returns {
        equity *= 1.0 + r;
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    metrics.insert("max_drawdown".to_string(), -max_dd);

    // Skewness
    let skewness: f64 = returns
        .iter()
        .map(|r| ((r - mean_return) / variance.sqrt()).powi(3))
        .sum::<f64>()
        / n;
    metrics.insert("skewness".to_string(), skewness);

    // Kurtosis
    let kurtosis: f64 = returns
        .iter()
        .map(|r| ((r - mean_return) / variance.sqrt()).powi(4))
        .sum::<f64>()
        / n
        - 3.0;
    metrics.insert("kurtosis".to_string(), kurtosis);

    metrics
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TradingModel;
    use ndarray::Array2;

    #[test]
    fn test_backtester_basic() {
        let model = TradingModel::new(5, vec![10], 1);
        let backtester = Backtester::new(model, 100_000.0, 0.001, 0.1);

        // Create simple test data
        let features = Array2::from_shape_fn((100, 5), |_| 0.1);
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.1).collect();

        let results = backtester.run(&features, &prices, 0.55, 5);

        assert!(results.equity_curve.len() > 0);
    }

    #[test]
    fn test_metrics_calculation() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let metrics = calculate_metrics(&returns);

        assert!(metrics.contains_key("sharpe_ratio"));
        assert!(metrics.contains_key("max_drawdown"));
    }
}
