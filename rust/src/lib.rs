//! # Integrated Gradients for Financial Trading
//!
//! This crate implements the Integrated Gradients attribution method for
//! explaining predictions of neural network trading models.
//!
//! ## Features
//!
//! - Integrated Gradients computation with configurable steps
//! - Support for various baseline types (zero, mean, random)
//! - Batch processing with parallel computation
//! - Trading model implementations
//! - Bybit API integration for cryptocurrency data
//!
//! ## Example
//!
//! ```rust,no_run
//! use integrated_gradients_finance::{IntegratedGradients, TradingModel};
//! use ndarray::Array1;
//!
//! // Create a simple trading model
//! let model = TradingModel::new(10, vec![64, 32], 1);
//!
//! // Create IG explainer
//! let ig = IntegratedGradients::new(200);
//!
//! // Compute attributions
//! let input = Array1::from_vec(vec![0.1; 10]);
//! let attributions = ig.explain(&model, &input, None);
//! ```

pub mod integrated_gradients;
pub mod model;
pub mod data;
pub mod backtest;

pub use integrated_gradients::{IntegratedGradients, BaselineType};
pub use model::{TradingModel, NeuralNetwork};
pub use data::{FeatureEngineering, MarketData};
pub use backtest::{Backtester, BacktestResults, Trade};

#[cfg(feature = "bybit")]
pub use bybit_client::{BybitClient, Candle, Interval};
