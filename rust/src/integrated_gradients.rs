//! Integrated Gradients Implementation
//!
//! This module implements the Integrated Gradients attribution method
//! for explaining neural network predictions.

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use std::sync::Arc;

use crate::model::NeuralNetwork;

/// Type of baseline for integrated gradients computation
#[derive(Debug, Clone, Copy)]
pub enum BaselineType {
    /// Zero baseline (default)
    Zero,
    /// Mean of training data
    Mean,
    /// Random baseline with small noise
    Random,
}

/// Integrated Gradients explainer
///
/// Computes feature attributions by integrating gradients along a path
/// from a baseline to the input.
#[derive(Debug, Clone)]
pub struct IntegratedGradients {
    /// Number of integration steps
    n_steps: usize,
    /// Type of baseline to use
    baseline_type: BaselineType,
    /// Mean values for mean baseline (optional)
    mean_baseline: Option<Array1<f64>>,
}

impl IntegratedGradients {
    /// Create a new IntegratedGradients explainer
    ///
    /// # Arguments
    ///
    /// * `n_steps` - Number of steps for numerical integration (recommended: 50-300)
    ///
    /// # Example
    ///
    /// ```
    /// use integrated_gradients_finance::IntegratedGradients;
    ///
    /// let ig = IntegratedGradients::new(200);
    /// ```
    pub fn new(n_steps: usize) -> Self {
        Self {
            n_steps,
            baseline_type: BaselineType::Zero,
            mean_baseline: None,
        }
    }

    /// Set the baseline type
    pub fn with_baseline_type(mut self, baseline_type: BaselineType) -> Self {
        self.baseline_type = baseline_type;
        self
    }

    /// Set mean baseline values (for BaselineType::Mean)
    pub fn with_mean_baseline(mut self, mean: Array1<f64>) -> Self {
        self.mean_baseline = Some(mean);
        self.baseline_type = BaselineType::Mean;
        self
    }

    /// Get the baseline for a given input
    fn get_baseline(&self, input: &Array1<f64>) -> Array1<f64> {
        match self.baseline_type {
            BaselineType::Zero => Array1::zeros(input.len()),
            BaselineType::Mean => {
                self.mean_baseline
                    .clone()
                    .unwrap_or_else(|| Array1::zeros(input.len()))
            }
            BaselineType::Random => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                Array1::from_iter((0..input.len()).map(|_| rng.gen::<f64>() * 0.01))
            }
        }
    }

    /// Compute integrated gradients for a single input
    ///
    /// # Arguments
    ///
    /// * `model` - Neural network model implementing NeuralNetwork trait
    /// * `input` - Input features
    /// * `baseline` - Optional custom baseline (overrides baseline_type)
    ///
    /// # Returns
    ///
    /// Attribution scores for each input feature
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use integrated_gradients_finance::{IntegratedGradients, TradingModel};
    /// use ndarray::Array1;
    ///
    /// let model = TradingModel::new(10, vec![32], 1);
    /// let ig = IntegratedGradients::new(200);
    ///
    /// let input = Array1::from_vec(vec![0.5; 10]);
    /// let attrs = ig.explain(&model, &input, None);
    /// ```
    pub fn explain<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
        baseline: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        let baseline = baseline
            .cloned()
            .unwrap_or_else(|| self.get_baseline(input));

        let diff = input - &baseline;

        // Generate interpolated points
        let alphas: Vec<f64> = (1..=self.n_steps)
            .map(|k| k as f64 / self.n_steps as f64)
            .collect();

        // Compute gradients at each step
        let gradients: Vec<Array1<f64>> = alphas
            .iter()
            .map(|&alpha| {
                let interpolated = &baseline + &(&diff * alpha);
                model.gradient(&interpolated)
            })
            .collect();

        // Average gradients
        let n = gradients.len() as f64;
        let avg_gradient = gradients
            .into_iter()
            .fold(Array1::zeros(input.len()), |acc, g| acc + g)
            / n;

        // Attribution = diff * avg_gradient
        &diff * &avg_gradient
    }

    /// Compute integrated gradients for a batch of inputs in parallel
    ///
    /// # Arguments
    ///
    /// * `model` - Neural network model (must be thread-safe)
    /// * `inputs` - Batch of input features (n_samples x n_features)
    ///
    /// # Returns
    ///
    /// Attribution scores (n_samples x n_features)
    pub fn explain_batch<M: NeuralNetwork + Sync>(
        &self,
        model: &M,
        inputs: &Array2<f64>,
    ) -> Array2<f64> {
        let n_samples = inputs.nrows();
        let n_features = inputs.ncols();

        let attributions: Vec<Array1<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let input = inputs.row(i).to_owned();
                self.explain(model, &input, None)
            })
            .collect();

        // Stack into 2D array
        let mut result = Array2::zeros((n_samples, n_features));
        for (i, attr) in attributions.into_iter().enumerate() {
            result.row_mut(i).assign(&attr);
        }

        result
    }

    /// Compute convergence delta (completeness check)
    ///
    /// The delta measures how well the attributions sum to the prediction difference.
    /// A small delta indicates good numerical approximation.
    ///
    /// # Returns
    ///
    /// Absolute difference: |sum(attributions) - (f(input) - f(baseline))|
    pub fn convergence_delta<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
        baseline: Option<&Array1<f64>>,
    ) -> f64 {
        let baseline = baseline
            .cloned()
            .unwrap_or_else(|| self.get_baseline(input));

        let attributions = self.explain(model, input, Some(&baseline));
        let attr_sum: f64 = attributions.sum();

        let pred_input = model.forward(input)[0];
        let pred_baseline = model.forward(&baseline)[0];
        let pred_diff = pred_input - pred_baseline;

        (attr_sum - pred_diff).abs()
    }
}

/// Expected Integrated Gradients - averages over multiple baselines
pub struct ExpectedIntegratedGradients {
    ig: IntegratedGradients,
    n_baselines: usize,
}

impl ExpectedIntegratedGradients {
    /// Create a new Expected IG explainer
    ///
    /// # Arguments
    ///
    /// * `n_steps` - Integration steps per baseline
    /// * `n_baselines` - Number of baselines to average over
    pub fn new(n_steps: usize, n_baselines: usize) -> Self {
        Self {
            ig: IntegratedGradients::new(n_steps),
            n_baselines,
        }
    }

    /// Compute expected integrated gradients
    ///
    /// Averages IG over multiple random baselines for more robust attributions.
    pub fn explain<M: NeuralNetwork>(
        &self,
        model: &M,
        input: &Array1<f64>,
        training_data: Option<&Array2<f64>>,
    ) -> Array1<f64> {
        use rand::seq::SliceRandom;
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut all_attributions = Vec::with_capacity(self.n_baselines);

        for _ in 0..self.n_baselines {
            let baseline = if let Some(data) = training_data {
                // Sample from training data
                let idx = rng.gen_range(0..data.nrows());
                data.row(idx).to_owned()
            } else {
                // Random baseline
                Array1::from_iter((0..input.len()).map(|_| rng.gen::<f64>() * 0.1 - 0.05))
            };

            let attr = self.ig.explain(model, input, Some(&baseline));
            all_attributions.push(attr);
        }

        // Average attributions
        let n = all_attributions.len() as f64;
        all_attributions
            .into_iter()
            .fold(Array1::zeros(input.len()), |acc, a| acc + a)
            / n
    }
}

/// Attribution result with metadata
#[derive(Debug, Clone)]
pub struct AttributionResult {
    /// Feature attributions
    pub attributions: Array1<f64>,
    /// Convergence delta (completeness error)
    pub delta: f64,
    /// Model prediction for input
    pub prediction: f64,
    /// Model prediction for baseline
    pub baseline_prediction: f64,
}

impl AttributionResult {
    /// Get top k features by absolute attribution
    pub fn top_k(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .attributions
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed.truncate(k);
        indexed
    }

    /// Get features with attribution above threshold
    pub fn above_threshold(&self, threshold: f64) -> Vec<(usize, f64)> {
        self.attributions
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > threshold)
            .map(|(i, &v)| (i, v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TradingModel;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ig_basic() {
        let model = TradingModel::new(5, vec![10], 1);
        let ig = IntegratedGradients::new(100);

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let attrs = ig.explain(&model, &input, None);

        assert_eq!(attrs.len(), 5);
    }

    #[test]
    fn test_ig_completeness() {
        let model = TradingModel::new(5, vec![10], 1);
        let ig = IntegratedGradients::new(200);

        let input = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
        let delta = ig.convergence_delta(&model, &input, None);

        // Delta should be small for sufficient steps
        assert!(delta < 0.1, "Delta too large: {}", delta);
    }

    #[test]
    fn test_ig_batch() {
        let model = TradingModel::new(5, vec![10], 1);
        let ig = IntegratedGradients::new(50);

        let inputs = Array2::from_shape_vec(
            (3, 5),
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5,
                0.5, 0.4, 0.3, 0.2, 0.1,
                0.3, 0.3, 0.3, 0.3, 0.3,
            ],
        )
        .unwrap();

        let attrs = ig.explain_batch(&model, &inputs);
        assert_eq!(attrs.shape(), &[3, 5]);
    }
}
