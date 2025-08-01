//! Neural Network Models for Trading
//!
//! This module provides neural network implementations suitable for
//! trading signal prediction with gradient computation support.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Trait for neural networks that support gradient computation
pub trait NeuralNetwork {
    /// Forward pass: compute output from input
    fn forward(&self, input: &Array1<f64>) -> Array1<f64>;

    /// Compute gradient of output with respect to input
    fn gradient(&self, input: &Array1<f64>) -> Array1<f64>;
}

/// Activation function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

impl Activation {
    /// Apply activation function
    fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Linear => x,
        }
    }

    /// Compute derivative of activation function
    fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Linear => 1.0,
        }
    }
}

/// Dense (fully connected) layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    weights: Array2<f64>,
    bias: Array1<f64>,
    activation: Activation,
}

impl DenseLayer {
    /// Create a new dense layer with random initialization
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();

        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.gen::<f64>() * 2.0 * scale - scale
        });

        let bias = Array1::zeros(output_size);

        Self {
            weights,
            bias,
            activation,
        }
    }

    /// Forward pass through the layer
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = self.weights.dot(input) + &self.bias;
        z.mapv(|x| self.activation.apply(x))
    }

    /// Compute pre-activation values
    fn pre_activation(&self, input: &Array1<f64>) -> Array1<f64> {
        self.weights.dot(input) + &self.bias
    }
}

/// Multi-layer perceptron trading model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingModel {
    layers: Vec<DenseLayer>,
    input_size: usize,
    hidden_sizes: Vec<usize>,
    output_size: usize,
}

impl TradingModel {
    /// Create a new trading model
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `hidden_sizes` - Sizes of hidden layers
    /// * `output_size` - Number of outputs
    ///
    /// # Example
    ///
    /// ```
    /// use integrated_gradients_finance::TradingModel;
    ///
    /// // Create model with 10 inputs, two hidden layers (64, 32), and 1 output
    /// let model = TradingModel::new(10, vec![64, 32], 1);
    /// ```
    pub fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        let mut layers = Vec::new();
        let mut in_size = input_size;

        // Hidden layers with ReLU
        for &hidden_size in &hidden_sizes {
            layers.push(DenseLayer::new(in_size, hidden_size, Activation::ReLU));
            in_size = hidden_size;
        }

        // Output layer with linear activation
        layers.push(DenseLayer::new(in_size, output_size, Activation::Linear));

        Self {
            layers,
            input_size,
            hidden_sizes,
            output_size,
        }
    }

    /// Create model with custom activation
    pub fn with_activation(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        hidden_activation: Activation,
        output_activation: Activation,
    ) -> Self {
        let mut layers = Vec::new();
        let mut in_size = input_size;

        for &hidden_size in &hidden_sizes {
            layers.push(DenseLayer::new(in_size, hidden_size, hidden_activation));
            in_size = hidden_size;
        }

        layers.push(DenseLayer::new(in_size, output_size, output_activation));

        Self {
            layers,
            input_size,
            hidden_sizes,
            output_size,
        }
    }

    /// Get model architecture info
    pub fn architecture(&self) -> String {
        let mut sizes = vec![self.input_size];
        sizes.extend(&self.hidden_sizes);
        sizes.push(self.output_size);

        sizes
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    /// Save model to JSON
    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Load model from JSON
    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        let model: Self = serde_json::from_str(&json)?;
        Ok(model)
    }

    /// Predict probability (sigmoid output)
    pub fn predict_proba(&self, input: &Array1<f64>) -> f64 {
        let output = self.forward(input);
        1.0 / (1.0 + (-output[0]).exp())
    }

    /// Predict class (threshold at 0.5)
    pub fn predict_class(&self, input: &Array1<f64>) -> i32 {
        if self.predict_proba(input) > 0.5 { 1 } else { 0 }
    }
}

impl NeuralNetwork for TradingModel {
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    fn gradient(&self, input: &Array1<f64>) -> Array1<f64> {
        // Forward pass, storing intermediate values
        let mut activations = vec![input.clone()];
        let mut pre_activations = Vec::new();

        let mut current = input.clone();
        for layer in &self.layers {
            let z = layer.pre_activation(&current);
            pre_activations.push(z.clone());
            current = z.mapv(|x| layer.activation.apply(x));
            activations.push(current.clone());
        }

        // Backward pass
        // Start with gradient of output (assuming we want gradient w.r.t. first output)
        let mut delta = Array1::ones(self.output_size);

        // Backpropagate through layers
        for (i, layer) in self.layers.iter().enumerate().rev() {
            // Apply activation derivative
            let z = &pre_activations[i];
            let act_deriv: Array1<f64> = z.mapv(|x| layer.activation.derivative(x));
            delta = &delta * &act_deriv;

            // Propagate through weights
            if i > 0 {
                delta = layer.weights.t().dot(&delta);
            } else {
                // Final gradient w.r.t. input
                delta = layer.weights.t().dot(&delta);
            }
        }

        delta
    }
}

// Make TradingModel thread-safe for parallel batch processing
unsafe impl Send for TradingModel {}
unsafe impl Sync for TradingModel {}

/// Multi-task trading model
///
/// Predicts multiple outputs: direction, volatility, magnitude
#[derive(Debug, Clone)]
pub struct MultiTaskTradingModel {
    shared_layers: Vec<DenseLayer>,
    direction_head: DenseLayer,
    volatility_head: DenseLayer,
    magnitude_head: DenseLayer,
}

impl MultiTaskTradingModel {
    /// Create a new multi-task model
    pub fn new(input_size: usize, hidden_sizes: Vec<usize>) -> Self {
        let mut shared_layers = Vec::new();
        let mut in_size = input_size;

        for &hidden_size in &hidden_sizes {
            shared_layers.push(DenseLayer::new(in_size, hidden_size, Activation::ReLU));
            in_size = hidden_size;
        }

        let head_hidden = 32;

        Self {
            shared_layers,
            direction_head: DenseLayer::new(in_size, 1, Activation::Linear),
            volatility_head: DenseLayer::new(in_size, 1, Activation::Linear),
            magnitude_head: DenseLayer::new(in_size, 1, Activation::Linear),
        }
    }

    /// Forward pass returning all outputs
    pub fn forward(&self, input: &Array1<f64>) -> (f64, f64, f64) {
        let mut shared = input.clone();
        for layer in &self.shared_layers {
            shared = layer.forward(&shared);
        }

        let direction = self.direction_head.forward(&shared)[0];
        let volatility = self.volatility_head.forward(&shared)[0].exp(); // Softplus-like
        let magnitude = self.magnitude_head.forward(&shared)[0];

        (direction, volatility, magnitude)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_forward() {
        let model = TradingModel::new(5, vec![10, 5], 1);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_model_gradient() {
        let model = TradingModel::new(5, vec![10], 1);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let grad = model.gradient(&input);
        assert_eq!(grad.len(), 5);
    }

    #[test]
    fn test_model_architecture() {
        let model = TradingModel::new(10, vec![64, 32], 1);
        assert_eq!(model.architecture(), "10 -> 64 -> 32 -> 1");
    }

    #[test]
    fn test_activation_functions() {
        let relu = Activation::ReLU;
        assert_eq!(relu.apply(-1.0), 0.0);
        assert_eq!(relu.apply(1.0), 1.0);

        let sigmoid = Activation::Sigmoid;
        assert!((sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
    }
}
