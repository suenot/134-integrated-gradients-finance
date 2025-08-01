"""
Trading Model with Integrated Gradients Support

This module provides neural network models for trading with built-in
interpretability through integrated gradients.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List, Tuple, Dict, Union
from tqdm import tqdm

from .integrated_gradients import IntegratedGradients


class TradingModel(nn.Module):
    """
    Multi-layer perceptron for trading signal prediction.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_sizes : list of int
        Sizes of hidden layers
    n_outputs : int
        Number of outputs (e.g., 1 for direction, 3 for direction + volatility + magnitude)
    dropout : float
        Dropout probability
    activation : str
        Activation function: "relu", "gelu", "silu"
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64, 32],
        n_outputs: int = 1,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.n_outputs = n_outputs

        # Build layers
        layers = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())

            layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        # Output layer
        layers.append(nn.Linear(in_features, n_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probabilities (sigmoid for binary, softmax for multi-class)."""
        logits = self.forward(x)
        if self.n_outputs == 1:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=-1)


class TradingModelWithIG:
    """
    Trading model wrapper with integrated gradients explanation capabilities.

    This class combines a trading model with an integrated gradients explainer
    for generating predictions along with feature attributions.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_sizes : list of int
        Sizes of hidden layers
    n_outputs : int
        Number of outputs
    dropout : float
        Dropout probability
    ig_steps : int
        Number of integration steps for IG
    device : str, optional
        Device for computation
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64, 32],
        n_outputs: int = 1,
        dropout: float = 0.2,
        ig_steps: int = 200,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TradingModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            n_outputs=n_outputs,
            dropout=dropout,
        ).to(self.device)

        self.ig = IntegratedGradients(
            model=self.model,
            n_steps=ig_steps,
            device=self.device,
        )

        self.feature_names: Optional[List[str]] = None
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def set_feature_names(self, names: List[str]):
        """Set feature names for interpretability."""
        self.feature_names = names

    def fit(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Parameters
        ----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation targets
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        weight_decay : float
            L2 regularization
        early_stopping_patience : int
            Patience for early stopping
        verbose : bool
            Print progress

        Returns
        -------
        history : dict
            Training history
        """
        # Convert to tensors
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32)

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        if y_train.dim() == 1:
            y_train = y_train.unsqueeze(-1)

        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation data
        if X_val is not None and y_val is not None:
            if not isinstance(X_val, torch.Tensor):
                X_val = torch.tensor(X_val, dtype=torch.float32)
            if not isinstance(y_val, torch.Tensor):
                y_val = torch.tensor(y_val, dtype=torch.float32)
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            if y_val.dim() == 1:
                y_val = y_val.unsqueeze(-1)

        # Loss and optimizer
        if self.model.n_outputs == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in iterator:
            # Training
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            self.training_history["train_loss"].append(avg_train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()

                self.training_history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

                if verbose:
                    iterator.set_postfix({
                        'train_loss': f'{avg_train_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}'
                    })
            else:
                if verbose:
                    iterator.set_postfix({'train_loss': f'{avg_train_loss:.4f}'})

        return self.training_history

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : array-like
            Input features
        return_proba : bool
            Return probabilities instead of raw outputs

        Returns
        -------
        predictions : np.ndarray
            Model predictions
        """
        self.model.eval()

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)

        with torch.no_grad():
            if return_proba:
                outputs = self.model.predict_proba(X)
            else:
                outputs = self.model(X)

        return outputs.cpu().numpy()

    def predict_with_explanations(
        self,
        X: Union[np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        return_proba: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with integrated gradients explanations.

        Parameters
        ----------
        X : array-like
            Input features
        target_class : int, optional
            Target class for attribution
        return_proba : bool
            Return probabilities

        Returns
        -------
        predictions : np.ndarray
            Model predictions
        attributions : np.ndarray
            Feature attributions
        """
        predictions = self.predict(X, return_proba)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)

        attributions = self.ig.explain(X, target_class=target_class)

        return predictions, attributions.cpu().numpy()

    def get_feature_importance(
        self,
        X: Union[np.ndarray, torch.Tensor],
        aggregation: str = "mean_abs",
    ) -> Dict[str, float]:
        """
        Compute feature importance across all samples.

        Parameters
        ----------
        X : array-like
            Input features
        aggregation : str
            How to aggregate attributions: "mean_abs", "mean", "std"

        Returns
        -------
        importance : dict
            Feature importance scores
        """
        _, attributions = self.predict_with_explanations(X)

        if aggregation == "mean_abs":
            importance = np.abs(attributions).mean(axis=0)
        elif aggregation == "mean":
            importance = attributions.mean(axis=0)
        elif aggregation == "std":
            importance = attributions.std(axis=0)
        else:
            importance = np.abs(attributions).mean(axis=0)

        if self.feature_names is not None:
            return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
        else:
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.model.input_size,
            'hidden_sizes': self.model.hidden_sizes,
            'n_outputs': self.model.n_outputs,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
        }, path)

    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model = TradingModel(
            input_size=checkpoint['input_size'],
            hidden_sizes=checkpoint['hidden_sizes'],
            n_outputs=checkpoint['n_outputs'],
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_names = checkpoint.get('feature_names')
        self.training_history = checkpoint.get('training_history', {})

        self.ig = IntegratedGradients(model=self.model, device=self.device)


class MultiTaskTradingModel(nn.Module):
    """
    Multi-task trading model predicting direction, volatility, and return magnitude.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_sizes : list of int
        Sizes of shared hidden layers
    task_hidden_size : int
        Size of task-specific hidden layers
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64],
        task_hidden_size: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Shared backbone
        shared_layers = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            shared_layers.append(nn.Linear(in_features, hidden_size))
            shared_layers.append(nn.BatchNorm1d(hidden_size))
            shared_layers.append(nn.GELU())
            shared_layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        self.shared = nn.Sequential(*shared_layers)

        # Task-specific heads
        # Direction prediction (binary classification)
        self.direction_head = nn.Sequential(
            nn.Linear(in_features, task_hidden_size),
            nn.ReLU(),
            nn.Linear(task_hidden_size, 1),
        )

        # Volatility prediction (regression)
        self.volatility_head = nn.Sequential(
            nn.Linear(in_features, task_hidden_size),
            nn.ReLU(),
            nn.Linear(task_hidden_size, 1),
            nn.Softplus(),  # Ensure positive output
        )

        # Return magnitude prediction (regression)
        self.magnitude_head = nn.Sequential(
            nn.Linear(in_features, task_hidden_size),
            nn.ReLU(),
            nn.Linear(task_hidden_size, 1),
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning all task outputs."""
        shared_features = self.shared(x)

        direction = self.direction_head(shared_features)
        volatility = self.volatility_head(shared_features)
        magnitude = self.magnitude_head(shared_features)

        return direction, volatility, magnitude

    def forward_combined(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning combined output for IG."""
        direction, volatility, magnitude = self.forward(x)
        return torch.cat([direction, volatility, magnitude], dim=-1)
