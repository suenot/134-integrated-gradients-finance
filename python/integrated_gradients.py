"""
Integrated Gradients Implementation for Financial Models

This module implements the Integrated Gradients attribution method as described in:
"Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Callable
import matplotlib.pyplot as plt


class IntegratedGradients:
    """
    Integrated Gradients explainer for PyTorch models.

    Computes feature attributions by integrating gradients along a path
    from a baseline to the input.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to explain
    n_steps : int
        Number of steps for numerical integration (default: 200)
    baseline_type : str
        Type of baseline: "zero", "mean", "random", or "custom"
    device : str
        Device to run computations on

    Examples
    --------
    >>> model = TradingModel(input_size=20, hidden_sizes=[64, 32])
    >>> ig = IntegratedGradients(model, n_steps=200)
    >>> attributions = ig.explain(input_tensor)
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 200,
        baseline_type: str = "zero",
        device: Optional[str] = None,
    ):
        self.model = model
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _get_baseline(
        self,
        inputs: torch.Tensor,
        custom_baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate baseline based on specified type."""
        if custom_baseline is not None:
            return custom_baseline.to(self.device)

        if self.baseline_type == "zero":
            return torch.zeros_like(inputs)
        elif self.baseline_type == "mean":
            return inputs.mean(dim=0, keepdim=True).expand_as(inputs)
        elif self.baseline_type == "random":
            return torch.randn_like(inputs) * 0.01
        else:
            return torch.zeros_like(inputs)

    def _compute_gradients(
        self,
        inputs: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """Compute gradients of output with respect to inputs."""
        inputs = inputs.clone().detach().requires_grad_(True)

        outputs = self.model(inputs)

        if target_class is not None:
            if outputs.dim() > 1:
                outputs = outputs[:, target_class]
            else:
                outputs = outputs.sum()
        else:
            if outputs.dim() > 1:
                outputs = outputs[:, 0]
            outputs = outputs.sum()

        self.model.zero_grad()
        outputs.backward()

        return inputs.grad

    def explain(
        self,
        inputs: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        return_convergence_delta: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Compute integrated gradients for the given inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, n_features) or (n_features,)
        baseline : torch.Tensor, optional
            Custom baseline tensor
        target_class : int, optional
            Target class index for multi-output models
        return_convergence_delta : bool
            If True, also return the convergence delta (completeness error)

        Returns
        -------
        attributions : torch.Tensor
            Attribution scores for each feature
        delta : float (optional)
            Convergence delta if return_convergence_delta=True
        """
        # Ensure inputs are on correct device
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(self.device)

        # Handle 1D inputs
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        # Get baseline
        baseline = self._get_baseline(inputs, baseline)

        # Generate interpolated inputs along the path
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=self.device)

        # Compute gradients at each step
        scaled_gradients = []
        for alpha in alphas[1:]:  # Skip alpha=0
            interpolated = baseline + alpha * (inputs - baseline)
            grad = self._compute_gradients(interpolated, target_class)
            scaled_gradients.append(grad)

        # Stack and average gradients
        gradients = torch.stack(scaled_gradients, dim=0)
        avg_gradients = gradients.mean(dim=0)

        # Compute attributions: (input - baseline) * avg_gradients
        attributions = (inputs - baseline) * avg_gradients

        if return_convergence_delta:
            # Compute convergence delta (completeness check)
            with torch.no_grad():
                pred_input = self.model(inputs)
                pred_baseline = self.model(baseline)

                if target_class is not None and pred_input.dim() > 1:
                    pred_diff = pred_input[:, target_class] - pred_baseline[:, target_class]
                else:
                    if pred_input.dim() > 1:
                        pred_diff = pred_input[:, 0] - pred_baseline[:, 0]
                    else:
                        pred_diff = pred_input - pred_baseline

                attr_sum = attributions.sum(dim=-1)
                delta = (pred_diff - attr_sum).abs().mean().item()

            return attributions, delta

        return attributions

    def explain_batch(
        self,
        inputs: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Compute integrated gradients for a batch of inputs efficiently.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (n_samples, n_features)
        baseline : torch.Tensor, optional
            Custom baseline tensor
        target_class : int, optional
            Target class index
        batch_size : int
            Batch size for processing

        Returns
        -------
        attributions : torch.Tensor
            Attribution scores of shape (n_samples, n_features)
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(self.device)

        n_samples = inputs.shape[0]
        all_attributions = []

        for i in range(0, n_samples, batch_size):
            batch = inputs[i:i + batch_size]
            batch_baseline = None
            if baseline is not None:
                batch_baseline = baseline[i:i + batch_size]

            attributions = self.explain(batch, batch_baseline, target_class)
            all_attributions.append(attributions)

        return torch.cat(all_attributions, dim=0)

    def plot_attributions(
        self,
        attributions: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Integrated Gradients Attribution",
    ) -> plt.Figure:
        """
        Plot feature attributions as a horizontal bar chart.

        Parameters
        ----------
        attributions : torch.Tensor
            Attribution scores
        feature_names : list of str, optional
            Names of features
        top_k : int, optional
            Show only top k features by absolute attribution
        figsize : tuple
            Figure size
        title : str
            Plot title

        Returns
        -------
        fig : matplotlib.Figure
            The figure object
        """
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.detach().cpu().numpy()

        if attributions.ndim > 1:
            attributions = attributions.mean(axis=0)

        n_features = len(attributions)
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(attributions))[::-1]

        if top_k is not None:
            sorted_idx = sorted_idx[:top_k]

        sorted_attrs = attributions[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        colors = ['green' if a > 0 else 'red' for a in sorted_attrs]
        y_pos = np.arange(len(sorted_attrs))

        ax.barh(y_pos, sorted_attrs, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()
        ax.set_xlabel('Attribution')
        ax.set_title(title)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        return fig


class ExpectedIntegratedGradients(IntegratedGradients):
    """
    Expected Integrated Gradients - averages IG over multiple baselines.

    This provides more robust attributions by marginalizing over the baseline choice.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to explain
    n_steps : int
        Number of steps for numerical integration
    n_baselines : int
        Number of baselines to average over
    baseline_distribution : str
        Distribution for sampling baselines: "training", "gaussian", "uniform"
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 200,
        n_baselines: int = 10,
        baseline_distribution: str = "gaussian",
        device: Optional[str] = None,
    ):
        super().__init__(model, n_steps, "custom", device)
        self.n_baselines = n_baselines
        self.baseline_distribution = baseline_distribution
        self.training_data = None

    def set_training_data(self, data: torch.Tensor):
        """Set training data for sampling baselines."""
        self.training_data = data.to(self.device)

    def _sample_baselines(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """Sample multiple baselines."""
        baselines = []

        for _ in range(self.n_baselines):
            if self.baseline_distribution == "training" and self.training_data is not None:
                idx = torch.randint(0, len(self.training_data), (inputs.shape[0],))
                baseline = self.training_data[idx]
            elif self.baseline_distribution == "gaussian":
                baseline = torch.randn_like(inputs) * 0.1
            elif self.baseline_distribution == "uniform":
                baseline = torch.rand_like(inputs) * 2 - 1
            else:
                baseline = torch.zeros_like(inputs)

            baselines.append(baseline)

        return baselines

    def explain(
        self,
        inputs: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        return_convergence_delta: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Compute expected integrated gradients by averaging over multiple baselines.
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs.to(self.device)

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        # Sample baselines
        baselines = self._sample_baselines(inputs)

        # Compute IG for each baseline
        all_attributions = []
        all_deltas = []

        for bl in baselines:
            if return_convergence_delta:
                attr, delta = super().explain(inputs, bl, target_class, True)
                all_deltas.append(delta)
            else:
                attr = super().explain(inputs, bl, target_class)
            all_attributions.append(attr)

        # Average attributions
        avg_attributions = torch.stack(all_attributions, dim=0).mean(dim=0)

        if return_convergence_delta:
            avg_delta = np.mean(all_deltas)
            return avg_attributions, avg_delta

        return avg_attributions
