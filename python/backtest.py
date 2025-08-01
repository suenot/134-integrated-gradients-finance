"""
Backtesting Framework with Attribution Logging

This module provides backtesting capabilities for trading models
with integrated gradients attribution tracking.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: Optional[float]
    attributions: Optional[np.ndarray]
    prediction_confidence: float


@dataclass
class BacktestResults:
    """Container for backtest results."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    equity_curve: pd.Series
    trades: List[Trade]
    attribution_analysis: Dict


class IGBacktester:
    """
    Backtester with integrated gradients attribution logging.

    Tracks feature attributions for each trade to analyze
    which features drive profitable vs. losing trades.

    Parameters
    ----------
    model : TradingModelWithIG
        Trading model with IG capability
    initial_capital : float
        Starting capital
    transaction_cost : float
        Transaction cost as fraction of trade value
    position_size : float
        Position size as fraction of capital
    log_attributions : bool
        Whether to log attributions for each trade
    attribution_threshold : float
        Only log features with |attribution| > threshold
    """

    def __init__(
        self,
        model,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        position_size: float = 0.1,
        log_attributions: bool = True,
        attribution_threshold: float = 0.05,
    ):
        self.model = model
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        self.log_attributions = log_attributions
        self.attribution_threshold = attribution_threshold

        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.attribution_log: List[Dict] = []

    def _generate_signal(
        self,
        features: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[int, float, Optional[np.ndarray]]:
        """
        Generate trading signal from model prediction.

        Returns
        -------
        signal : int
            Trading signal: 1 (long), -1 (short), 0 (neutral)
        confidence : float
            Prediction confidence
        attributions : np.ndarray or None
            Feature attributions if logging enabled
        """
        if self.log_attributions:
            pred, attr = self.model.predict_with_explanations(
                features.reshape(1, -1),
                return_proba=True
            )
            attributions = attr[0]
        else:
            pred = self.model.predict(features.reshape(1, -1), return_proba=True)
            attributions = None

        confidence = pred[0, 0] if pred.ndim > 1 else pred[0]

        if confidence > threshold:
            signal = 1  # Long
        elif confidence < (1 - threshold):
            signal = -1  # Short
        else:
            signal = 0  # Neutral

        return signal, confidence, attributions

    def run(
        self,
        data: pd.DataFrame,
        features_cols: List[str],
        price_col: str = "close",
        timestamp_col: str = "timestamp",
        signal_threshold: float = 0.55,
        holding_period: int = 1,
    ) -> BacktestResults:
        """
        Run backtest on historical data.

        Parameters
        ----------
        data : pd.DataFrame
            Historical data with features and prices
        features_cols : list of str
            Column names for features
        price_col : str
            Column name for price
        timestamp_col : str
            Column name for timestamp
        signal_threshold : float
            Threshold for generating trading signals
        holding_period : int
            Number of bars to hold position

        Returns
        -------
        results : BacktestResults
            Backtest results
        """
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.attribution_log = []

        capital = self.initial_capital
        position = None
        bars_held = 0

        for i in range(len(data) - holding_period):
            row = data.iloc[i]
            features = row[features_cols].values.astype(np.float32)
            current_price = row[price_col]
            current_time = row[timestamp_col] if timestamp_col in row else i

            # Check if we need to close position
            if position is not None:
                bars_held += 1
                if bars_held >= holding_period:
                    # Close position
                    exit_price = current_price
                    pnl = position.direction * (exit_price - position.entry_price) / position.entry_price
                    pnl -= self.transaction_cost  # Exit cost

                    position.exit_time = current_time
                    position.exit_price = exit_price
                    position.pnl = pnl * position.size * capital

                    capital += position.pnl
                    self.trades.append(position)
                    position = None
                    bars_held = 0

            # Generate signal if no position
            if position is None:
                signal, confidence, attributions = self._generate_signal(
                    features, signal_threshold
                )

                if signal != 0:
                    # Open position
                    trade_capital = capital * self.position_size
                    entry_cost = trade_capital * self.transaction_cost

                    position = Trade(
                        entry_time=current_time,
                        exit_time=None,
                        direction=signal,
                        entry_price=current_price,
                        exit_price=None,
                        size=self.position_size,
                        pnl=None,
                        attributions=attributions,
                        prediction_confidence=confidence,
                    )

                    capital -= entry_cost

                    if self.log_attributions and attributions is not None:
                        self.attribution_log.append({
                            'time': current_time,
                            'direction': signal,
                            'confidence': confidence,
                            'attributions': attributions,
                        })

            self.equity_curve.append(capital)

        # Close any remaining position
        if position is not None:
            exit_price = data.iloc[-1][price_col]
            pnl = position.direction * (exit_price - position.entry_price) / position.entry_price
            pnl -= self.transaction_cost

            position.exit_time = data.iloc[-1][timestamp_col] if timestamp_col in data.columns else len(data) - 1
            position.exit_price = exit_price
            position.pnl = pnl * position.size * capital

            capital += position.pnl
            self.trades.append(position)

        # Calculate metrics
        results = self._calculate_metrics()
        return results

    def _calculate_metrics(self) -> BacktestResults:
        """Calculate backtest performance metrics."""
        equity = pd.Series(self.equity_curve)
        returns = equity.pct_change().dropna()

        # Total return
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # Sharpe ratio (assuming 252 trading days)
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio

        # Maximum drawdown
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()

        # Trade statistics
        if self.trades:
            pnls = [t.pnl for t in self.trades if t.pnl is not None]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            win_rate = len(wins) / len(pnls) if pnls else 0
            profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
            avg_trade_pnl = np.mean(pnls) if pnls else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_pnl = 0

        # Attribution analysis
        attribution_analysis = self._analyze_attributions()

        return BacktestResults(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            avg_trade_pnl=avg_trade_pnl,
            equity_curve=equity,
            trades=self.trades,
            attribution_analysis=attribution_analysis,
        )

    def _analyze_attributions(self) -> Dict:
        """Analyze attributions for winning vs. losing trades."""
        if not self.trades or not self.log_attributions:
            return {}

        # Separate winning and losing trades
        winning_attrs = []
        losing_attrs = []

        for trade in self.trades:
            if trade.attributions is None or trade.pnl is None:
                continue

            if trade.pnl > 0:
                winning_attrs.append(trade.attributions)
            else:
                losing_attrs.append(trade.attributions)

        analysis = {
            'n_winning_trades': len(winning_attrs),
            'n_losing_trades': len(losing_attrs),
        }

        if winning_attrs:
            winning_attrs = np.array(winning_attrs)
            analysis['winning_mean_attr'] = winning_attrs.mean(axis=0)
            analysis['winning_std_attr'] = winning_attrs.std(axis=0)

        if losing_attrs:
            losing_attrs = np.array(losing_attrs)
            analysis['losing_mean_attr'] = losing_attrs.mean(axis=0)
            analysis['losing_std_attr'] = losing_attrs.std(axis=0)

        if winning_attrs and losing_attrs:
            # Discriminative features (difference between winners and losers)
            analysis['discriminative_attr'] = (
                analysis['winning_mean_attr'] - analysis['losing_mean_attr']
            )

        return analysis

    def get_top_features(
        self,
        n_features: int = 10,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get top features by attribution for winning trades.

        Parameters
        ----------
        n_features : int
            Number of top features to return
        feature_names : list of str, optional
            Feature names

        Returns
        -------
        df : pd.DataFrame
            Top features with attributions
        """
        analysis = self._analyze_attributions()

        if 'winning_mean_attr' not in analysis:
            return pd.DataFrame()

        attr = analysis['winning_mean_attr']

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(attr))]

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(attr))[::-1][:n_features]

        df = pd.DataFrame({
            'feature': [feature_names[i] for i in sorted_idx],
            'winning_attr': attr[sorted_idx],
        })

        if 'losing_mean_attr' in analysis:
            losing_attr = analysis['losing_mean_attr']
            df['losing_attr'] = losing_attr[sorted_idx]
            df['discriminative'] = df['winning_attr'] - df['losing_attr']

        return df


class AttributionFilteredBacktester(IGBacktester):
    """
    Backtester that filters trades based on attribution patterns.

    Rejects trades where attributions don't meet certain criteria,
    such as excessive dependence on volatile features.

    Parameters
    ----------
    model : TradingModelWithIG
        Trading model with IG capability
    initial_capital : float
        Starting capital
    feature_constraints : dict
        Constraints on feature attributions
        Example: {'sentiment': 0.3}  # max attribution for sentiment
    min_fundamental_support : float
        Minimum combined attribution from fundamental features
    fundamental_features : list of str
        Names of fundamental features
    """

    def __init__(
        self,
        model,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        position_size: float = 0.1,
        feature_constraints: Optional[Dict[str, float]] = None,
        min_fundamental_support: float = 0.0,
        fundamental_features: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__(
            model=model,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            position_size=position_size,
            log_attributions=True,
        )

        self.feature_constraints = feature_constraints or {}
        self.min_fundamental_support = min_fundamental_support
        self.fundamental_features = fundamental_features or []
        self.feature_names = feature_names or []

        self.rejected_trades: List[Dict] = []

    def _should_execute_trade(
        self,
        attributions: np.ndarray,
        confidence: float,
    ) -> Tuple[bool, str]:
        """
        Check if trade should be executed based on attribution criteria.

        Returns
        -------
        should_execute : bool
            Whether to execute the trade
        reason : str
            Reason for decision
        """
        if attributions is None:
            return True, "No attributions available"

        # Check feature constraints
        for feature_name, max_attr in self.feature_constraints.items():
            if feature_name in self.feature_names:
                idx = self.feature_names.index(feature_name)
                if abs(attributions[idx]) > max_attr:
                    return False, f"Excessive dependence on {feature_name}: {attributions[idx]:.3f}"

        # Check fundamental support
        if self.fundamental_features and self.min_fundamental_support > 0:
            fundamental_attr = 0.0
            for feature_name in self.fundamental_features:
                if feature_name in self.feature_names:
                    idx = self.feature_names.index(feature_name)
                    fundamental_attr += abs(attributions[idx])

            if fundamental_attr < self.min_fundamental_support:
                return False, f"Insufficient fundamental support: {fundamental_attr:.3f}"

        return True, "Trade approved"

    def _generate_signal(
        self,
        features: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[int, float, Optional[np.ndarray]]:
        """Generate signal with attribution filtering."""
        signal, confidence, attributions = super()._generate_signal(features, threshold)

        if signal != 0 and attributions is not None:
            should_execute, reason = self._should_execute_trade(attributions, confidence)

            if not should_execute:
                self.rejected_trades.append({
                    'original_signal': signal,
                    'confidence': confidence,
                    'attributions': attributions.copy(),
                    'reason': reason,
                })
                return 0, confidence, attributions  # Return neutral signal

        return signal, confidence, attributions


def calculate_trading_metrics(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """
    Calculate common trading performance metrics.

    Parameters
    ----------
    returns : array-like
        Return series
    risk_free_rate : float
        Annual risk-free rate

    Returns
    -------
    metrics : dict
        Performance metrics
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {}

    # Daily risk-free rate
    rf_daily = risk_free_rate / 252

    # Excess returns
    excess_returns = returns - rf_daily

    metrics = {
        'total_return': np.prod(1 + returns) - 1,
        'annualized_return': np.mean(returns) * 252,
        'annualized_volatility': np.std(returns) * np.sqrt(252),
        'sharpe_ratio': np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
        'max_drawdown': np.min(np.minimum.accumulate(np.cumprod(1 + returns)) / np.maximum.accumulate(np.cumprod(1 + returns)) - 1),
        'skewness': pd.Series(returns).skew(),
        'kurtosis': pd.Series(returns).kurtosis(),
    }

    # Sortino ratio
    downside = returns[returns < 0]
    if len(downside) > 0:
        downside_std = np.std(downside) * np.sqrt(252)
        metrics['sortino_ratio'] = (np.mean(returns) * 252 - risk_free_rate) / downside_std if downside_std > 0 else 0
    else:
        metrics['sortino_ratio'] = metrics['sharpe_ratio']

    # Calmar ratio
    if metrics['max_drawdown'] != 0:
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = 0

    return metrics
