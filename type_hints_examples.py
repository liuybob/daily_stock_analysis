# -*- coding: utf-8 -*-
"""
类型提示增强模块

本模块展示如何为函数添加完整的类型提示
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


def calculate_returns(
    prices: pd.Series,
    periods: int = 1
) -> pd.Series:
    """
    计算收益率

    Args:
        prices: 价格序列
        periods: 周期数，默认为1

    Returns:
        收益率序列
    """
    return prices.pct_change(periods)


def calculate_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> float:
    """
    计算波动率

    Args:
        returns: 收益率序列
        window: 滚动窗口大小
        annualize: 是否年化

    Returns:
        波动率值
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol.iloc[-1]


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    annualize: bool = True
) -> float:
    """
    计算夏普比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        annualize: 是否年化

    Returns:
        夏普比率
    """
    excess_returns = returns - risk_free_rate / 252
    sharpe = excess_returns.mean() / excess_returns.std()
    if annualize:
        sharpe = sharpe * np.sqrt(252)
    return sharpe


def calculate_max_drawdown(
    prices: pd.Series
) -> Tuple[float, int, int]:
    """
    计算最大回撤

    Args:
        prices: 价格序列

    Returns:
        (最大回撤, 回撤开始索引, 回撤结束索引)
    """
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    max_dd = drawdown.min()

    # 找到回撤的开始和结束
    min_idx = drawdown.idxmin()
    start_idx = prices[:min_idx].idxmax()

    return max_dd, start_idx, min_idx


def analyze_portfolio(
    symbols: List[str],
    weights: List[float],
    returns: pd.DataFrame
) -> Dict[str, Any]:
    """
    分析投资组合

    Args:
        symbols: 股票代码列表
        weights: 权重列表
        returns: 收益率DataFrame

    Returns:
        包含组合指标的字典
    """
    # 计算组合收益
    portfolio_returns = (returns * weights).sum(axis=1)

    # 计算指标
    total_return = portfolio_returns.sum()
    volatility = calculate_volatility(portfolio_returns)
    sharpe = calculate_sharpe_ratio(portfolio_returns)
    max_dd, _, _ = calculate_max_drawdown((1 + portfolio_returns).cumprod())

    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'annual_return': portfolio_returns.mean() * 252,
    }
