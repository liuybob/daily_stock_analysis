# -*- coding: utf-8 -*-
"""
===================================
投资组合管理模块
===================================

功能：
1. 创建和管理投资组合
2. 计算组合收益和风险
3. 组合权重分配
4. 组合分析和报告

设计原则：
- 模块化设计
- 支持多种资产配置策略
- 完整的风险分析
- 易于扩展
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """再平衡频率枚举"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class WeightMethod(Enum):
    """权重分配方法枚举"""
    EQUAL_WEIGHT = "equal_weight"              # 等权重
    MARKET_CAP = "market_cap"                  # 市值加权
    VOLATILITY_TARGET = "volatility_target"    # 目标波动率
    RISK_PARITY = "risk_parity"                # 风险平价
    CUSTOM = "custom"                          # 自定义


@dataclass
class Position:
    """持仓信息"""
    symbol: str              # 股票代码
    shares: float            # 持股数量
    entry_price: float       # 入场价格
    current_price: float     # 当前价格
    market_value: float      # 市值
    weight: float            # 权重
    entry_date: str          # 入场日期

    @property
    def pnl(self) -> float:
        """盈亏"""
        return (self.current_price - self.entry_price) * self.shares

    @property
    def pnl_percentage(self) -> float:
        """盈亏百分比"""
        return (self.current_price / self.entry_price - 1) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'shares': self.shares,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'weight': self.weight,
            'entry_date': self.entry_date,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
        }


@dataclass
class PortfolioMetrics:
    """组合指标"""
    total_value: float = 0.0          # 总市值
    cash: float = 0.0                 # 现金
    total_return: float = 0.0         # 总收益率
    daily_return: float = 0.0         # 日收益率
    annualized_return: float = 0.0    # 年化收益率
    volatility: float = 0.0           # 波动率
    sharpe_ratio: float = 0.0         # 夏普比率
    max_drawdown: float = 0.0         # 最大回撤
    win_rate: float = 0.0             # 胜率

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'total_return': self.total_return,
            'daily_return': self.daily_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
        }


@dataclass
class PortfolioConfig:
    """组合配置"""
    # 组合名称
    name: str = "Default Portfolio"

    # 初始资金
    initial_capital: float = 100000.0

    # 最大持仓数量
    max_positions: int = 10

    # 单个股票最大权重
    max_single_weight: float = 0.3

    # 权重分配方法
    weight_method: WeightMethod = WeightMethod.EQUAL_WEIGHT

    # 再平衡频率
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY

    # 止损阈值
    stop_loss_threshold: float = -0.05

    # 止盈阈值
    take_profit_threshold: float = 0.15


class Portfolio:
    """
    投资组合类

    功能：
    1. 管理持仓
    2. 计算收益和风险
    3. 再平衡
    4. 分析和报告
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        初始化投资组合

        Args:
            config: 组合配置
        """
        self.config = config or PortfolioConfig()
        self.positions: Dict[str, Position] = {}
        self.cash = self.config.initial_capital
        self.initial_capital = self.config.initial_capital
        self.transaction_history: List[Dict[str, Any]] = []
        self.daily_values: List[Tuple[str, float]] = []
        self.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def add_position(
        self,
        symbol: str,
        shares: float,
        price: float,
        date: Optional[str] = None
    ) -> None:
        """
        添加持仓

        Args:
            symbol: 股票代码
            shares: 股数
            price: 价格
            date: 日期
        """
        if symbol in self.positions:
            logger.warning(f"股票 {symbol} 已存在，将更新持仓")
            self.update_position(symbol, shares, price)
            return

        # 检查资金是否足够
        cost = shares * price
        if cost > self.cash:
            raise ValueError(f"资金不足，需要 {cost:.2f}，可用 {self.cash:.2f}")

        # 检查持仓数量限制
        if len(self.positions) >= self.config.max_positions:
            raise ValueError(f"达到最大持仓数量 {self.config.max_positions}")

        # 计算权重
        total_value = self.get_total_value()
        weight = cost / total_value if total_value > 0 else 0

        if weight > self.config.max_single_weight:
            raise ValueError(f"单个股票权重超限，最大 {self.config.max_single_weight * 100}%")

        # 创建持仓
        position = Position(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            current_price=price,
            market_value=cost,
            weight=weight,
            entry_date=date or datetime.now().strftime('%Y-%m-%d')
        )

        self.positions[symbol] = position
        self.cash -= cost

        # 记录交易
        self._record_transaction('BUY', symbol, shares, price, date)

        logger.info(f"买入 {symbol}: {shares}股 @ {price:.2f}, 权重 {weight*100:.2f}%")

    def remove_position(
        self,
        symbol: str,
        price: float,
        date: Optional[str] = None
    ) -> float:
        """
        移除持仓

        Args:
            symbol: 股票代码
            price: 卖出价格
            date: 日期

        Returns:
            卖出金额
        """
        if symbol not in self.positions:
            raise ValueError(f"股票 {symbol} 不在持仓中")

        position = self.positions[symbol]
        proceeds = position.shares * price
        pnl = proceeds - position.market_value

        self.cash += proceeds
        del self.positions[symbol]

        # 记录交易
        self._record_transaction('SELL', symbol, position.shares, price, date, pnl)

        logger.info(f"卖出 {symbol}: {position.shares}股 @ {price:.2f}, 盈亏 {pnl:.2f}")

        return proceeds

    def update_position(
        self,
        symbol: str,
        shares: Optional[float] = None,
        price: Optional[float] = None
    ) -> None:
        """
        更新持仓

        Args:
            symbol: 股票代码
            shares: 新股数（可选）
            price: 新价格（可选）
        """
        if symbol not in self.positions:
            raise ValueError(f"股票 {symbol} 不在持仓中")

        position = self.positions[symbol]

        if price is not None:
            position.current_price = price
            position.market_value = position.shares * price

        if shares is not None:
            # 如果股数变化，需要调整现金
            shares_diff = shares - position.shares
            cost = shares_diff * position.current_price
            self.cash -= cost
            position.shares = shares
            position.market_value = shares * position.current_price

        # 重新计算权重
        total_value = self.get_total_value()
        position.weight = position.market_value / total_value if total_value > 0 else 0

    def get_total_value(self) -> float:
        """
        获取总市值

        Returns:
            总市值（持仓市值 + 现金）
        """
        positions_value = sum(p.market_value for p in self.positions.values())
        return positions_value + self.cash

    def get_metrics(self) -> PortfolioMetrics:
        """
        计算组合指标

        Returns:
            组合指标
        """
        total_value = self.get_total_value()

        # 总收益率
        total_return = (total_value / self.initial_capital) - 1

        # 日收益率（简化计算）
        if len(self.daily_values) > 1:
            recent_values = [v for _, v in self.daily_values[-10:]]
            if len(recent_values) > 1:
                daily_return = (recent_values[-1] / recent_values[-2]) - 1
            else:
                daily_return = 0
        else:
            daily_return = 0

        # 年化收益率（简化）
        n_days = max(len(self.daily_values), 1)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if total_return != 0 else 0

        # 波动率（简化）
        if len(self.daily_values) > 1:
            values = [v for _, v in self.daily_values]
            returns = pd.Series(values).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        else:
            volatility = 0

        # 夏普比率（简化，假设无风险利率为3%）
        risk_free_rate = 0.03
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0

        # 最大回撤
        if len(self.daily_values) > 1:
            values = [v for _, v in self.daily_values]
            cum_returns = pd.Series(values) / self.initial_capital
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        # 胜率
        winning_trades = [t for t in self.transaction_history if t.get('pnl', 0) > 0]
        total_trades = len([t for t in self.transaction_history if t.get('pnl', 0) != 0])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        return PortfolioMetrics(
            total_value=total_value,
            cash=self.cash,
            total_return=total_return,
            daily_return=daily_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
        )

    def rebalance(self, target_weights: Dict[str, float], prices: Dict[str, float]) -> None:
        """
        再平衡组合

        Args:
            target_weights: 目标权重 {symbol: weight}
            prices: 当前价格 {symbol: price}
        """
        logger.info("开始再平衡...")

        total_value = self.get_total_value()

        # 调整现有持仓
        for symbol, position in list(self.positions.items()):
            if symbol not in target_weights:
                # 不在目标权重中，卖出
                self.remove_position(symbol, prices[symbol])
            else:
                # 调整到目标权重
                target_value = total_value * target_weights[symbol]
                current_value = position.market_value

                if abs(target_value - current_value) / current_value > 0.05:  # 5%阈值
                    target_shares = target_value / prices[symbol]
                    shares_diff = target_shares - position.shares

                    if shares_diff > 0:
                        # 买入
                        cost = shares_diff * prices[symbol]
                        if cost <= self.cash:
                            self.update_position(symbol, target_shares, prices[symbol])
                    else:
                        # 卖出
                        self.remove_position(symbol, prices[symbol])
                        self.add_position(symbol, target_shares, prices[symbol])

        # 添加新持仓
        for symbol, weight in target_weights.items():
            if symbol not in self.positions:
                target_value = total_value * weight
                shares = target_value / prices[symbol]

                if shares * prices[symbol] <= self.cash:
                    self.add_position(symbol, shares, prices[symbol])

        logger.info("再平衡完成")

    def update_daily_value(self, date: Optional[str] = None) -> None:
        """
        更新每日市值

        Args:
            date: 日期
        """
        date = date or datetime.now().strftime('%Y-%m-%d')
        total_value = self.get_total_value()
        self.daily_values.append((date, total_value))

    def _record_transaction(
        self,
        action: str,
        symbol: str,
        shares: float,
        price: float,
        date: Optional[str] = None,
        pnl: float = 0.0
    ) -> None:
        """
        记录交易

        Args:
            action: 操作类型（BUY/SELL）
            symbol: 股票代码
            shares: 股数
            price: 价格
            date: 日期
            pnl: 盈亏
        """
        transaction = {
            'date': date or datetime.now().strftime('%Y-%m-%d'),
            'action': action,
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'value': shares * price,
            'pnl': pnl
        }

        self.transaction_history.append(transaction)

    def get_positions_summary(self) -> pd.DataFrame:
        """
        获取持仓摘要

        Returns:
            持仓摘要DataFrame
        """
        if not self.positions:
            return pd.DataFrame()

        data = [p.to_dict() for p in self.positions.values()]
        return pd.DataFrame(data).sort_values('weight', ascending=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            组合信息字典
        """
        return {
            'name': self.config.name,
            'created_at': self.created_at,
            'initial_capital': self.initial_capital,
            'current_value': self.get_total_value(),
            'cash': self.cash,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'metrics': self.get_metrics().to_dict(),
            'num_positions': len(self.positions),
        }


if __name__ == "__main__":
    # 测试组合管理
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建组合
    config = PortfolioConfig(
        name="Test Portfolio",
        initial_capital=100000,
        max_positions=5,
        max_single_weight=0.3
    )

    portfolio = Portfolio(config)

    # 添加持仓
    portfolio.add_position("AAPL", 100, 150)
    portfolio.add_position("MSFT", 50, 300)
    portfolio.add_position("GOOGL", 20, 2500)

    # 更新价格
    portfolio.update_position("AAPL", price=155)
    portfolio.update_position("MSFT", price=310)
    portfolio.update_position("GOOGL", price=2600)

    # 计算指标
    metrics = portfolio.get_metrics()

    print("\n组合指标:")
    print(f"总市值: {metrics.total_value:.2f}")
    print(f"总收益率: {metrics.total_return * 100:.2f}%")
    print(f"夏普比率: {metrics.sharpe_ratio:.2f}")

    print("\n持仓摘要:")
    print(portfolio.get_positions_summary())
