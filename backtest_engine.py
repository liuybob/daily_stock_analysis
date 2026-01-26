# -*- coding: utf-8 -*-
"""
===================================
回测系统 - Backtesting Engine
===================================

功能：
1. 回测引擎实现
2. 绩效指标计算
3. 参数优化
4. 历史验证交易理念

核心组件：
- BacktestEngine: 回测引擎
- PerformanceCalculator: 绩效计算器
- ParameterOptimizer: 参数优化器
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TradeType(Enum):
    """交易类型"""
    BUY = "买入"
    SELL = "卖出"


class TradeStatus(Enum):
    """交易状态"""
    OPEN = "开仓"
    CLOSED = "平仓"


@dataclass
class Trade:
    """单笔交易记录"""
    trade_id: str
    code: str
    trade_type: TradeType
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    quantity: int = 100  # 默认100股
    status: TradeStatus = TradeStatus.OPEN
    profit_loss: float = 0.0  # 盈亏金额
    profit_loss_pct: float = 0.0  # 盈亏百分比
    holding_days: int = 0  # 持有天数

    def close_position(self, exit_date: str, exit_price: float):
        """平仓"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.status = TradeStatus.CLOSED

        # 计算盈亏
        self.profit_loss = (exit_price - self.entry_price) * self.quantity
        self.profit_loss_pct = (exit_price - self.entry_price) / self.entry_price * 100


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0  # 初始资金
    max_position_pct: float = 0.3  # 单只股票最大仓位（30%）
    stop_loss_pct: float = 0.05  # 止损百分比（5%）
    take_profit_pct: float = 0.15  # 止盈百分比（15%）
    max_holding_days: int = 30  # 最大持有天数
    commission_rate: float = 0.0003  # 手续费率（万三）


@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig

    # 交易记录
    trades: List[Trade] = field(default_factory=list)

    # 绩效指标
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_return: float = 0.0  # 总收益率
    annual_return: float = 0.0  # 年化收益率
    max_drawdown: float = 0.0  # 最大回撤
    sharpe_ratio: float = 0.0  # 夏普比率

    total_profit: float = 0.0  # 总盈利
    total_loss: float = 0.0  # 总亏损
    profit_factor: float = 0.0  # 盈亏比

    avg_profit: float = 0.0  # 平均盈利
    avg_loss: float = 0.0  # 平均亏损
    avg_holding_days: float = 0.0  # 平均持有天数

    equity_curve: List[float] = field(default_factory=list)  # 净值曲线


class BacktestEngine:
    """
    回测引擎

    基于趋势交易策略进行历史回测
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测引擎

        Args:
            config: 回测配置
        """
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """重置回测状态"""
        self.capital = self.config.initial_capital
        self.position = {}  # 当前持仓 {code: quantity}
        self.trades = []  # 所有交易记录
        self.equity_curve = [self.config.initial_capital]
        self.current_date = None

    def run_backtest(self, df: pd.DataFrame, code: str) -> BacktestResult:
        """
        运行回测

        Args:
            df: 股票数据，必须包含技术指标和买入信号
            code: 股票代码

        Returns:
            BacktestResult 回测结果
        """
        logger.info(f"{code}: 开始回测，数据量 {len(df)} 天")

        self.reset()

        if df is None or df.empty or len(df) < 20:
            logger.warning(f"{code}: 数据不足，无法回测")
            return self._create_result(code)

        try:
            # 确保数据按日期排序
            df = df.sort_values('date').reset_index(drop=True)

            # 遍历每个交易日
            for idx, row in df.iterrows():
                self.current_date = row['date']

                # 1. 检查止损止盈
                self._check_exit_signals(row, code)

                # 2. 检查买入信号
                if code not in self.position or self.position[code] == 0:
                    self._check_entry_signals(row, df, idx, code)

                # 3. 更新净值曲线
                self._update_equity(row, code)

            # 4. 平掉所有剩余持仓
            self._close_all_positions(df, code)

            # 5. 计算绩效指标
            result = self._create_result(code)
            result = self._calculate_performance_metrics(result)

            logger.info(f"{code}: 回测完成，总收益率 {result.total_return:.2f}%")

            return result

        except Exception as e:
            logger.error(f"{code}: 回测失败 - {str(e)}")
            return self._create_result(code)

    def _check_entry_signals(self, row: pd.Series, df: pd.DataFrame,
                            idx: int, code: str):
        """
        检查入场信号

        入场条件：
        - 买入信号为 强烈买入 或 买入
        - 有足够资金
        """
        # 检查是否有买入信号
        buy_signal = row.get('buy_signal', '')

        if buy_signal not in ['强烈买入', '买入']:
            return

        # 检查是否有足够资金
        price = row['close']
        max_position_value = self.capital * self.config.max_position_pct

        if max_position_value < price * 100:  # 至少买100股
            logger.debug(f"{code} {row['date']}: 资金不足，无法开仓")
            return

        # 计算买入数量（100股的整数倍）
        quantity = int(max_position_value / price / 100) * 100

        if quantity <= 0:
            return

        # 开仓
        trade = Trade(
            trade_id=f"{code}_{row['date']}_{idx}",
            code=code,
            trade_type=TradeType.BUY,
            entry_date=str(row['date']),
            entry_price=price,
            quantity=quantity,
            status=TradeStatus.OPEN
        )

        self.trades.append(trade)
        self.position[code] = self.position.get(code, 0) + quantity

        # 扣除资金（含手续费）
        cost = price * quantity * (1 + self.config.commission_rate)
        self.capital -= cost

        logger.debug(f"{code} {row['date']}: 开仓 {quantity}股 @{price:.2f}")

    def _check_exit_signals(self, row: pd.Series, code: str):
        """
        检查出场信号

        出场条件：
        - 止损：亏损超过止损百分比
        - 止盈：盈利超过止盈百分比
        - 超过最大持有天数
        - 卖出信号
        """
        if code not in self.position or self.position[code] == 0:
            return

        # 查找该股票的开仓交易
        open_trades = [t for t in self.trades
                      if t.code == code and t.status == TradeStatus.OPEN]

        for trade in open_trades:
            current_price = row['close']
            entry_price = trade.entry_price

            # 计算盈亏百分比
            profit_pct = (current_price - entry_price) / entry_price * 100

            # 计算持有天数
            entry_date = pd.to_datetime(trade.entry_date)
            current_date = pd.to_datetime(row['date'])
            holding_days = (current_date - entry_date).days

            # 1. 止损检查
            if profit_pct <= -self.config.stop_loss_pct * 100:
                trade.close_position(str(row['date']), current_price)
                logger.debug(f"{code} {row['date']}: 止损平仓，盈亏{profit_pct:.2f}%")
                self._close_trade(trade, current_price)
                continue

            # 2. 止盈检查
            if profit_pct >= self.config.take_profit_pct * 100:
                trade.close_position(str(row['date']), current_price)
                logger.debug(f"{code} {row['date']}: 止盈平仓，盈亏{profit_pct:.2f}%")
                self._close_trade(trade, current_price)
                continue

            # 3. 最大持有天数检查
            if holding_days >= self.config.max_holding_days:
                trade.close_position(str(row['date']), current_price)
                logger.debug(f"{code} {row['date']}: 超过最大持有天数平仓，盈亏{profit_pct:.2f}%")
                self._close_trade(trade, current_price)
                continue

            # 4. 卖出信号检查
            sell_signal = row.get('buy_signal', '')
            if sell_signal in ['卖出', '强烈卖出']:
                trade.close_position(str(row['date']), current_price)
                logger.debug(f"{code} {row['date']}: 卖出信号平仓，盈亏{profit_pct:.2f}%")
                self._close_trade(trade, current_price)

    def _close_trade(self, trade: Trade, exit_price: float):
        """
        平仓处理

        Args:
            trade: 交易记录
            exit_price: 平仓价格
        """
        trade.holding_days = (pd.to_datetime(trade.exit_date) -
                            pd.to_datetime(trade.entry_date)).days

        # 更新持仓
        self.position[trade.code] = self.position.get(trade.code, 0) - trade.quantity

        # 增加资金（含手续费）
        proceeds = exit_price * trade.quantity * (1 - self.config.commission_rate)
        self.capital += proceeds

    def _close_all_positions(self, df: pd.DataFrame, code: str):
        """
        平掉所有剩余持仓

        Args:
            df: 股票数据
            code: 股票代码
        """
        if code not in self.position or self.position[code] == 0:
            return

        last_price = df['close'].iloc[-1]
        last_date = df['date'].iloc[-1]

        open_trades = [t for t in self.trades
                      if t.code == code and t.status == TradeStatus.OPEN]

        for trade in open_trades:
            trade.close_position(str(last_date), last_price)
            logger.debug(f"{code} {last_date}: 强制平仓，盈亏{trade.profit_loss_pct:.2f}%")
            self._close_trade(trade, last_price)

    def _update_equity(self, row: pd.Series, code: str):
        """
        更新净值曲线

        Args:
            row: 当前交易日数据
            code: 股票代码
        """
        total_equity = self.capital

        # 计算持仓市值
        if code in self.position and self.position[code] > 0:
            open_trades = [t for t in self.trades
                          if t.code == code and t.status == TradeStatus.OPEN]
            for trade in open_trades:
                market_value = row['close'] * trade.quantity
                total_equity += market_value

        self.equity_curve.append(total_equity)

    def _create_result(self, code: str) -> BacktestResult:
        """创建回测结果"""
        result = BacktestResult(config=self.config)
        result.trades = self.trades
        result.equity_curve = self.equity_curve
        result.total_trades = len(self.trades)

        return result

    def _calculate_performance_metrics(self, result: BacktestResult) -> BacktestResult:
        """
        计算绩效指标

        Args:
            result: 回测结果

        Returns:
            更新后的回测结果
        """
        if not result.trades:
            return result

        # 1. 统计盈亏交易
        closed_trades = [t for t in result.trades if t.status == TradeStatus.CLOSED]

        result.winning_trades = len([t for t in closed_trades if t.profit_loss > 0])
        result.losing_trades = len([t for t in closed_trades if t.profit_loss < 0])

        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades * 100

        # 2. 计算总收益率
        final_equity = result.equity_curve[-1] if result.equity_curve else self.config.initial_capital
        result.total_return = (final_equity / self.config.initial_capital - 1) * 100

        # 3. 计算年化收益率（假设252个交易日）
        if len(result.equity_curve) > 1:
            trading_days = len(result.equity_curve)
            result.annual_return = ((final_equity / self.config.initial_capital) **
                                   (252 / trading_days) - 1) * 100

        # 4. 计算最大回撤
        if result.equity_curve:
            equity_series = pd.Series(result.equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            result.max_drawdown = drawdown.min()

        # 5. 计算夏普比率
        if len(result.equity_curve) > 2:
            equity_returns = pd.Series(result.equity_curve).pct_change().dropna()
            if equity_returns.std() > 0:
                # 假设无风险利率为3%年化
                daily_rf = 0.03 / 252
                result.sharpe_ratio = (equity_returns.mean() - daily_rf) / equity_returns.std() * np.sqrt(252)

        # 6. 计算盈亏统计
        profits = [t.profit_loss for t in closed_trades if t.profit_loss > 0]
        losses = [t.profit_loss for t in closed_trades if t.profit_loss < 0]

        result.total_profit = sum(profits) if profits else 0
        result.total_loss = sum(losses) if losses else 0

        if result.total_loss != 0:
            result.profit_factor = abs(result.total_profit / result.total_loss)

        result.avg_profit = np.mean(profits) if profits else 0
        result.avg_loss = np.mean(losses) if losses else 0

        # 7. 平均持有天数
        holding_days = [t.holding_days for t in closed_trades]
        result.avg_holding_days = np.mean(holding_days) if holding_days else 0

        return result
