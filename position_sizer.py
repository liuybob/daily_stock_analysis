# -*- coding: utf-8 -*-
"""
===================================
动态仓位配置系统
===================================

功能：
1. 基于风险的仓位大小计算
2. Kelly公式仓位配置
3. 风险平价方法
4. 固定比例法

设计原则：
- 多种仓位配置策略
- 风险感知
- 易于集成
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """仓位配置方法枚举"""
    FIXED_RATIO = "fixed_ratio"          # 固定比例
    KELLY = "kelly"                      # Kelly公式
    RISK_PARITY = "risk_parity"          # 风险平价
    VOLATILITY_TARGET = "volatility_target"  # 目标波动率
    ATR_BASED = "atr_based"              # 基于ATR


@dataclass
class PositionSize:
    """仓位大小"""
    symbol: str              # 股票代码
    shares: int              # 股数
    weight: float            # 权重
    dollar_amount: float     # 金额
    risk_amount: float       # 风险金额


@dataclass
class SizingConfig:
    """仓位配置配置"""
    # 方法
    method: SizingMethod = SizingMethod.FIXED_RATIO

    # 资金配置
    total_capital: float = 100000.0      # 总资金
    max_position_pct: float = 0.3        # 单个仓位最大比例
    max_total_exposure: float = 1.0      # 最大总敞口

    # 固定比例参数
    fixed_ratio: float = 0.1             # 固定比例（10%）

    # Kelly公式参数
    win_rate: float = 0.55               # 胜率
    avg_win: float = 0.05                # 平均盈利
    avg_loss: float = 0.03               # 平均亏损
    kelly_fraction: float = 0.5          # Kelly分数（半Kelly）

    # 风险平价参数
    risk_tolerance: float = 0.02         # 风险容忍度

    # 目标波动率参数
    target_volatility: float = 0.15      # 目标年化波动率

    # ATR参数
    atr_multiplier: float = 2.0          # ATR倍数
    risk_per_trade: float = 0.02         # 单笔交易风险


# 别名，为了兼容性
PositionSizerConfig = SizingConfig


class PositionSizer:
    """
    仓位配置器

    功能：
    1. 计算仓位大小
    2. 风险控制
    3. 多种配置方法
    """

    def __init__(self, config: Optional[SizingConfig] = None):
        """
        初始化仓位配置器

        Args:
            config: 配置
        """
        self.config = config or SizingConfig()

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        confidence: Optional[float] = None,
        volatility: Optional[float] = None,
        atr: Optional[float] = None
    ) -> PositionSize:
        """
        计算仓位大小

        Args:
            symbol: 股票代码
            price: 当前价格
            confidence: 置信度（0-1）
            volatility: 波动率
            atr: ATR值

        Returns:
            仓位大小
        """
        if self.config.method == SizingMethod.FIXED_RATIO:
            return self._fixed_ratio_sizing(symbol, price)
        elif self.config.method == SizingMethod.KELLY:
            return self._kelly_sizing(symbol, price, confidence)
        elif self.config.method == SizingMethod.RISK_PARITY:
            return self._risk_parity_sizing(symbol, price, volatility)
        elif self.config.method == SizingMethod.VOLATILITY_TARGET:
            return self._volatility_target_sizing(symbol, price, volatility)
        elif self.config.method == SizingMethod.ATR_BASED:
            return self._atr_based_sizing(symbol, price, atr)
        else:
            raise ValueError(f"不支持的仓位配置方法: {self.config.method}")

    def _fixed_ratio_sizing(self, symbol: str, price: float) -> PositionSize:
        """
        固定比例法

        Args:
            symbol: 股票代码
            price: 价格

        Returns:
            仓位大小
        """
        dollar_amount = self.config.total_capital * self.config.fixed_ratio
        shares = int(dollar_amount / price)

        return PositionSize(
            symbol=symbol,
            shares=shares,
            weight=self.config.fixed_ratio,
            dollar_amount=dollar_amount,
            risk_amount=dollar_amount * 0.05  # 假设5%风险
        )

    def _kelly_sizing(
        self,
        symbol: str,
        price: float,
        confidence: Optional[float] = None
    ) -> PositionSize:
        """
        Kelly公式法

        f = (p*b - q) / b
        其中：
        p = 胜率
        q = 1-p (败率)
        b = 盈亏比

        Args:
            symbol: 股票代码
            price: 价格
            confidence: 置信度（调整胜率）

        Returns:
            仓位大小
        """
        # 使用置信度调整胜率
        win_rate = self.config.win_rate
        if confidence is not None:
            win_rate = 0.5 + (win_rate - 0.5) * confidence

        lose_rate = 1 - win_rate
        win_loss_ratio = self.config.avg_win / self.config.avg_loss

        # Kelly比例
        kelly_pct = (win_rate * win_loss_ratio - lose_rate) / win_loss_ratio

        # 使用半Kelly或部分Kelly
        kelly_pct *= self.config.kelly_fraction

        # 限制在最大仓位范围内
        kelly_pct = min(kelly_pct, self.config.max_position_pct)
        kelly_pct = max(kelly_pct, 0)

        dollar_amount = self.config.total_capital * kelly_pct
        shares = int(dollar_amount / price)

        return PositionSize(
            symbol=symbol,
            shares=shares,
            weight=kelly_pct,
            dollar_amount=dollar_amount,
            risk_amount=dollar_amount * self.config.risk_tolerance
        )

    def _risk_parity_sizing(
        self,
        symbol: str,
        price: float,
        volatility: Optional[float] = None
    ) -> PositionSize:
        """
        风险平价法

        权重 ∝ 1/波动率

        Args:
            symbol: 股票代码
            price: 价格
            volatility: 波动率

        Returns:
            仓位大小
        """
        if volatility is None:
            volatility = 0.2  # 默认20%年化波动率

        # 风险平价权重
        weight = min(self.config.risk_tolerance / volatility, self.config.max_position_pct)

        dollar_amount = self.config.total_capital * weight
        shares = int(dollar_amount / price)

        return PositionSize(
            symbol=symbol,
            shares=shares,
            weight=weight,
            dollar_amount=dollar_amount,
            risk_amount=dollar_amount * volatility
        )

    def _volatility_target_sizing(
        self,
        symbol: str,
        price: float,
        volatility: Optional[float] = None
    ) -> PositionSize:
        """
        目标波动率法

        权重 = 目标波动率 / 资产波动率

        Args:
            symbol: 股票代码
            price: 价格
            volatility: 波动率

        Returns:
            仓位大小
        """
        if volatility is None:
            volatility = 0.2  # 默认20%年化波动率

        # 目标波动率权重
        weight = min(self.config.target_volatility / volatility, self.config.max_position_pct)

        dollar_amount = self.config.total_capital * weight
        shares = int(dollar_amount / price)

        return PositionSize(
            symbol=symbol,
            shares=shares,
            weight=weight,
            dollar_amount=dollar_amount,
            risk_amount=dollar_amount * (self.config.target_volatility / self.config.total_capital)
        )

    def _atr_based_sizing(
        self,
        symbol: str,
        price: float,
        atr: Optional[float] = None
    ) -> PositionSize:
        """
        基于ATR的仓位配置

        股数 = (总资金 * 风险比例) / (ATR * 倍数)

        Args:
            symbol: 股票代码
            price: 价格
            atr: ATR值

        Returns:
            仓位大小
        """
        if atr is None:
            atr = price * 0.02  # 默认ATR为价格的2%

        # 计算风险金额
        risk_amount = self.config.total_capital * self.config.risk_per_trade

        # 计算止损距离
        stop_distance = atr * self.config.atr_multiplier

        # 计算股数
        shares = int(risk_amount / stop_distance)

        # 计算权重
        dollar_amount = shares * price
        weight = min(dollar_amount / self.config.total_capital, self.config.max_position_pct)

        return PositionSize(
            symbol=symbol,
            shares=shares,
            weight=weight,
            dollar_amount=dollar_amount,
            risk_amount=risk_amount
        )

    def calculate_portfolio_sizes(
        self,
        symbols: List[str],
        prices: Dict[str, float],
        confidences: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
        atrs: Optional[Dict[str, float]] = None
    ) -> List[PositionSize]:
        """
        计算投资组合中所有股票的仓位大小

        Args:
            symbols: 股票代码列表
            prices: 价格字典
            confidences: 置信度字典
            volatilities: 波动率字典
            atrs: ATR字典

        Returns:
            仓位大小列表
        """
        positions = []

        for symbol in symbols:
            if symbol not in prices:
                logger.warning(f"股票 {symbol} 没有价格信息，跳过")
                continue

            confidence = confidences.get(symbol) if confidences else None
            volatility = volatilities.get(symbol) if volatilities else None
            atr = atrs.get(symbol) if atrs else None

            try:
                position = self.calculate_position_size(
                    symbol, prices[symbol], confidence, volatility, atr
                )
                positions.append(position)
            except Exception as e:
                logger.error(f"计算 {symbol} 仓位失败: {e}")

        return positions


def calculate_position_size_simple(
    capital: float,
    price: float,
    risk_per_trade: float = 0.02,
    stop_loss_pct: float = 0.05
) -> int:
    """
    简化的仓位计算

    股数 = (总资金 * 风险比例) / (价格 * 止损比例)

    Args:
        capital: 总资金
        price: 价格
        risk_per_trade: 单笔交易风险比例
        stop_loss_pct: 止损比例

    Returns:
        股数
    """
    risk_amount = capital * risk_per_trade
    risk_per_share = price * stop_loss_pct
    shares = int(risk_amount / risk_per_share)

    return shares


if __name__ == "__main__":
    # 测试仓位配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 固定比例法
    config = SizingConfig(
        method=SizingMethod.FIXED_RATIO,
        total_capital=100000,
        fixed_ratio=0.1
    )

    sizer = PositionSizer(config)
    position = sizer.calculate_position_size("AAPL", 150)

    print(f"\n固定比例法:")
    print(f"  股数: {position.shares}")
    print(f"  权重: {position.weight * 100:.2f}%")
    print(f"  金额: ${position.dollar_amount:.2f}")

    # Kelly公式
    config.method = SizingMethod.KELLY
    config.win_rate = 0.55
    config.avg_win = 0.05
    config.avg_loss = 0.03

    sizer = PositionSizer(config)
    position = sizer.calculate_position_size("AAPL", 150, confidence=0.8)

    print(f"\nKelly公式:")
    print(f"  股数: {position.shares}")
    print(f"  权重: {position.weight * 100:.2f}%")
    print(f"  金额: ${position.dollar_amount:.2f}")

    # 基于ATR
    config.method = SizingMethod.ATR_BASED
    sizer = PositionSizer(config)
    position = sizer.calculate_position_size("AAPL", 150, atr=3)

    print(f"\n基于ATR:")
    print(f"  股数: {position.shares}")
    print(f"  权重: {position.weight * 100:.2f}%")
    print(f"  金额: ${position.dollar_amount:.2f}")
