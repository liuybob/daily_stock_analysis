# -*- coding: utf-8 -*-
"""
===================================
多维度风险评估系统
===================================

功能：
1. 多维度风险评分
2. 黑天鹅事件检测
3. 风险警示系统

风险评估维度：
- 技术面风险
- 市场风险
- 流动性风险
- 波动性风险
- 趋势风险
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "低风险"           # 0-30分
    MEDIUM = "中等风险"      # 31-60分
    HIGH = "高风险"          # 61-80分
    EXTREME = "极高风险"     # 81-100分


class RiskCategory(Enum):
    """风险类别"""
    TECHNICAL = "技术面风险"
    MARKET = "市场风险"
    LIQUIDITY = "流动性风险"
    VOLATILITY = "波动性风险"
    TREND = "趋势风险"
    BLACK_SWAN = "黑天鹅风险"


@dataclass
class RiskConfig:
    """
    风险分析配置

    Attributes:
        max_position_pct: 单个股票最大仓位百分比
        stop_loss_pct: 止损百分比
        max_drawdown_pct: 最大回撤百分比
        volatility_window: 波动率计算窗口
    """
    max_position_pct: float = 0.2
    stop_loss_pct: float = 0.08
    max_drawdown_pct: float = 0.15
    volatility_window: int = 20


@dataclass
class RiskFactor:
    """单个风险因素"""
    category: RiskCategory
    name: str
    description: str
    score: int  # 0-100
    weight: float  # 权重 0-1


@dataclass
class RiskAssessmentResult:
    """风险评估结果"""
    code: str
    total_risk_score: int = 0  # 0-100
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # 分类风险评分
    technical_risk: int = 0
    market_risk: int = 0
    liquidity_risk: int = 0
    volatility_risk: int = 0
    trend_risk: int = 0

    # 风险因素列表
    risk_factors: List[RiskFactor] = field(default_factory=list)

    # 黑天鹅事件
    black_swans: List[str] = field(default_factory=list)

    # 风险建议
    risk_warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'total_risk_score': self.total_risk_score,
            'risk_level': self.risk_level.value,
            'technical_risk': self.technical_risk,
            'market_risk': self.market_risk,
            'liquidity_risk': self.liquidity_risk,
            'volatility_risk': self.volatility_risk,
            'trend_risk': self.trend_risk,
            'risk_factors': [
                {
                    'category': rf.category.value,
                    'name': rf.name,
                    'description': rf.description,
                    'score': rf.score,
                    'weight': rf.weight
                }
                for rf in self.risk_factors
            ],
            'black_swans': self.black_swans,
            'risk_warnings': self.risk_warnings,
            'suggestions': self.suggestions
        }


class RiskAnalyzer:
    """
    多维度风险分析器

    风险评估维度：
    1. 技术面风险 (30%) - 趋势、均线、形态
    2. 市场风险 (20%) - 量价关系、买卖力度
    3. 流动性风险 (20%) - 换手率、成交量
    4. 波动性风险 (15%) - ATR、波动率
    5. 趋势风险 (15%) - 趋势强度、持续性
    """

    # 风险阈值配置
    LIQUIDITY_RISK_THRESHOLD = 15.0  # 低换手率阈值
    HIGH_TURNOVER_THRESHOLD = 20.0   # 高换手率阈值
    VOLATILITY_THRESHOLD = 0.05      # 5%日波动率阈值

    def __init__(self, config: Optional[RiskConfig] = None):
        """
        初始化风险分析器

        Args:
            config: 风险配置
        """
        self.config = config or RiskConfig()

    def analyze_risk(
        self,
        symbol: str,
        data: pd.DataFrame,
        lookback: int = 20
    ) -> Dict[str, Any]:
        """
        分析股票风险（便捷方法）

        Args:
            symbol: 股票代码
            data: 价格数据
            lookback: 回看周期

        Returns:
            风险分析结果字典
        """
        result = self.assess_risk(data, symbol)
        return result.to_dict()

    def assess_risk(self, df: pd.DataFrame, code: str,
                    trend_result: Optional[Any] = None) -> RiskAssessmentResult:
        """
        综合风险评估

        Args:
            df: 股票数据
            code: 股票代码
            trend_result: 趋势分析结果（可选）

        Returns:
            RiskAssessmentResult 风险评估结果
        """
        result = RiskAssessmentResult(code=code)

        if df is None or df.empty or len(df) < 20:
            logger.warning(f"{code}: 数据不足，无法进行风险评估")
            result.total_risk_score = 50
            result.risk_level = RiskLevel.MEDIUM
            result.risk_warnings.append("数据不足，无法准确评估风险")
            return result

        try:
            # 1. 技术面风险评估 (30%)
            result.technical_risk = self._assess_technical_risk(df, trend_result)

            # 2. 市场风险评估 (20%)
            result.market_risk = self._assess_market_risk(df)

            # 3. 流动性风险评估 (20%)
            result.liquidity_risk = self._assess_liquidity_risk(df)

            # 4. 波动性风险评估 (15%)
            result.volatility_risk = self._assess_volatility_risk(df)

            # 5. 趋势风险评估 (15%)
            result.trend_risk = self._assess_trend_risk(df, trend_result)

            # 6. 黑天鹅事件检测
            result.black_swans = self._detect_black_swans(df)

            # 7. 计算总体风险评分（加权平均）
            result.total_risk_score = self._calculate_total_risk_score(result)

            # 8. 确定风险等级
            result.risk_level = self._determine_risk_level(result.total_risk_score)

            # 9. 生成风险建议
            self._generate_risk_suggestions(result)

        except Exception as e:
            logger.error(f"{code}: 风险评估出错 - {str(e)}")
            result.total_risk_score = 50
            result.risk_level = RiskLevel.MEDIUM
            result.risk_warnings.append(f"风险评估异常: {str(e)}")

        return result

    def _assess_technical_risk(self, df: pd.DataFrame,
                               trend_result: Optional[Any]) -> int:
        """
        技术面风险评估 (0-100)

        评估因素：
        - 均线排列状态
        - 趋势强度
        - 支撑压力位
        """
        score = 0  # 风险分数，越高越危险

        latest = df.iloc[-1]

        # 1. 均线排列风险 (40分)
        ma5 = latest.get('MA5', latest.get('ma5', 0))
        ma10 = latest.get('MA10', latest.get('ma10', 0))
        ma20 = latest.get('MA20', latest.get('ma20', 0))

        if ma5 < ma10 < ma20:
            score += 40  # 空头排列，高风险
            self.risk_factors_append(RiskCategory.TECHNICAL,
                                   "空头排列", "MA5<MA10<MA20", 40, 0.4)
        elif ma5 < ma10:
            score += 25  # 弱势
            self.risk_factors_append(RiskCategory.TECHNICAL,
                                   "弱势排列", "MA5<MA10", 25, 0.25)
        else:
            score += 5  # 多头，低风险

        # 2. 趋势强度风险 (30分)
        if trend_result and hasattr(trend_result, 'trend_strength'):
            trend_strength = trend_result.trend_strength
            if trend_strength < 30:
                score += 30  # 趋势很弱
            elif trend_strength < 50:
                score += 20  # 趋势较弱
            else:
                score += 5  # 趋势强

        # 3. 均线乖离风险 (30分)
        if ma5 > 0:
            price = latest['close']
            bias = abs((price - ma5) / ma5 * 100)
            if bias > 10:
                score += 30  # 乖离率过大
            elif bias > 5:
                score += 20  # 乖离率较大
            else:
                score += 5  # 乖离率正常

        return min(score, 100)

    def _assess_market_risk(self, df: pd.DataFrame) -> int:
        """
        市场风险评估 (0-100)

        评估因素：
        - 量价关系
        - 换手率
        - 涨跌幅
        """
        score = 0

        if len(df) < 5:
            return 50

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 1. 放量下跌风险 (50分)
        price_change = (latest['close'] - prev['close']) / prev['close']
        volume_change = (latest['volume'] - df['volume'].iloc[-6:-1].mean()) / df['volume'].iloc[-6:-1].mean()

        if price_change < -0.03 and volume_change > 0.5:
            score += 50  # 放量下跌，高风险
            self.risk_factors_append(RiskCategory.MARKET,
                                   "放量下跌", "跌幅>3%且放量", 50, 0.5)
        elif price_change < -0.02:
            score += 30  # 下跌
        elif price_change > 0.05 and volume_change > 0.5:
            score += 20  # 放量上涨，中等风险
        else:
            score += 10  # 正常

        # 2. 涨跌停风险 (50分)
        if price_change > 0.095:  # 接近涨停
            score += 30
        elif price_change < -0.095:  # 接近跌停
            score += 50  # 跌停风险

        return min(score, 100)

    def _assess_liquidity_risk(self, df: pd.DataFrame) -> int:
        """
        流动性风险评估 (0-100)

        评估因素：
        - 换手率
        - 成交量
        """
        score = 0

        if len(df) < 5:
            return 50

        latest = df.iloc[-1]

        # 1. 换手率风险 (70分)
        # 注意：这里需要从外部获取换手率数据，暂时用成交量替代
        avg_volume = df['volume'].iloc[-20:].mean()
        if avg_volume > 0:
            volume_ratio = latest['volume'] / avg_volume

            if volume_ratio < 0.3:
                score += 50  # 缩量严重，流动性不足
                self.risk_factors_append(RiskCategory.LIQUIDITY,
                                       "缩量严重", "成交量<30%均量", 50, 0.5)
            elif volume_ratio > 3.0:
                score += 30  # 放量过度
            else:
                score += 10  # 正常

        # 2. 成交量趋势风险 (30分)
        recent_volumes = df['volume'].iloc[-5:]
        if recent_volumes.is_monotonic_decreasing:
            score += 30  # 成交量持续萎缩
        else:
            score += 10

        return min(score, 100)

    def _assess_volatility_risk(self, df: pd.DataFrame) -> int:
        """
        波动性风险评估 (0-100)

        评估因素：
        - ATR (平均真实波幅)
        - 价格波动率
        """
        score = 0

        if len(df) < 20:
            return 50

        latest = df.iloc[-1]
        price = latest['close']

        # 1. ATR波动率风险 (60分)
        if 'atr' in df.columns:
            atr = df['atr'].iloc[-1]
            atr_ratio = atr / price if price > 0 else 0

            if atr_ratio > 0.08:  # 日波动>8%
                score += 60  # 极高波动
            elif atr_ratio > 0.05:  # 日波动>5%
                score += 40  # 高波动
            elif atr_ratio > 0.03:  # 日波动>3%
                score += 20  # 中等波动
            else:
                score += 10  # 低波动
        else:
            # 如果没有ATR，用简单的波动率计算
            returns = df['close'].pct_change().iloc[-20:]
            volatility = returns.std()
            if volatility > 0.08:
                score += 60
            elif volatility > 0.05:
                score += 40
            else:
                score += 20

        # 2. 连续涨跌风险 (40分)
        # 检查最近5天的连续涨跌
        recent_changes = df['close'].diff().iloc[-5:]
        if all(recent_changes > 0):
            score += 30  # 连续上涨，回调风险
        elif all(recent_changes < 0):
            score += 40  # 连续下跌，高风险
        else:
            score += 10

        return min(score, 100)

    def _assess_trend_risk(self, df: pd.DataFrame,
                          trend_result: Optional[Any]) -> int:
        """
        趋势风险评估 (0-100)

        评估因素：
        - 趋势强度
        - 趋势持续性
        - 趋势转折信号
        """
        score = 0

        if len(df) < 20:
            return 50

        # 1. 趋势强度风险 (50分)
        if trend_result and hasattr(trend_result, 'trend_strength'):
            strength = trend_result.trend_strength
            if strength < 30:
                score += 50  # 趋势很弱
            elif strength < 50:
                score += 30  # 趋势较弱
            else:
                score += 10  # 趋势强
        else:
            # 计算趋势强度
            ma5 = df['close'].rolling(5).mean().iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]

            if ma5 > ma20:
                score += 10  # 上升趋势
            elif ma5 < ma20:
                score += 40  # 下降趋势
            else:
                score += 20  # 盘整

        # 2. 趋势转折风险 (50分)
        # 检查是否有趋势转折信号
        if 'rsi' in df.columns and 'macd' in df.columns:
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]

            # RSI背离
            if rsi > 70:
                score += 25  # 超买风险

            # MACD死叉
            if macd < 0:
                score += 25  # MACD空头风险

        return min(score, 100)

    def _detect_black_swans(self, df: pd.DataFrame) -> List[str]:
        """
        黑天鹅事件检测

        检测内容：
        - 连续跌停
        - 暴跌（单日跌幅>10%）
        - 异常停牌
        - 成交量异常
        """
        black_swans = []

        if len(df) < 5:
            return black_swans

        # 1. 检测连续暴跌
        for i in range(len(df) - 3, len(df)):
            if i >= 0:
                change = (df.iloc[i]['close'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close'] if i > 0 else 0
                if change < -0.095:  # 跌停
                    black_swans.append(f"⚠️ 跌停风险：{df.iloc[i]['date'].strftime('%Y-%m-%d')}")

        # 2. 检测异常缩量（流动性危机）
        recent_volumes = df['volume'].iloc[-5:]
        avg_volume = df['volume'].iloc[-20:].mean()
        if recent_volumes.iloc[-1] < avg_volume * 0.2:
            black_swans.append("⚠️ 流动性危机：成交量萎缩至20%以下")

        # 3. 检测断崖式下跌
        if len(df) >= 3:
            recent_high = df['high'].iloc[-5:].max()
            current_price = df['close'].iloc[-1]
            drop_ratio = (recent_high - current_price) / recent_high

            if drop_ratio > 0.20:  # 5天内跌幅超过20%
                black_swans.append(f"⚠️ 断崖式下跌：5日内跌幅{drop_ratio*100:.1f}%")

        # 4. 检测RSI极端值
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if rsi < 20:
                black_swans.append(f"⚠️ RSI极度超卖：{rsi:.1f}")
            elif rsi > 80:
                black_swans.append(f"⚠️ RSI极度超买：{rsi:.1f}")

        return black_swans

    def _calculate_total_risk_score(self, result: RiskAssessmentResult) -> int:
        """
        计算总体风险评分（加权平均）

        权重分配：
        - 技术面风险: 30%
        - 市场风险: 20%
        - 流动性风险: 20%
        - 波动性风险: 15%
        - 趋势风险: 15%
        """
        weights = {
            'technical': 0.30,
            'market': 0.20,
            'liquidity': 0.20,
            'volatility': 0.15,
            'trend': 0.15
        }

        total_score = (
            result.technical_risk * weights['technical'] +
            result.market_risk * weights['market'] +
            result.liquidity_risk * weights['liquidity'] +
            result.volatility_risk * weights['volatility'] +
            result.trend_risk * weights['trend']
        )

        # 如果有黑天鹅事件，增加风险分数
        if result.black_swans:
            black_swans_penalty = len(result.black_swans) * 10
            total_score = min(total_score + black_swans_penalty, 100)

        return int(total_score)

    def _determine_risk_level(self, score: int) -> RiskLevel:
        """确定风险等级"""
        if score <= 30:
            return RiskLevel.LOW
        elif score <= 60:
            return RiskLevel.MEDIUM
        elif score <= 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME

    def _generate_risk_suggestions(self, result: RiskAssessmentResult):
        """生成风险建议"""
        # 根据风险等级生成建议
        if result.risk_level == RiskLevel.LOW:
            result.suggestions.append("✅ 风险较低，可考虑建仓")
            result.suggestions.append("✅ 建议设置止损位")

        elif result.risk_level == RiskLevel.MEDIUM:
            result.suggestions.append("⚡ 风险适中，谨慎参与")
            result.suggestions.append("⚡ 建议小仓位试探")
            result.suggestions.append("⚡ 密切关注风险变化")

        elif result.risk_level == RiskLevel.HIGH:
            result.risk_warnings.append("❌ 风险较高，不建议建仓")
            result.suggestions.append("🛑 观望为主，等待更好时机")
            result.suggestions.append("🛑 如已持有，考虑减仓")

        else:  # EXTREME
            result.risk_warnings.append("🚨 极高风险，严禁介入")
            result.suggestions.append("⛔ 坚决观望，等待风险释放")
            result.suggestions.append("⛔ 如已持有，立即止损")

        # 根据具体风险因素给出建议
        if result.technical_risk > 70:
            result.risk_warnings.append("⚠️ 技术面风险高，趋势不明")

        if result.liquidity_risk > 70:
            result.risk_warnings.append("⚠️ 流动性风险高，注意成交量")

        if result.volatility_risk > 70:
            result.risk_warnings.append("⚠️ 波动性风险高，注意控制仓位")

        if result.black_swans:
            result.risk_warnings.append(f"🚨 检测到{len(result.black_swans)}个黑天鹅风险信号")

    def risk_factors_append(self, category: RiskCategory, name: str,
                           description: str, score: int, weight: float):
        """辅助方法：添加风险因素（如果存在result对象）"""
        # 这个方法将在实际使用时被整合到assess方法中
        pass
