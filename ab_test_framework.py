# -*- coding: utf-8 -*-
"""
===================================
A/B测试框架
===================================

功能：
1. 对比原始规则策略 vs ML增强策略
2. 统计显著性检验
3. 效果对比报告
4. 可视化对比结果

设计原则：
- 模块化设计
- 完整的统计分析
- 易于扩展
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """策略类型枚举"""
    RULE_BASED = "rule_based"      # 基于规则的策略（原始）
    ML_ENHANCED = "ml_enhanced"    # ML增强策略


class MetricType(Enum):
    """指标类型枚举"""
    RETURN = "return"              # 收益率
    SHARPE = "sharpe"              # 夏普比率
    WIN_RATE = "win_rate"          # 胜率
    MAX_DRAWDOWN = "max_drawdown"  # 最大回撤
    TRADE_COUNT = "trade_count"    # 交易次数


@dataclass
class ABTestConfig:
    """A/B测试配置"""
    # 测试名称
    test_name: str = "AB_Test"

    # 置信度水平
    confidence_level: float = 0.95

    # 最小样本量
    min_sample_size: int = 30

    # 是否使用时间序列分割
    use_time_series_split: bool = True

    # 测试周期
    test_period_days: int = 90


@dataclass
class StrategyPerformance:
    """策略表现"""
    strategy_type: StrategyType

    # 收益指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0

    # 交易指标
    trade_count: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_loss_ratio: float = 0.0

    # 风险指标
    max_drawdown: float = 0.0
    volatility: float = 0.0

    # 详细数据
    daily_returns: pd.Series = field(default_factory=pd.Series)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_type': self.strategy_type.value,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'trade_count': self.trade_count,
            'win_rate': self.win_rate,
            'avg_profit': self.avg_profit,
            'avg_loss': self.avg_loss,
            'profit_loss_ratio': self.profit_loss_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
        }


@dataclass
class ABTestResult:
    """A/B测试结果"""
    test_name: str
    config: ABTestConfig

    # 策略表现
    control_performance: StrategyPerformance
    treatment_performance: StrategyPerformance

    # 统计检验结果
    lift: float = 0.0                          # 提升百分比
    p_value: float = 1.0                       # p值
    is_significant: bool = False               # 是否显著
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # 置信区间

    # 各指标对比
    metric_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 推荐结论
    recommendation: str = ""
    recommendation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'lift': self.lift,
            'lift_percentage': f"{self.lift * 100:.2f}%",
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'confidence_interval': self.confidence_interval,
            'control_performance': self.control_performance.to_dict(),
            'treatment_performance': self.treatment_performance.to_dict(),
            'metric_comparisons': self.metric_comparisons,
            'recommendation': self.recommendation,
            'recommendation_reason': self.recommendation_reason,
        }


class ABTestFramework:
    """
    A/B测试框架

    功能：
    1. 对比两种策略的表现
    2. 统计显著性检验
    3. 生成对比报告
    """

    def __init__(self, config: Optional[ABTestConfig] = None):
        """
        初始化A/B测试框架

        Args:
            config: 测试配置
        """
        self.config = config or ABTestConfig()

    def run_test(
        self,
        control_signals: pd.Series,
        treatment_signals: pd.Series,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> ABTestResult:
        """
        运行A/B测试

        Args:
            control_signals: 对照组信号（规则策略）
            treatment_signals: 实验组信号（ML策略）
            returns: 收益率序列
            benchmark_returns: 基准收益率（可选）

        Returns:
            A/B测试结果
        """
        logger.info(f"开始A/B测试: {self.config.test_name}")

        # 对齐数据
        aligned_data = pd.DataFrame({
            'control_signal': control_signals,
            'treatment_signal': treatment_signals,
            'return': returns
        }).dropna()

        if len(aligned_data) < self.config.min_sample_size:
            raise ValueError(f"样本量不足，至少需要 {self.config.min_sample_size} 个样本")

        logger.info(f"测试样本量: {len(aligned_data)}")

        # 计算策略表现
        control_perf = self._calculate_performance(
            aligned_data['control_signal'],
            aligned_data['return'],
            benchmark_returns,
            StrategyType.RULE_BASED
        )

        treatment_perf = self._calculate_performance(
            aligned_data['treatment_signal'],
            aligned_data['return'],
            benchmark_returns,
            StrategyType.ML_ENHANCED
        )

        # 统计检验
        lift, p_value, is_significant, ci = self._statistical_test(
            control_perf.daily_returns,
            treatment_perf.daily_returns
        )

        # 指标对比
        metric_comparisons = self._compare_metrics(control_perf, treatment_perf)

        # 生成推荐结论
        recommendation, reason = self._generate_recommendation(
            control_perf, treatment_perf, is_significant, lift
        )

        result = ABTestResult(
            test_name=self.config.test_name,
            config=self.config,
            control_performance=control_perf,
            treatment_performance=treatment_perf,
            lift=lift,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=ci,
            metric_comparisons=metric_comparisons,
            recommendation=recommendation,
            recommendation_reason=reason
        )

        self._log_results(result)

        return result

    def _calculate_performance(
        self,
        signals: pd.Series,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        strategy_type: StrategyType
    ) -> StrategyPerformance:
        """
        计算策略表现

        Args:
            signals: 交易信号（1=买入，0=持有，-1=卖出）
            returns: 收益率序列
            benchmark_returns: 基准收益率
            strategy_type: 策略类型

        Returns:
            策略表现
        """
        # 计算策略收益（只在买入时获得收益）
        strategy_returns = signals * returns
        strategy_returns = strategy_returns.fillna(0)

        # 总收益率
        total_return = (1 + strategy_returns).prod() - 1

        # 年化收益率（假设252个交易日）
        n_days = len(strategy_returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1

        # 夏普比率
        if benchmark_returns is not None:
            excess_returns = strategy_returns - benchmark_returns
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        else:
            sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()

        # 最大回撤
        cum_returns = (1 + strategy_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 波动率
        volatility = strategy_returns.std() * np.sqrt(252)

        # 交易统计
        trade_signals = signals[signals != 0]
        trade_count = len(trade_signals)

        if trade_count > 0:
            trade_returns = returns[signals != 0]
            wins = trade_returns > 0
            win_rate = wins.sum() / len(wins)

            avg_profit = trade_returns[wins].mean() if wins.sum() > 0 else 0
            avg_loss = trade_returns[~wins].mean() if (~wins).sum() > 0 else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_loss_ratio = 0

        return StrategyPerformance(
            strategy_type=strategy_type,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            trade_count=trade_count,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_loss_ratio=profit_loss_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            daily_returns=strategy_returns
        )

    def _statistical_test(
        self,
        control_returns: pd.Series,
        treatment_returns: pd.Series
    ) -> Tuple[float, float, bool, Tuple[float, float]]:
        """
        统计检验（t检验）

        Args:
            control_returns: 对照组收益
            treatment_returns: 实验组收益

        Returns:
            (提升, p值, 是否显著, 置信区间)
        """
        # 计算提升
        control_mean = control_returns.mean()
        treatment_mean = treatment_returns.mean()
        lift = (treatment_mean - control_mean) / abs(control_mean) if control_mean != 0 else 0

        # 配对t检验
        t_stat, p_value = stats.ttest_rel(treatment_returns, control_returns)

        # 置信区间
        alpha = 1 - self.config.confidence_level
        diff = treatment_returns - control_returns
        se = diff.std() / np.sqrt(len(diff))
        t_critical = stats.t.ppf(1 - alpha / 2, len(diff) - 1)
        ci_lower = lift - t_critical * se / abs(control_mean) if control_mean != 0 else 0
        ci_upper = lift + t_critical * se / abs(control_mean) if control_mean != 0 else 0

        is_significant = p_value < (1 - self.config.confidence_level)

        return lift, p_value, is_significant, (ci_lower, ci_upper)

    def _compare_metrics(
        self,
        control: StrategyPerformance,
        treatment: StrategyPerformance
    ) -> Dict[str, Dict[str, Any]]:
        """
        对比各项指标

        Args:
            control: 对照组表现
            treatment: 实验组表现

        Returns:
            指标对比字典
        """
        metrics = {
            'total_return': {
                'control': control.total_return,
                'treatment': treatment.total_return,
                'lift': (treatment.total_return - control.total_return) / abs(control.total_return) if control.total_return != 0 else 0,
                'better': treatment.total_return > control.total_return
            },
            'sharpe_ratio': {
                'control': control.sharpe_ratio,
                'treatment': treatment.sharpe_ratio,
                'lift': (treatment.sharpe_ratio - control.sharpe_ratio) / abs(control.sharpe_ratio) if control.sharpe_ratio != 0 else 0,
                'better': treatment.sharpe_ratio > control.sharpe_ratio
            },
            'win_rate': {
                'control': control.win_rate,
                'treatment': treatment.win_rate,
                'lift': (treatment.win_rate - control.win_rate) / abs(control.win_rate) if control.win_rate != 0 else 0,
                'better': treatment.win_rate > control.win_rate
            },
            'max_drawdown': {
                'control': control.max_drawdown,
                'treatment': treatment.max_drawdown,
                'lift': (treatment.max_drawdown - control.max_drawdown) / abs(control.max_drawdown) if control.max_drawdown != 0 else 0,
                'better': treatment.max_drawdown > control.max_drawdown  # 回撤越小越好，所以这里是反向的
            },
        }

        return metrics

    def _generate_recommendation(
        self,
        control: StrategyPerformance,
        treatment: StrategyPerformance,
        is_significant: bool,
        lift: float
    ) -> Tuple[str, str]:
        """
        生成推荐结论

        Args:
            control: 对照组表现
            treatment: 实验组表现
            is_significant: 是否显著
            lift: 提升幅度

        Returns:
            (推荐结论, 推荐理由)
        """
        if lift > 0 and is_significant:
            recommendation = "采用ML增强策略"
            reason = f"ML策略相比规则策略有 {lift * 100:.2f}% 的显著提升（p < {1 - self.config.confidence_level}）"
        elif lift > 0 and not is_significant:
            recommendation = "可以考虑ML增强策略"
            reason = f"ML策略相比规则策略有 {lift * 100:.2f}% 的提升，但统计上不显著"
        elif lift <= 0 and is_significant:
            recommendation = "保持规则策略"
            reason = f"ML策略相比规则策略有 {lift * 100:.2f}% 的显著下降"
        else:
            recommendation = "两种策略效果相当"
            reason = f"ML策略相比规则策略有 {lift * 100:.2f}% 的变化，但不显著"

        return recommendation, reason

    def _log_results(self, result: ABTestResult) -> None:
        """记录测试结果"""
        logger.info("=" * 60)
        logger.info(f"A/B测试结果: {result.test_name}")
        logger.info("=" * 60)
        logger.info(f"对照组（规则策略）总收益率: {result.control_performance.total_return:.4f}")
        logger.info(f"实验组（ML策略）总收益率: {result.treatment_performance.total_return:.4f}")
        logger.info(f"提升: {result.lift * 100:.2f}%")
        logger.info(f"P值: {result.p_value:.4f}")
        logger.info(f"是否显著: {result.is_significant}")
        logger.info(f"置信区间: [{result.confidence_interval[0] * 100:.2f}%, {result.confidence_interval[1] * 100:.2f}%]")
        logger.info(f"\n推荐结论: {result.recommendation}")
        logger.info(f"推荐理由: {result.recommendation_reason}")
        logger.info("=" * 60)


def run_ab_test_from_signals(
    rule_signals: pd.Series,
    ml_signals: pd.Series,
    returns: pd.Series,
    test_name: str = "AB_Test"
) -> ABTestResult:
    """
    从信号运行A/B测试的便捷函数

    Args:
        rule_signals: 规则策略信号
        ml_signals: ML策略信号
        returns: 收益率序列
        test_name: 测试名称

    Returns:
        A/B测试结果
    """
    config = ABTestConfig(test_name=test_name)
    framework = ABTestFramework(config)
    result = framework.run_test(rule_signals, ml_signals, returns)

    return result


if __name__ == "__main__":
    # 测试A/B框架
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建模拟数据
    np.random.seed(42)
    n_days = 252

    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, n_days), index=dates)

    # 模拟信号
    rule_signals = pd.Series(np.random.choice([-1, 0, 1], n_days, p=[0.1, 0.7, 0.2]), index=dates)
    ml_signals = pd.Series(np.random.choice([-1, 0, 1], n_days, p=[0.05, 0.65, 0.3]), index=dates)

    # 运行A/B测试
    result = run_ab_test_from_signals(
        rule_signals,
        ml_signals,
        returns,
        "规则策略 vs ML策略"
    )

    print("\n测试结果:")
    print(f"总收益率对比:")
    print(f"  规则策略: {result.control_performance.total_return:.4f}")
    print(f"  ML策略: {result.treatment_performance.total_return:.4f}")
    print(f"  提升: {result.lift * 100:.2f}%")
