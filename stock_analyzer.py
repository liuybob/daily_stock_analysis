# -*- coding: utf-8 -*-
"""
===================================
趋势交易分析器 - 基于用户交易理念
===================================

交易理念核心原则：
1. 严进策略 - 不追高，追求每笔交易成功率
2. 趋势交易 - MA5>MA10>MA20 多头排列，顺势而为
3. 效率优先 - 关注筹码结构好的股票
4. 买点偏好 - 在 MA5/MA10 附近回踩买入

技术标准：
- 多头排列：MA5 > MA10 > MA20
- 乖离率：(Close - MA5) / MA5 < 5%（不追高）
- 量能形态：缩量回调优先
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TrendStatus(Enum):
    """趋势状态枚举"""
    STRONG_BULL = "强势多头"      # MA5 > MA10 > MA20，且间距扩大
    BULL = "多头排列"             # MA5 > MA10 > MA20
    WEAK_BULL = "弱势多头"        # MA5 > MA10，但 MA10 < MA20
    CONSOLIDATION = "盘整"        # 均线缠绕
    WEAK_BEAR = "弱势空头"        # MA5 < MA10，但 MA10 > MA20
    BEAR = "空头排列"             # MA5 < MA10 < MA20
    STRONG_BEAR = "强势空头"      # MA5 < MA10 < MA20，且间距扩大


class VolumeStatus(Enum):
    """量能状态枚举"""
    HEAVY_VOLUME_UP = "放量上涨"       # 量价齐升
    HEAVY_VOLUME_DOWN = "放量下跌"     # 放量杀跌
    SHRINK_VOLUME_UP = "缩量上涨"      # 无量上涨
    SHRINK_VOLUME_DOWN = "缩量回调"    # 缩量回调（好）
    NORMAL = "量能正常"


class BuySignal(Enum):
    """买入信号枚举"""
    STRONG_BUY = "强烈买入"       # 多条件满足
    BUY = "买入"                  # 基本条件满足
    HOLD = "持有"                 # 已持有可继续
    WAIT = "观望"                 # 等待更好时机
    SELL = "卖出"                 # 趋势转弱
    STRONG_SELL = "强烈卖出"      # 趋势破坏


@dataclass
class TrendAnalysisResult:
    """趋势分析结果"""
    code: str
    
    # 趋势判断
    trend_status: TrendStatus = TrendStatus.CONSOLIDATION
    ma_alignment: str = ""           # 均线排列描述
    trend_strength: float = 0.0      # 趋势强度 0-100
    
    # 均线数据
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    current_price: float = 0.0
    
    # 乖离率（与 MA5 的偏离度）
    bias_ma5: float = 0.0            # (Close - MA5) / MA5 * 100
    bias_ma10: float = 0.0
    bias_ma20: float = 0.0
    
    # 量能分析
    volume_status: VolumeStatus = VolumeStatus.NORMAL
    volume_ratio_5d: float = 0.0     # 当日成交量/5日均量
    volume_trend: str = ""           # 量能趋势描述
    
    # 支撑压力
    support_ma5: bool = False        # MA5 是否构成支撑
    support_ma10: bool = False       # MA10 是否构成支撑
    resistance_levels: List[float] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)

    # 新技术指标
    rsi: Optional[float] = None      # RSI指标值
    macd_signal: Optional[str] = None  # MACD信号：golden_cross/death_cross/bullish/bearish/neutral
    boll_position: Optional[str] = None  # BOLL位置：lower_touch/lower_near/middle/upper_near/upper_touch

    # 风险评估（第二周新增）
    risk_score: int = 0              # 风险评分 0-100
    risk_level: str = "中等风险"      # 风险等级

    # 买入信号
    buy_signal: BuySignal = BuySignal.WAIT
    signal_score: int = 0            # 综合评分 0-120
    signal_reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'trend_status': self.trend_status.value,
            'ma_alignment': self.ma_alignment,
            'trend_strength': self.trend_strength,
            'ma5': self.ma5,
            'ma10': self.ma10,
            'ma20': self.ma20,
            'ma60': self.ma60,
            'current_price': self.current_price,
            'bias_ma5': self.bias_ma5,
            'bias_ma10': self.bias_ma10,
            'bias_ma20': self.bias_ma20,
            'volume_status': self.volume_status.value,
            'volume_ratio_5d': self.volume_ratio_5d,
            'volume_trend': self.volume_trend,
            'support_ma5': self.support_ma5,
            'support_ma10': self.support_ma10,
            'rsi': self.rsi,
            'macd_signal': self.macd_signal,
            'boll_position': self.boll_position,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'buy_signal': self.buy_signal.value,
            'signal_score': self.signal_score,
            'signal_reasons': self.signal_reasons,
            'risk_factors': self.risk_factors,
        }


class StockTrendAnalyzer:
    """
    股票趋势分析器
    
    基于用户交易理念实现：
    1. 趋势判断 - MA5>MA10>MA20 多头排列
    2. 乖离率检测 - 不追高，偏离 MA5 超过 5% 不买
    3. 量能分析 - 偏好缩量回调
    4. 买点识别 - 回踩 MA5/MA10 支撑
    """
    
    # 交易参数配置
    BIAS_THRESHOLD = 5.0        # 乖离率阈值（%），超过此值不买入
    VOLUME_SHRINK_RATIO = 0.7   # 缩量判断阈值（当日量/5日均量）
    VOLUME_HEAVY_RATIO = 1.5    # 放量判断阈值
    MA_SUPPORT_TOLERANCE = 0.02  # MA 支撑判断容忍度（2%）
    
    def __init__(self):
        """初始化分析器"""
        pass
    
    def analyze(self, df: pd.DataFrame, code: str) -> TrendAnalysisResult:
        """
        分析股票趋势
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            code: 股票代码
            
        Returns:
            TrendAnalysisResult 分析结果
        """
        result = TrendAnalysisResult(code=code)
        
        if df is None or df.empty or len(df) < 20:
            logger.warning(f"{code} 数据不足，无法进行趋势分析")
            result.risk_factors.append("数据不足，无法完成分析")
            return result
        
        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 计算均线
        df = self._calculate_mas(df)
        
        # 获取最新数据
        latest = df.iloc[-1]
        result.current_price = float(latest['close'])
        result.ma5 = float(latest['MA5'])
        result.ma10 = float(latest['MA10'])
        result.ma20 = float(latest['MA20'])
        result.ma60 = float(latest.get('MA60', 0))
        
        # 1. 趋势判断
        self._analyze_trend(df, result)

        # 2. 乖离率计算
        self._calculate_bias(result)

        # 3. 量能分析
        self._analyze_volume(df, result)

        # 4. 支撑压力分析
        self._analyze_support_resistance(df, result)

        # 5. 新技术指标分析
        self._analyze_rsi(df, result)
        self._analyze_macd(df, result)
        self._analyze_boll(df, result)

        # 6. 风险评估（第二周新增）
        self._assess_risk(df, result)

        # 7. 生成买入信号
        self._generate_signal(result)

        return result
    
    def _calculate_mas(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均线"""
        df = df.copy()
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        if len(df) >= 60:
            df['MA60'] = df['close'].rolling(window=60).mean()
        else:
            df['MA60'] = df['MA20']  # 数据不足时使用 MA20 替代
        return df
    
    def _analyze_trend(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析趋势状态
        
        核心逻辑：判断均线排列和趋势强度
        """
        ma5, ma10, ma20 = result.ma5, result.ma10, result.ma20
        
        # 判断均线排列
        if ma5 > ma10 > ma20:
            # 检查间距是否在扩大（强势）
            prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]
            prev_spread = (prev['MA5'] - prev['MA20']) / prev['MA20'] * 100 if prev['MA20'] > 0 else 0
            curr_spread = (ma5 - ma20) / ma20 * 100 if ma20 > 0 else 0
            
            if curr_spread > prev_spread and curr_spread > 5:
                result.trend_status = TrendStatus.STRONG_BULL
                result.ma_alignment = "强势多头排列，均线发散上行"
                result.trend_strength = 90
            else:
                result.trend_status = TrendStatus.BULL
                result.ma_alignment = "多头排列 MA5>MA10>MA20"
                result.trend_strength = 75
                
        elif ma5 > ma10 and ma10 <= ma20:
            result.trend_status = TrendStatus.WEAK_BULL
            result.ma_alignment = "弱势多头，MA5>MA10 但 MA10≤MA20"
            result.trend_strength = 55
            
        elif ma5 < ma10 < ma20:
            prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]
            prev_spread = (prev['MA20'] - prev['MA5']) / prev['MA5'] * 100 if prev['MA5'] > 0 else 0
            curr_spread = (ma20 - ma5) / ma5 * 100 if ma5 > 0 else 0
            
            if curr_spread > prev_spread and curr_spread > 5:
                result.trend_status = TrendStatus.STRONG_BEAR
                result.ma_alignment = "强势空头排列，均线发散下行"
                result.trend_strength = 10
            else:
                result.trend_status = TrendStatus.BEAR
                result.ma_alignment = "空头排列 MA5<MA10<MA20"
                result.trend_strength = 25
                
        elif ma5 < ma10 and ma10 >= ma20:
            result.trend_status = TrendStatus.WEAK_BEAR
            result.ma_alignment = "弱势空头，MA5<MA10 但 MA10≥MA20"
            result.trend_strength = 40
            
        else:
            result.trend_status = TrendStatus.CONSOLIDATION
            result.ma_alignment = "均线缠绕，趋势不明"
            result.trend_strength = 50
    
    def _calculate_bias(self, result: TrendAnalysisResult) -> None:
        """
        计算乖离率
        
        乖离率 = (现价 - 均线) / 均线 * 100%
        
        严进策略：乖离率超过 5% 不追高
        """
        price = result.current_price
        
        if result.ma5 > 0:
            result.bias_ma5 = (price - result.ma5) / result.ma5 * 100
        if result.ma10 > 0:
            result.bias_ma10 = (price - result.ma10) / result.ma10 * 100
        if result.ma20 > 0:
            result.bias_ma20 = (price - result.ma20) / result.ma20 * 100
    
    def _analyze_volume(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析量能
        
        偏好：缩量回调 > 放量上涨 > 缩量上涨 > 放量下跌
        """
        if len(df) < 5:
            return
        
        latest = df.iloc[-1]
        vol_5d_avg = df['volume'].iloc[-6:-1].mean()
        
        if vol_5d_avg > 0:
            result.volume_ratio_5d = float(latest['volume']) / vol_5d_avg
        
        # 判断价格变化
        prev_close = df.iloc[-2]['close']
        price_change = (latest['close'] - prev_close) / prev_close * 100
        
        # 量能状态判断
        if result.volume_ratio_5d >= self.VOLUME_HEAVY_RATIO:
            if price_change > 0:
                result.volume_status = VolumeStatus.HEAVY_VOLUME_UP
                result.volume_trend = "放量上涨，多头力量强劲"
            else:
                result.volume_status = VolumeStatus.HEAVY_VOLUME_DOWN
                result.volume_trend = "放量下跌，注意风险"
        elif result.volume_ratio_5d <= self.VOLUME_SHRINK_RATIO:
            if price_change > 0:
                result.volume_status = VolumeStatus.SHRINK_VOLUME_UP
                result.volume_trend = "缩量上涨，上攻动能不足"
            else:
                result.volume_status = VolumeStatus.SHRINK_VOLUME_DOWN
                result.volume_trend = "缩量回调，洗盘特征明显（好）"
        else:
            result.volume_status = VolumeStatus.NORMAL
            result.volume_trend = "量能正常"
    
    def _analyze_support_resistance(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析支撑压力位
        
        买点偏好：回踩 MA5/MA10 获得支撑
        """
        price = result.current_price
        
        # 检查是否在 MA5 附近获得支撑
        if result.ma5 > 0:
            ma5_distance = abs(price - result.ma5) / result.ma5
            if ma5_distance <= self.MA_SUPPORT_TOLERANCE and price >= result.ma5:
                result.support_ma5 = True
                result.support_levels.append(result.ma5)
        
        # 检查是否在 MA10 附近获得支撑
        if result.ma10 > 0:
            ma10_distance = abs(price - result.ma10) / result.ma10
            if ma10_distance <= self.MA_SUPPORT_TOLERANCE and price >= result.ma10:
                result.support_ma10 = True
                if result.ma10 not in result.support_levels:
                    result.support_levels.append(result.ma10)
        
        # MA20 作为重要支撑
        if result.ma20 > 0 and price >= result.ma20:
            result.support_levels.append(result.ma20)
        
        # 近期高点作为压力
        if len(df) >= 20:
            recent_high = df['high'].iloc[-20:].max()
            if recent_high > price:
                result.resistance_levels.append(recent_high)

    def _analyze_rsi(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析RSI指标

        判断逻辑：
        - RSI < 30: 超卖，买入信号
        - 30 <= RSI < 40: 低位，积极
        - 40 <= RSI <= 60: 中性
        - 60 < RSI < 70: 高位，观望
        - RSI >= 70: 超买，卖出信号
        """
        try:
            if 'rsi' not in df.columns:
                logger.debug(f"{result.code}: RSI指标不存在，跳过分析")
                return

            if df.empty:
                logger.warning(f"{result.code}: 数据为空，无法分析RSI")
                return

            latest_rsi = df['rsi'].iloc[-1]
            result.rsi = float(latest_rsi) if not pd.isna(latest_rsi) else None

            if result.rsi is not None and (result.rsi < 0 or result.rsi > 100):
                logger.warning(f"{result.code}: RSI值异常 {result.rsi}，超出范围[0,100]")

        except Exception as e:
            logger.error(f"{result.code}: RSI分析出错 - {str(e)}")
            result.rsi = None

    def _analyze_macd(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析MACD指标

        判断逻辑：
        - 金叉（DIF上穿DEA）：买入信号
        - 死叉（DIF下穿DEA）：卖出信号
        - 多头（DIF>0 且 DEA>0）：趋势向上
        - 空头（DIF<0 且 DEA<0）：趋势向下
        """
        try:
            if len(df) < 2 or 'dif' not in df.columns or 'dea' not in df.columns:
                logger.debug(f"{result.code}: MACD指标数据不足，跳过分析")
                return

            # 当前和前一天的DIF、DEA
            curr_dif = df['dif'].iloc[-1]
            curr_dea = df['dea'].iloc[-1]
            prev_dif = df['dif'].iloc[-2]
            prev_dea = df['dea'].iloc[-2]

            # 检查数据有效性
            if pd.isna(curr_dif) or pd.isna(curr_dea) or pd.isna(prev_dif) or pd.isna(prev_dea):
                logger.debug(f"{result.code}: MACD数据包含NaN，跳过分析")
                return

            # 判断金叉/死叉
            if prev_dif <= prev_dea and curr_dif > curr_dea:
                result.macd_signal = 'golden_cross'
            elif prev_dif >= prev_dea and curr_dif < curr_dea:
                result.macd_signal = 'death_cross'
            elif curr_dif > 0 and curr_dea > 0:
                result.macd_signal = 'bullish'
            elif curr_dif < 0 and curr_dea < 0:
                result.macd_signal = 'bearish'
            else:
                result.macd_signal = 'neutral'

        except Exception as e:
            logger.error(f"{result.code}: MACD分析出错 - {str(e)}")
            result.macd_signal = None

    def _analyze_boll(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析布林带指标

        判断逻辑：
        - 价格触及下轨：超卖，买入信号
        - 价格接近下轨（下轨+5%以内）：积极
        - 价格在中轨附近：中性
        - 价格接近上轨（上轨-5%以内）：注意压力
        - 价格触及上轨：超买，注意回调
        """
        try:
            if len(df) < 1 or 'boll_upper' not in df.columns or 'boll_lower' not in df.columns:
                logger.debug(f"{result.code}: 布林带指标不存在，跳过分析")
                return

            price = result.current_price
            boll_upper = df['boll_upper'].iloc[-1]
            boll_middle = df['boll_middle'].iloc[-1]
            boll_lower = df['boll_lower'].iloc[-1]

            # 检查数据有效性
            if pd.isna(boll_upper) or pd.isna(boll_lower) or pd.isna(boll_middle):
                logger.debug(f"{result.code}: 布林带数据包含NaN，跳过分析")
                return

            if boll_upper == boll_lower:
                logger.debug(f"{result.code}: 布林带上下轨相等，无法分析")
                return

            # 计算价格在布林带中的位置百分比
            boll_range = boll_upper - boll_lower
            position_pct = (price - boll_lower) / boll_range if boll_range > 0 else 0.5

            # 判断位置
            if position_pct <= 0.05:  # 触及下轨
                result.boll_position = 'lower_touch'
            elif position_pct <= 0.2:  # 接近下轨
                result.boll_position = 'lower_near'
            elif 0.4 <= position_pct <= 0.6:  # 中轨附近
                result.boll_position = 'middle'
            elif position_pct >= 0.95:  # 触及上轨
                result.boll_position = 'upper_touch'
            elif position_pct >= 0.8:  # 接近上轨
                result.boll_position = 'upper_near'
            else:
                result.boll_position = 'unknown'

        except Exception as e:
            logger.error(f"{result.code}: 布林带分析出错 - {str(e)}")
            result.boll_position = None
    
    def _generate_signal(self, result: TrendAnalysisResult) -> None:
        """
        生成买入信号

        综合评分系统（总分120分）：
        - 趋势（40分）：多头排列得分高
        - 乖离率（20分）：接近 MA5 得分高
        - 量能（15分）：缩量回调得分高
        - 支撑（5分）：获得均线支撑得分高
        - RSI（15分）：超卖区加分
        - MACD（15分）：金叉加分
        - BOLL（10分）：接近下轨加分
        """
        score = 0
        reasons = []
        risks = []
        
        # === 趋势评分（40分）===
        trend_scores = {
            TrendStatus.STRONG_BULL: 40,
            TrendStatus.BULL: 35,
            TrendStatus.WEAK_BULL: 25,
            TrendStatus.CONSOLIDATION: 15,
            TrendStatus.WEAK_BEAR: 10,
            TrendStatus.BEAR: 5,
            TrendStatus.STRONG_BEAR: 0,
        }
        trend_score = trend_scores.get(result.trend_status, 15)
        score += trend_score
        
        if result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL]:
            reasons.append(f"✅ {result.trend_status.value}，顺势做多")
        elif result.trend_status in [TrendStatus.BEAR, TrendStatus.STRONG_BEAR]:
            risks.append(f"⚠️ {result.trend_status.value}，不宜做多")
        
        # === 乖离率评分（20分）===
        bias = result.bias_ma5
        if bias < 0:
            # 价格在 MA5 下方（回调中）
            if bias > -3:
                score += 20
                reasons.append(f"✅ 价格略低于MA5({bias:.1f}%)，回踩买点")
            elif bias > -5:
                score += 15
                reasons.append(f"✅ 价格回踩MA5({bias:.1f}%)，观察支撑")
            else:
                score += 5
                risks.append(f"⚠️ 乖离率过大({bias:.1f}%)，可能破位")
        elif bias < 2:
            score += 18
            reasons.append(f"✅ 价格贴近MA5({bias:.1f}%)，介入好时机")
        elif bias < self.BIAS_THRESHOLD:
            score += 12
            reasons.append(f"⚡ 价格略高于MA5({bias:.1f}%)，可小仓介入")
        else:
            score += 3
            risks.append(f"❌ 乖离率过高({bias:.1f}%>5%)，严禁追高！")

        # === 量能评分（15分）===
        volume_scores = {
            VolumeStatus.SHRINK_VOLUME_DOWN: 15,  # 缩量回调最佳
            VolumeStatus.HEAVY_VOLUME_UP: 12,     # 放量上涨次之
            VolumeStatus.NORMAL: 10,
            VolumeStatus.SHRINK_VOLUME_UP: 6,     # 无量上涨较差
            VolumeStatus.HEAVY_VOLUME_DOWN: 0,    # 放量下跌最差
        }
        vol_score = volume_scores.get(result.volume_status, 8)
        score += vol_score

        if result.volume_status == VolumeStatus.SHRINK_VOLUME_DOWN:
            reasons.append("✅ 缩量回调，主力洗盘")
        elif result.volume_status == VolumeStatus.HEAVY_VOLUME_DOWN:
            risks.append("⚠️ 放量下跌，注意风险")

        # === 支撑评分（5分）===
        if result.support_ma5:
            score += 3
            reasons.append("✅ MA5支撑有效")
        if result.support_ma10:
            score += 2
            reasons.append("✅ MA10支撑有效")

        # === RSI评分（15分）===
        if hasattr(result, 'rsi') and result.rsi is not None:
            rsi = result.rsi
            if rsi < 30:
                score += 15
                reasons.append(f"✅ RSI超卖区({rsi:.1f})，反弹机会")
            elif rsi < 40:
                score += 12
                reasons.append(f"✅ RSI低位({rsi:.1f})，偏积极")
            elif 40 <= rsi <= 60:
                score += 8
                reasons.append(f"⚡ RSI中性({rsi:.1f})，观望")
            elif rsi > 70:
                score += 0
                risks.append(f"⚠️ RSI超买({rsi:.1f})，注意回调")
            else:
                score += 5

        # === MACD评分（15分）===
        if hasattr(result, 'macd_signal') and result.macd_signal:
            if result.macd_signal == 'golden_cross':
                score += 15
                reasons.append("✅ MACD金叉，趋势向上")
            elif result.macd_signal == 'bullish':
                score += 10
                reasons.append("✅ MACD多头，趋势良好")
            elif result.macd_signal == 'death_cross':
                score += 0
                risks.append("⚠️ MACD死叉，趋势转弱")
            elif result.macd_signal == 'bearish':
                score += 3
                risks.append("⚠️ MACD空头，谨慎参与")
            else:
                score += 7

        # === BOLL评分（10分）===
        if hasattr(result, 'boll_position') and result.boll_position:
            if result.boll_position == 'lower_touch':
                score += 10
                reasons.append("✅ 价格触及下轨，超卖反弹")
            elif result.boll_position == 'lower_near':
                score += 8
                reasons.append("✅ 接近下轨，反弹可能")
            elif result.boll_position == 'middle':
                score += 5
                reasons.append("⚡ 价格在中轨，震荡")
            elif result.boll_position == 'upper_near':
                score += 3
                risks.append("⚠️ 接近上轨，注意压力")
            elif result.boll_position == 'upper_touch':
                score += 0
                risks.append("⚠️ 触及上轨，回调风险")
            else:
                score += 5
        
        # === 综合判断 ===
        result.signal_score = score
        result.signal_reasons = reasons
        result.risk_factors = risks

        # 生成买入信号（总分120分，按比例调整阈值）
        if score >= 95 and result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL]:
            result.buy_signal = BuySignal.STRONG_BUY
        elif score >= 78 and result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL, TrendStatus.WEAK_BULL]:
            result.buy_signal = BuySignal.BUY
        elif score >= 60:
            result.buy_signal = BuySignal.HOLD
        elif score >= 42:
            result.buy_signal = BuySignal.WAIT
        elif result.trend_status in [TrendStatus.BEAR, TrendStatus.STRONG_BEAR]:
            result.buy_signal = BuySignal.STRONG_SELL
        else:
            result.buy_signal = BuySignal.SELL

    def _assess_risk(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        风险评估（第二周新增）

        使用多维度风险评估系统进行综合风险分析

        Args:
            df: 股票数据
            result: 趋势分析结果
        """
        try:
            from risk_analyzer import RiskAnalyzer

            risk_analyzer = RiskAnalyzer()
            risk_assessment = risk_analyzer.assess_risk(df, result.code, result)

            # 将风险评估结果添加到趋势分析结果中
            result.risk_score = risk_assessment.total_risk_score
            result.risk_level = risk_assessment.risk_level.value

            # 合并风险警告
            if risk_assessment.risk_warnings:
                result.risk_factors.extend(risk_assessment.risk_warnings)

            # 合并黑天鹅事件
            if risk_assessment.black_swans:
                result.risk_factors.extend(risk_assessment.black_swans)

            logger.debug(f"{result.code}: 风险评估完成 - 风险等级{result.risk_level}")

        except Exception as e:
            logger.error(f"{result.code}: 风险评估失败 - {str(e)}")
            # 设置默认风险等级
            result.risk_score = 50
            result.risk_level = "中等风险"

    def format_analysis(self, result: TrendAnalysisResult) -> str:
        """
        格式化分析结果为文本
        
        Args:
            result: 分析结果
            
        Returns:
            格式化的分析文本
        """
        lines = [
            f"=== {result.code} 趋势分析 ===",
            f"",
            f"📊 趋势判断: {result.trend_status.value}",
            f"   均线排列: {result.ma_alignment}",
            f"   趋势强度: {result.trend_strength}/100",
            f"",
            f"📈 均线数据:",
            f"   现价: {result.current_price:.2f}",
            f"   MA5:  {result.ma5:.2f} (乖离 {result.bias_ma5:+.2f}%)",
            f"   MA10: {result.ma10:.2f} (乖离 {result.bias_ma10:+.2f}%)",
            f"   MA20: {result.ma20:.2f} (乖离 {result.bias_ma20:+.2f}%)",
            f"",
            f"📊 量能分析: {result.volume_status.value}",
            f"   量比(vs5日): {result.volume_ratio_5d:.2f}",
            f"   量能趋势: {result.volume_trend}",
            f"",
            f"🎯 操作建议: {result.buy_signal.value}",
            f"   综合评分: {result.signal_score}/100",
        ]
        
        if result.signal_reasons:
            lines.append(f"")
            lines.append(f"✅ 买入理由:")
            for reason in result.signal_reasons:
                lines.append(f"   {reason}")
        
        if result.risk_factors:
            lines.append(f"")
            lines.append(f"⚠️ 风险因素:")
            for risk in result.risk_factors:
                lines.append(f"   {risk}")
        
        return "\n".join(lines)


def analyze_stock(df: pd.DataFrame, code: str) -> TrendAnalysisResult:
    """
    便捷函数：分析单只股票
    
    Args:
        df: 包含 OHLCV 数据的 DataFrame
        code: 股票代码
        
    Returns:
        TrendAnalysisResult 分析结果
    """
    analyzer = StockTrendAnalyzer()
    return analyzer.analyze(df, code)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 模拟数据测试
    import numpy as np
    
    dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
    np.random.seed(42)
    
    # 模拟多头排列的数据
    base_price = 10.0
    prices = [base_price]
    for i in range(59):
        change = np.random.randn() * 0.02 + 0.003  # 轻微上涨趋势
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices],
    })
    
    analyzer = StockTrendAnalyzer()
    result = analyzer.analyze(df, '000001')
    print(analyzer.format_analysis(result))
