# -*- coding: utf-8 -*-
"""
===================================
机器学习特征工程模块
===================================

功能：
1. 从现有技术指标提取特征
2. 创建衍生特征（价格动量、波动率等）
3. 特征标准化和归一化
4. 特征选择

设计原则：
- 模块化设计，易于扩展
- 支持多种特征转换方法
- 完整的类型提示和文档
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """特征类型枚举"""
    PRICE_MOMENTUM = "price_momentum"      # 价格动量
    VOLATILITY = "volatility"              # 波动率
    VOLUME = "volume"                      # 成交量
    TECHNICAL_INDICATOR = "technical"      # 技术指标
    DERIVED = "derived"                    # 衍生特征


class ScalingMethod(Enum):
    """标准化方法枚举"""
    STANDARD = "standard"      # StandardScaler (均值0,方差1)
    MINMAX = "minmax"          # MinMaxScaler (0-1)
    ROBUST = "robust"          # RobustScaler (基于中位数和四分位数)
    NONE = "none"              # 不做标准化


@dataclass
class FeatureConfig:
    """特征配置"""
    # 是否包含价格动量特征
    include_price_momentum: bool = True
    # 是否包含波动率特征
    include_volatility: bool = True
    # 是否包含成交量特征
    include_volume: bool = True
    # 是否包含技术指标特征
    include_technical: bool = True
    # 是否包含衍生特征
    include_derived: bool = True
    # 标准化方法
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    # 特征选择方法：'all', 'correlation', 'variance'
    feature_selection: str = 'all'
    # 相关性阈值（用于特征选择）
    correlation_threshold: float = 0.95
    # 方差阈值（用于特征选择）
    variance_threshold: float = 0.01


class FeatureEngineering:
    """
    特征工程类

    功能：
    1. 提取技术指标特征
    2. 创建衍生特征
    3. 特征标准化
    4. 特征选择
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        初始化特征工程

        Args:
            config: 特征配置，默认使用默认配置
        """
        self.config = config or FeatureConfig()
        self.scaler: Optional[any] = None
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []

    def extract_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        提取所有特征

        Args:
            df: 包含OHLCV和技术指标的DataFrame
            fit_scaler: 是否拟合scaler（训练时为True，预测时为False）

        Returns:
            包含所有特征的DataFrame
        """
        features_df = pd.DataFrame(index=df.index)

        # 1. 价格动量特征
        if self.config.include_price_momentum:
            momentum_features = self._extract_price_momentum_features(df)
            features_df = pd.concat([features_df, momentum_features], axis=1)

        # 2. 波动率特征
        if self.config.include_volatility:
            volatility_features = self._extract_volatility_features(df)
            features_df = pd.concat([features_df, volatility_features], axis=1)

        # 3. 成交量特征
        if self.config.include_volume:
            volume_features = self._extract_volume_features(df)
            features_df = pd.concat([features_df, volume_features], axis=1)

        # 4. 技术指标特征
        if self.config.include_technical:
            technical_features = self._extract_technical_features(df)
            features_df = pd.concat([features_df, technical_features], axis=1)

        # 5. 衍生特征
        if self.config.include_derived:
            derived_features = self._extract_derived_features(df)
            features_df = pd.concat([features_df, derived_features], axis=1)

        # 处理无穷值和NaN
        features_df = self._handle_invalid_values(features_df)

        # 特征标准化
        if self.config.scaling_method != ScalingMethod.NONE:
            features_df = self._scale_features(features_df, fit=fit_scaler)

        # 特征选择
        if self.config.feature_selection != 'all' and fit_scaler:
            features_df = self._select_features(features_df)

        self.feature_names = features_df.columns.tolist()

        logger.info(f"提取了 {len(self.feature_names)} 个特征")
        return features_df

    def _extract_price_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取价格动量特征

        Args:
            df: 原始数据

        Returns:
            价格动量特征DataFrame
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # 1. 简单收益率
        features['return_1d'] = close.pct_change(1, fill_method=None)
        features['return_3d'] = close.pct_change(3, fill_method=None)
        features['return_5d'] = close.pct_change(5, fill_method=None)
        features['return_10d'] = close.pct_change(10, fill_method=None)
        features['return_20d'] = close.pct_change(20, fill_method=None)

        # 2. 对数收益率
        features['log_return_1d'] = np.log(close / close.shift(1))
        features['log_return_5d'] = np.log(close / close.shift(5))

        # 3. 动量指标
        features['momentum_5d'] = close / close.shift(5) - 1
        features['momentum_10d'] = close / close.shift(10) - 1
        features['momentum_20d'] = close / close.shift(20) - 1

        # 4. 价格相对位置
        if 'high' in df.columns and 'low' in df.columns:
            features['price_position'] = (close - df['low']) / (df['high'] - df['low'])
            features['price_position'] = features['price_position'].fillna(0.5)

        # 5. 与均线的关系
        if 'ma5' in df.columns:
            features['close_to_ma5'] = close / df['ma5'] - 1
        if 'ma10' in df.columns:
            features['close_to_ma10'] = close / df['ma10'] - 1
        if 'ma20' in df.columns:
            features['close_to_ma20'] = close / df['ma20'] - 1

        logger.debug(f"提取价格动量特征: {features.shape[1]}个")
        return features

    def _extract_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取波动率特征

        Args:
            df: 原始数据

        Returns:
            波动率特征DataFrame
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # 1. 历史波动率（标准差）
        features['volatility_5d'] = close.pct_change(fill_method=None).rolling(5).std()
        features['volatility_10d'] = close.pct_change(fill_method=None).rolling(10).std()
        features['volatility_20d'] = close.pct_change(fill_method=None).rolling(20).std()

        # 2. ATR相关（如果存在）
        if 'atr' in df.columns:
            features['atr_ratio'] = df['atr'] / close
            if 'ma20' in df.columns:
                features['atr_to_ma20'] = df['atr'] / df['ma20']

        # 3. 布林带宽度（如果存在）
        if 'boll_upper' in df.columns and 'boll_lower' in df.columns:
            features['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_middle']

        # 4. 价格范围
        if 'high' in df.columns and 'low' in df.columns:
            features['range_ratio'] = (df['high'] - df['low']) / close

        logger.debug(f"提取波动率特征: {features.shape[1]}个")
        return features

    def _extract_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取成交量特征

        Args:
            df: 原始数据

        Returns:
            成交量特征DataFrame
        """
        features = pd.DataFrame(index=df.index)

        if 'volume' not in df.columns:
            return features

        volume = df['volume']

        # 1. 成交量变化率
        features['volume_change_1d'] = volume.pct_change(1)
        features['volume_change_5d'] = volume.pct_change(5)

        # 2. 成交量移动平均
        features['volume_ma5_ratio'] = volume / volume.rolling(5).mean()
        features['volume_ma10_ratio'] = volume / volume.rolling(10).mean()
        features['volume_ma20_ratio'] = volume / volume.rolling(20).mean()

        # 3. 量比（如果存在）
        if 'volume_ratio' in df.columns:
            features['volume_ratio_raw'] = df['volume_ratio']

        # 4. 价格成交量关系
        if 'close' in df.columns:
            price_change = df['close'].pct_change(fill_method=None)
            volume_change = volume.pct_change(fill_method=None)
            # 量价相关性（正相关=量价齐升）
            features['price_volume_corr_5d'] = price_change.rolling(5).corr(volume_change)
            features['price_volume_corr_10d'] = price_change.rolling(10).corr(volume_change)

        # 5. 成交额（如果存在）
        if 'amount' in df.columns:
            features['amount_ma5_ratio'] = df['amount'] / df['amount'].rolling(5).mean()
            features['amount_ma10_ratio'] = df['amount'] / df['amount'].rolling(10).mean()

        logger.debug(f"提取成交量特征: {features.shape[1]}个")
        return features

    def _extract_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取技术指标特征

        Args:
            df: 包含技术指标的DataFrame

        Returns:
            技术指标特征DataFrame
        """
        features = pd.DataFrame(index=df.index)

        # 1. RSI相关
        if 'rsi' in df.columns:
            features['rsi_raw'] = df['rsi']
            features['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            features['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            features['rsi_neutral'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(int)

        # 2. MACD相关
        if 'dif' in df.columns and 'dea' in df.columns:
            features['macd_dif'] = df['dif']
            features['macd_dea'] = df['dea']
            features['macd_bar'] = df['macd'] if 'macd' in df.columns else 2 * (df['dif'] - df['dea'])
            features['macd_golden_cross'] = ((df['dif'] > df['dea']) & (df['dif'].shift(1) <= df['dea'].shift(1))).astype(int)
            features['macd_death_cross'] = ((df['dif'] < df['dea']) & (df['dif'].shift(1) >= df['dea'].shift(1))).astype(int)

        # 3. 布林带相关
        if 'boll_upper' in df.columns and 'boll_lower' in df.columns and 'close' in df.columns:
            features['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])
            features['boll_position'] = features['boll_position'].fillna(0.5)

        # 4. 均线相关
        if 'ma5' in df.columns and 'ma10' in df.columns and 'ma20' in df.columns:
            # 均线多头排列
            features['ma_bullish_alignment'] = ((df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])).astype(int)
            # 均线空头排列
            features['ma_bearish_alignment'] = ((df['ma5'] < df['ma10']) & (df['ma10'] < df['ma20'])).astype(int)
            # 均线间距
            features['ma5_ma10_spread'] = (df['ma5'] - df['ma10']) / df['ma10']
            features['ma10_ma20_spread'] = (df['ma10'] - df['ma20']) / df['ma20']

        logger.debug(f"提取技术指标特征: {features.shape[1]}个")
        return features

    def _extract_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取衍生特征

        Args:
            df: 原始数据

        Returns:
            衍生特征DataFrame
        """
        features = pd.DataFrame(index=df.index)

        if 'close' not in df.columns:
            return features

        close = df['close']

        # 1. 加速度（二阶导数）
        features['acceleration_5d'] = close.pct_change(1, fill_method=None).diff(5)
        features['acceleration_10d'] = close.pct_change(1, fill_method=None).diff(10)

        # 2. 连续上涨/下跌天数
        price_change = close.pct_change(fill_method=None)
        features['consecutive_up_days'] = (price_change > 0).astype(int).groupby((price_change <= 0).cumsum()).cumsum()
        features['consecutive_down_days'] = (price_change < 0).astype(int).groupby((price_change >= 0).cumsum()).cumsum()

        # 3. 高低点位置
        if 'high' in df.columns and 'low' in df.columns:
            features['high_close_ratio'] = df['high'] / close - 1
            features['low_close_ratio'] = 1 - df['low'] / close

        # 4. 跳空特征
        features['gap_up'] = ((close > df['low'].shift(1)) & (close < df['high'].shift(1))).astype(int)
        features['gap_down'] = ((close < df['low'].shift(1)) | (close > df['high'].shift(1))).astype(int)

        # 5. 极值检测
        features['is_20d_high'] = (close == close.rolling(20).max()).astype(int)
        features['is_20d_low'] = (close == close.rolling(20).min()).astype(int)

        logger.debug(f"提取衍生特征: {features.shape[1]}个")
        return features

    def _handle_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理无穷值和NaN

        Args:
            df: 特征DataFrame

        Returns:
            处理后的DataFrame
        """
        df = df.copy()

        # 替换无穷值为NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # 填充NaN（使用前向填充）
        df = df.ffill()
        df = df.fillna(0)  # 剩余的NaN用0填充

        return df

    def _scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        特征标准化

        Args:
            df: 特征DataFrame
            fit: 是否拟合scaler

        Returns:
            标准化后的DataFrame
        """
        df = df.copy()

        # 创建scaler
        if self.scaler is None or fit:
            if self.config.scaling_method == ScalingMethod.STANDARD:
                self.scaler = StandardScaler()
            elif self.config.scaling_method == ScalingMethod.MINMAX:
                self.scaler = MinMaxScaler()
            elif self.config.scaling_method == ScalingMethod.ROBUST:
                self.scaler = RobustScaler()
            else:
                return df

        # 拟合和转换
        if fit:
            scaled_data = self.scaler.fit_transform(df)
        else:
            scaled_data = self.scaler.transform(df)

        df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

        logger.debug(f"特征标准化方法: {self.config.scaling_method.value}")
        return df_scaled

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征选择

        Args:
            df: 特征DataFrame

        Returns:
            选择后的DataFrame
        """
        if self.config.feature_selection == 'variance':
            # 基于方差的特征选择
            variances = df.var()
            selected = variances[variances > self.config.variance_threshold].index
            df = df[selected]
            self.selected_features = selected.tolist()

        elif self.config.feature_selection == 'correlation':
            # 基于相关性的特征选择（移除高度相关的特征）
            corr_matrix = df.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # 找出高度相关的特征
            high_corr_features = [
                column for column in upper_triangle.columns
                if any(upper_triangle[column] > self.config.correlation_threshold)
            ]

            # 移除高度相关的特征
            df = df.drop(columns=high_corr_features)
            self.selected_features = df.columns.tolist()

        logger.info(f"特征选择方法: {self.config.feature_selection}, 选择了 {len(self.selected_features)} 个特征")
        return df

    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表

        Returns:
            特征名称列表
        """
        return self.feature_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用已拟合的特征工程转换新数据

        Args:
            df: 原始数据

        Returns:
            转换后的特征
        """
        return self.extract_features(df, fit_scaler=False)


def create_sample_data(n_days: int = 100) -> pd.DataFrame:
    """
    创建示例数据用于测试

    Args:
        n_days: 天数

    Returns:
        示例数据DataFrame
    """
    np.random.seed(42)

    # 生成随机价格数据
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    close = 100 + np.cumsum(np.random.randn(n_days) * 2)
    high = close + np.random.rand(n_days) * 2
    low = close - np.random.rand(n_days) * 2
    open_ = close + np.random.randn(n_days) * 0.5
    volume = 1000000 + np.random.randint(-500000, 500000, n_days)

    df = pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'amount': close * volume,
    })

    return df


if __name__ == "__main__":
    # 测试特征工程
    logging.basicConfig(level=logging.INFO)

    # 创建示例数据
    df = create_sample_data(100)

    # 添加技术指标
    from data_provider.base import BaseFetcher
    df = BaseFetcher.calculate_indicators(df)

    # 特征工程
    config = FeatureConfig(
        include_price_momentum=True,
        include_volatility=True,
        include_volume=True,
        include_technical=True,
        include_derived=True,
        scaling_method=ScalingMethod.STANDARD,
        feature_selection='all'
    )

    fe = FeatureEngineering(config)
    features = fe.extract_features(df)

    print(f"\n提取的特征数量: {len(fe.get_feature_names())}")
    print(f"\n特征列表: {fe.get_feature_names()}")
    print(f"\n特征形状: {features.shape}")
    print(f"\n前5行特征:")
    print(features.head())
