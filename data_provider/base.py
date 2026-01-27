# -*- coding: utf-8 -*-
"""
===================================
数据源基类与管理器
===================================

设计模式：策略模式 (Strategy Pattern)
- BaseFetcher: 抽象基类，定义统一接口
- DataFetcherManager: 策略管理器，实现自动切换

防封禁策略：
1. 每个 Fetcher 内置流控逻辑
2. 失败自动切换到下一个数据源
3. 指数退避重试机制
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# 配置日志
logger = logging.getLogger(__name__)


# === 标准化列名定义 ===
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']


class DataFetchError(Exception):
    """数据获取异常基类"""
    pass


class RateLimitError(DataFetchError):
    """API 速率限制异常"""
    pass


class DataSourceUnavailableError(DataFetchError):
    """数据源不可用异常"""
    pass


class BaseFetcher(ABC):
    """
    数据源抽象基类
    
    职责：
    1. 定义统一的数据获取接口
    2. 提供数据标准化方法
    3. 实现通用的技术指标计算
    
    子类实现：
    - _fetch_raw_data(): 从具体数据源获取原始数据
    - _normalize_data(): 将原始数据转换为标准格式
    """
    
    name: str = "BaseFetcher"
    priority: int = 99  # 优先级数字越小越优先
    
    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据源获取原始数据（子类必须实现）
        
        Args:
            stock_code: 股票代码，如 '600519', '000001'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            原始数据 DataFrame（列名因数据源而异）
        """
        pass
    
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化数据列名（子类必须实现）
        
        将不同数据源的列名统一为：
        ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        """
        pass
    
    def get_daily_data(
        self, 
        stock_code: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        获取日线数据（统一入口）
        
        流程：
        1. 计算日期范围
        2. 调用子类获取原始数据
        3. 标准化列名
        4. 计算技术指标
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选，默认今天）
            days: 获取天数（当 start_date 未指定时使用）
            
        Returns:
            标准化的 DataFrame，包含技术指标
        """
        # 计算日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # 默认获取最近 30 个交易日（按日历日估算，多取一些）
            from datetime import timedelta
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        logger.info(f"[{self.name}] 获取 {stock_code} 数据: {start_date} ~ {end_date}")
        
        try:
            # Step 1: 获取原始数据
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)
            
            if raw_df is None or raw_df.empty:
                raise DataFetchError(f"[{self.name}] 未获取到 {stock_code} 的数据")
            
            # Step 2: 标准化列名
            df = self._normalize_data(raw_df, stock_code)
            
            # Step 3: 数据清洗
            df = self._clean_data(df)
            
            # Step 4: 计算技术指标
            df = self._calculate_indicators(df)
            
            logger.info(f"[{self.name}] {stock_code} 获取成功，共 {len(df)} 条数据")
            return df
            
        except Exception as e:
            logger.error(f"[{self.name}] 获取 {stock_code} 失败: {str(e)}")
            raise DataFetchError(f"[{self.name}] {stock_code}: {str(e)}") from e
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        处理：
        1. 确保日期列格式正确
        2. 数值类型转换
        3. 去除空值行
        4. 按日期排序
        """
        df = df.copy()
        
        # 确保日期列为 datetime 类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 数值列类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 去除关键列为空的行
        df = df.dropna(subset=['close', 'volume'])
        
        # 按日期升序排序
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标

        计算指标：
        - MA5, MA10, MA20: 移动平均线
        - Volume_Ratio: 量比（今日成交量 / 5日平均成交量）
        - RSI: 相对强弱指标
        - MACD: 平滑异同移动平均线
        - BOLL: 布林带
        - ATR: 平均真实波幅
        """
        df = df.copy()

        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()

        # 量比：当日成交量 / 5日平均成交量
        avg_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / avg_volume_5.shift(1)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

        # RSI (相对强弱指标)
        df = self._calculate_rsi(df)

        # MACD (平滑异同移动平均线)
        df = self._calculate_macd(df)

        # BOLL (布林带)
        df = self._calculate_boll(df)

        # ATR (平均真实波幅)
        df = self._calculate_atr(df)

        # 保留2位小数
        for col in ['ma5', 'ma10', 'ma20', 'volume_ratio', 'rsi', 'dif', 'dea', 'macd',
                    'boll_upper', 'boll_middle', 'boll_lower', 'atr']:
            if col in df.columns:
                df[col] = df[col].round(2)

        return df

    def _calculate_rsi(self, df: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
        """
        计算RSI指标

        Args:
            df: 包含close列的DataFrame
            periods: RSI周期，默认14

        Returns:
            添加了rsi列的DataFrame
        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

        # 避免除零错误：当loss为0时，rs设为无穷大，RSI=100
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)  # 默认值

        # 当loss为0且gain>0时，RSI应该为100
        df.loc[loss == 0, 'rsi'] = 100
        # 当gain为0且loss>0时，RSI应该为0
        df.loc[gain == 0, 'rsi'] = 0

        return df

    def _calculate_macd(self, df: pd.DataFrame,
                        fast_period: int = 12,
                        slow_period: int = 26,
                        signal_period: int = 9) -> pd.DataFrame:
        """
        计算MACD指标

        Args:
            df: 包含close列的DataFrame
            fast_period: 快线周期，默认12
            slow_period: 慢线周期，默认26
            signal_period: 信号线周期，默认9

        Returns:
            添加了dif, dea, macd列的DataFrame
        """
        exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
        df['dif'] = exp1 - exp2
        df['dea'] = df['dif'].ewm(span=signal_period, adjust=False).mean()
        df['macd'] = 2 * (df['dif'] - df['dea'])
        # 初始值填充为0
        df['dif'] = df['dif'].fillna(0)
        df['dea'] = df['dea'].fillna(0)
        df['macd'] = df['macd'].fillna(0)
        return df

    def _calculate_boll(self, df: pd.DataFrame, periods: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        计算布林带指标

        Args:
            df: 包含close列的DataFrame
            periods: 周期，默认20
            std_dev: 标准差倍数，默认2

        Returns:
            添加了boll_upper, boll_middle, boll_lower列的DataFrame
        """
        df['boll_middle'] = df['close'].rolling(window=periods, min_periods=1).mean()
        std = df['close'].rolling(window=periods, min_periods=1).std()
        df['boll_upper'] = df['boll_middle'] + std_dev * std
        df['boll_lower'] = df['boll_middle'] - std_dev * std
        return df

    def _calculate_atr(self, df: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
        """
        计算ATR (Average True Range) 指标

        Args:
            df: 包含high, low, close列的DataFrame
            periods: ATR周期，默认14

        Returns:
            添加了atr列的DataFrame
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=periods, min_periods=1).mean()
        df['atr'] = df['atr'].fillna(0)
        return df
    
    @staticmethod
    def random_sleep(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:
        """
        智能随机休眠（Jitter）
        
        防封禁策略：模拟人类行为的随机延迟
        在请求之间加入不规则的等待时间
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        logger.debug(f"随机休眠 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)


class DataFetcherManager:
    """
    数据源策略管理器
    
    职责：
    1. 管理多个数据源（按优先级排序）
    2. 自动故障切换（Failover）
    3. 提供统一的数据获取接口
    
    切换策略：
    - 优先使用高优先级数据源
    - 失败后自动切换到下一个
    - 所有数据源都失败时抛出异常
    """
    
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        """
        初始化管理器
        
        Args:
            fetchers: 数据源列表（可选，默认按优先级自动创建）
        """
        self._fetchers: List[BaseFetcher] = []
        
        if fetchers:
            # 按优先级排序
            self._fetchers = sorted(fetchers, key=lambda f: f.priority)
        else:
            # 默认数据源将在首次使用时延迟加载
            self._init_default_fetchers()
    
    def _init_default_fetchers(self) -> None:
        """
        初始化默认数据源列表
        
        按优先级排序：
        0. EfinanceFetcher (Priority 0) - 最高优先级
        1. AkshareFetcher (Priority 1)
        2. TushareFetcher (Priority 2)
        3. BaostockFetcher (Priority 3)
        4. YfinanceFetcher (Priority 4)
        """
        from .efinance_fetcher import EfinanceFetcher
        from .akshare_fetcher import AkshareFetcher
        from .tushare_fetcher import TushareFetcher
        from .baostock_fetcher import BaostockFetcher
        from .yfinance_fetcher import YfinanceFetcher
        
        self._fetchers = [
            EfinanceFetcher(),   # 最高优先级
            AkshareFetcher(),
            TushareFetcher(),
            BaostockFetcher(),
            YfinanceFetcher(),
        ]
        
        # 按优先级排序
        self._fetchers.sort(key=lambda f: f.priority)
        
        logger.info(f"已初始化 {len(self._fetchers)} 个数据源: " + 
                   ", ".join([f.name for f in self._fetchers]))
    
    def add_fetcher(self, fetcher: BaseFetcher) -> None:
        """添加数据源并重新排序"""
        self._fetchers.append(fetcher)
        self._fetchers.sort(key=lambda f: f.priority)
    
    def get_daily_data(
        self, 
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> Tuple[pd.DataFrame, str]:
        """
        获取日线数据（自动切换数据源）
        
        故障切换策略：
        1. 从最高优先级数据源开始尝试
        2. 捕获异常后自动切换到下一个
        3. 记录每个数据源的失败原因
        4. 所有数据源失败后抛出详细异常
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 获取天数
            
        Returns:
            Tuple[DataFrame, str]: (数据, 成功的数据源名称)
            
        Raises:
            DataFetchError: 所有数据源都失败时抛出
        """
        errors = []
        
        for fetcher in self._fetchers:
            try:
                logger.info(f"尝试使用 [{fetcher.name}] 获取 {stock_code}...")
                df = fetcher.get_daily_data(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    days=days
                )
                
                if df is not None and not df.empty:
                    logger.info(f"[{fetcher.name}] 成功获取 {stock_code}")
                    return df, fetcher.name
                    
            except Exception as e:
                error_msg = f"[{fetcher.name}] 失败: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                # 继续尝试下一个数据源
                continue
        
        # 所有数据源都失败
        error_summary = f"所有数据源获取 {stock_code} 失败:\n" + "\n".join(errors)
        logger.error(error_summary)
        raise DataFetchError(error_summary)
    
    @property
    def available_fetchers(self) -> List[str]:
        """返回可用数据源名称列表"""
        return [f.name for f in self._fetchers]


class TechnicalIndicators:
    """
    技术指标计算类

    提供主流技术指标计算方法：

    趋势指标:
    - MA: 移动平均线
    - MACD: 指数平滑移动平均线
    - DMI: 趋向指标
    - TRIX: 三重指数平滑均线

    超买超卖指标:
    - RSI: 相对强弱指标
    - KDJ: 随机指标
    - WR: 威廉指标 (威廉指标)
    - CCI: 顺势指标

    能量指标:
    - OBV: 能量潮
    - VR: 成交量变异率

    波动性指标:
    - BOLL: 布林带
    - ATR: 真实波幅

    动量指标:
    - ROC: 变动率
    - BIAS: 乖离率
    """

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            添加了技术指标的DataFrame
        """
        df = df.copy()

        # ========== 趋势指标 ==========

        # MA 移动平均线
        df['MA_5'] = df['close'].rolling(window=5).mean()
        df['MA_10'] = df['close'].rolling(window=10).mean()
        df['MA_20'] = df['close'].rolling(window=20).mean()
        df['MA_60'] = df['close'].rolling(window=60).mean()

        # MACD
        df = self._calculate_macd(df)

        # DMI 趋向指标
        df = self._calculate_dmi(df)

        # TRIX 三重指数平滑均线
        df = self._calculate_trix(df)

        # ========== 超买超卖指标 ==========

        # RSI 相对强弱指标
        df = self._calculate_rsi(df)

        # KDJ 随机指标
        df = self._calculate_kdj(df)

        # WR 威廉指标
        df = self._calculate_wr(df)

        # CCI 顺势指标
        df = self._calculate_cci(df)

        # ========== 能量指标 ==========

        # OBV 能量潮
        df = self._calculate_obv(df)

        # VR 成交量变异率
        df = self._calculate_vr(df)

        # ========== 波动性指标 ==========

        # BOLL 布林带
        df = self._calculate_boll(df)

        # ATR 真实波幅
        df = self._calculate_atr(df)

        # ========== 动量指标 ==========

        # ROC 变动率
        df = self._calculate_roc(df)

        # BIAS 乖离率
        df = self._calculate_bias(df)

        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
        """计算RSI指标"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, 
                        slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """计算MACD指标"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_HIST'] = 2 * (df['MACD'] - df['MACD_SIGNAL'])
        return df
    
    def _calculate_boll(self, df: pd.DataFrame, periods: int = 20, 
                       std_dev: float = 2.0) -> pd.DataFrame:
        """计算布林带指标"""
        df['BOLL_MIDDLE'] = df['close'].rolling(window=periods).mean()
        std = df['close'].rolling(window=periods).std()
        df['BOLL_UPPER'] = df['BOLL_MIDDLE'] + std_dev * std
        df['BOLL_LOWER'] = df['BOLL_MIDDLE'] - std_dev * std
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
        """计算ATR指标"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=periods).mean()
        df['ATR'] = df['ATR'].fillna(0)
        return df

    def _calculate_kdj(self, df: pd.DataFrame, n: int = 9,
                       m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """
        计算KDJ随机指标

        Args:
            df: 包含OHLC数据的DataFrame
            n: RSV周期
            m1: K周期
            m2: D周期

        Returns:
            添加了KDJ指标的DataFrame
        """
        # 计算RSV
        low_min = df['low'].rolling(window=n).min()
        high_max = df['high'].rolling(window=n).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)

        # 计算K, D, J
        k = rsv.ewm(alpha=m1 / n, adjust=False).mean().fillna(50)
        d = rsv.ewm(alpha=m2 / n, adjust=False).mean().fillna(50)
        j = 3 * k - 2 * d

        # 标准化到0-100
        df['KDJ_K'] = k.clip(0, 100)
        df['KDJ_D'] = d.clip(0, 100)
        df['KDJ_J'] = j.clip(0, 100)

        return df

    def _calculate_wr(self, df: pd.DataFrame, periods: int = 14,
                     overbought: int = 80, oversold: int = 20) -> pd.DataFrame:
        """
        计算WR威廉指标

        Args:
            df: 包含OHLC数据的DataFrame
            periods: 周期
            overbought: 超买阈值
            oversold: 超卖阈值

        Returns:
            添加了WR指标的DataFrame
        """
        high_max = df['high'].rolling(window=periods).max()
        low_min = df['low'].rolling(window=periods).min()

        if high_max.max() == low_min.max():
            # 所有价格相同，返回50
            df['WR'] = 50.0
        else:
            wr = (high_max - df['close']) / (high_max - low_min) * -100
            df['WR'] = wr.fillna(50)

        return df

    def _calculate_cci(self, df: pd.DataFrame, periods: int = 20,
                       constant: float = 0.015) -> pd.DataFrame:
        """
        计算CCI顺势指标

        Args:
            df: 包含OHLCV数据的DataFrame
            periods: 周期
            constant: 常数

        Returns:
            添加了CCI指标的DataFrame
        """
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=periods).mean()

        md = (tp - ma).abs()
        md = md.rolling(window=periods).mean().fillna(0)

        df['CCI'] = (tp - ma) / (constant * md).replace(0, np.nan)
        df['CCI'] = df['CCI'].fillna(0)

        return df

    def _calculate_dmi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算DMI趋向指标

        Args:
            df: 包含OHLCV数据的DataFrame
            period: 周期

        Returns:
            添加了DMI指标的DataFrame
        """
        # 计算价格变化
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        # 计算+DM和-DM
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

        # 计算TR (True Range)
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算平滑的+DM, -DM和TR
        alpha = 1.0 / period
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
        tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()

        # 计算+DI和-DI
        plus_di = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
        minus_di = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)

        # 计算DX和ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = dx.ewm(alpha=alpha, adjust=False).mean().fillna(0)

        # 计算ADXR
        adxr_smooth = df['ADX'].ewm(alpha=alpha, adjust=False).mean()
        df['ADXR'] = ((df['ADX'] + adxr_smooth.shift(1)) / 2).fillna(0)

        # ADX趋势判断
        df['ADX_Trend'] = df['ADX'].apply(
            lambda x: '强势' if x > 25 else ('盘整' if x > 20 else '弱势')
        )

        return df

    def _calculate_trix(self, df: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
        """
        计算TRIX三重指数平滑均线

        Args:
            df: 包含收盘价的DataFrame
            periods: 周期

        Returns:
            添加了TRIX指标的DataFrame
        """
        # 计算三重指数平滑移动平均
        ema1 = df['close'].ewm(span=periods, adjust=False).mean()
        ema2 = ema1.ewm(span=periods, adjust=False).mean()
        trix = ema2.ewm(span=periods, adjust=False).mean()

        # 计算TRIX变化率
        df['TRIX'] = (trix - trix.shift(1)) / trix.shift(1) * 100
        df['TRIX'] = df['TRIX'].fillna(0)

        return df

    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算OBV能量潮

        Args:
            df: 包含收盘价和成交量的DataFrame

        Returns:
            添加了OBV指标的DataFrame
        """
        # 计算价格变动方向
        direction = df['close'].diff().apply(lambda x: 1 if x > 0 else -1)
        direction = direction.fillna(0)

        # 计算OBV
        obv = (direction * df['volume']).cumsum()
        df['OBV'] = obv

        return df

    def _calculate_vr(self, df: pd.DataFrame, periods: int = 26) -> pd.DataFrame:
        """
        计算VR成交量变异率

        Args:
            df: 包含收盘价和成交量的DataFrame
            periods: 周期

        Returns:
            添加了VR指标的DataFrame
        """
        # 计算上涨日和下跌日的成交量
        df['volume_up'] = df['volume'].where(df['close'] > df['close'].shift(1), 0)
        df['volume_down'] = df['volume'].where(df['close'] <= df['close'].shift(1), 0)

        # 计算N日平均成交量
        vol_mean = df['volume'].rolling(window=periods).mean()

        # 计算VR
        vr_up = df['volume_up'].rolling(window=periods).sum()
        vr_down = df['volume_down'].rolling(window=periods).sum()
        df['VR'] = (vr_up / vr_down).replace([np.inf, -np.inf], 1).fillna(1)

        # VR判断
        df['VR_Signal'] = df['VR'].apply(
            lambda x: '强势' if x > 450 else ('弱势' if x < 200 else '观望')
        )

        return df

    def _calculate_roc(self, df: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
        """
        计算ROC变动率

        Args:
            df: 包含收盘价的DataFrame
            periods: 周期

        Returns:
            添加了ROC指标的DataFrame
        """
        df['ROC'] = df['close'].pct_change(periods) * 100
        df['ROC'] = df['ROC'].fillna(0)

        # ROC信号
        df['ROC_Signal'] = df['ROC'].apply(
            lambda x: '强势' if x > 5 else ('弱势' if x < -5 else '观望')
        )

        return df

    def _calculate_bias(self, df: pd.DataFrame, periods: list = [6, 12, 24]) -> pd.DataFrame:
        """
        计算BIAS乖离率

        Args:
            df: 包含收盘价的DataFrame
            periods: 周期列表

        Returns:
            添加了BIAS指标的DataFrame
        """
        for period in periods:
            ma = df['close'].rolling(window=period).mean()
            bias = (df['close'] - ma) / ma * 100
            df[f'BIAS_{period}'] = bias.fillna(0)

        return df
