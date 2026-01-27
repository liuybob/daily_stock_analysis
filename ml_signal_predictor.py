# -*- coding: utf-8 -*-
"""
===================================
机器学习信号预测模型
===================================

功能：
1. 训练信号预测模型（买入/卖出/持有）
2. 模型评估和验证
3. 模型保存和加载
4. 预测接口

设计原则：
- 支持多种模型（随机森林、XGBoost、LightGBM）
- 完整的交叉验证
- 丰富的评估指标
- 易于扩展
"""

import logging
import pickle
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """模型类型枚举"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class SignalType(Enum):
    """信号类型枚举"""
    BUY = 1       # 买入信号
    HOLD = 0      # 持有/观望
    SELL = -1     # 卖出信号


@dataclass
class ModelConfig:
    """模型配置"""
    # 模型类型
    model_type: ModelType = ModelType.RANDOM_FOREST

    # 随机森林参数
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42

    # 训练参数
    test_size: float = 0.2
    use_time_series_split: bool = True
    n_splits: int = 5

    # 标签生成参数
    forward_days: int = 5      # 未来N天
    profit_threshold: float = 0.03   # 盈利阈值3%
    loss_threshold: float = -0.02    # 亏损阈值-2%


@dataclass
class ModelEvaluation:
    """模型评估结果"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    cv_scores: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc': self.auc,
            'cv_mean': self.cv_scores.mean(),
            'cv_std': self.cv_scores.std(),
        }


class SignalPredictionModel:
    """
    信号预测模型

    功能：
    1. 生成训练标签（基于未来收益）
    2. 训练模型
    3. 评估模型
    4. 预测信号
    5. 保存和加载模型
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        初始化模型

        Args:
            config: 模型配置
        """
        self.config = config or ModelConfig()
        self.model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.evaluation: Optional[ModelEvaluation] = None

    def generate_labels(
        self,
        df: pd.DataFrame,
        forward_days: Optional[int] = None,
        profit_threshold: Optional[float] = None,
        loss_threshold: Optional[float] = None
    ) -> pd.Series:
        """
        生成训练标签

        规则：
        - 未来N天收益率 > profit_threshold → BUY (1)
        - 未来N天收益率 < loss_threshold → SELL (-1)
        - 其他 → HOLD (0)

        Args:
            df: 包含close列的DataFrame
            forward_days: 向前看的天数
            profit_threshold: 盈利阈值
            loss_threshold: 亏损阈值

        Returns:
            标签Series
        """
        forward_days = forward_days or self.config.forward_days
        profit_threshold = profit_threshold or self.config.profit_threshold
        loss_threshold = loss_threshold or self.config.loss_threshold

        close = df['close']

        # 计算未来N天的收益率
        future_return = close.shift(-forward_days) / close - 1

        # 生成标签
        labels = pd.Series(0, index=df.index)
        labels[future_return > profit_threshold] = SignalType.BUY.value
        labels[future_return < loss_threshold] = SignalType.SELL.value

        # 移除最后N天的数据（没有未来收益）
        labels = labels.iloc[:-forward_days]

        # 统计标签分布
        label_counts = labels.value_counts()
        logger.info(f"标签分布: BUY={label_counts.get(1, 0)}, "
                   f"HOLD={label_counts.get(0, 0)}, "
                   f"SELL={label_counts.get(-1, 0)}")

        return labels

    def train(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        fit_label_encoder: bool = True
    ) -> None:
        """
        训练模型

        Args:
            features: 特征DataFrame
            df: 原始数据（用于生成标签）
            fit_label_encoder: 是否拟合标签编码器
        """
        # 生成标签
        labels = self.generate_labels(df)

        # 对齐特征和标签
        aligned_features = features.loc[labels.index]

        self.feature_names = aligned_features.columns.tolist()
        X = aligned_features.values
        y = labels.values

        logger.info(f"训练数据: X={X.shape}, y={y.shape}")

        # 分割训练集和测试集
        if self.config.use_time_series_split:
            # 时间序列分割
            split_idx = int(len(X) * (1 - self.config.test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            # 随机分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )

        logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

        # 创建模型
        self.model = self._create_model()

        # 训练模型
        logger.info("开始训练模型...")
        self.model.fit(X_train, y_train)
        logger.info("模型训练完成")

        # 评估模型
        self.evaluation = self._evaluate_model(X_test, y_test)
        self._log_evaluation_results()

        # 交叉验证
        self._cross_validate(X_train, y_train)

    def _create_model(self) -> Any:
        """
        创建模型实例

        Returns:
            模型对象
        """
        if self.config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.config.model_type == ModelType.XGBOOST:
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth or 6,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    eval_metric='mlogloss'
                )
            except ImportError:
                logger.warning("XGBoost未安装，使用随机森林")
                self.config.model_type = ModelType.RANDOM_FOREST
                return self._create_model()
        elif self.config.model_type == ModelType.LIGHTGBM:
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth or -1,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
            except ImportError:
                logger.warning("LightGBM未安装，使用随机森林")
                self.config.model_type = ModelType.RANDOM_FOREST
                return self._create_model()
        else:
            raise ValueError(f"不支持的模型类型: {self.config.model_type}")

    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelEvaluation:
        """
        评估模型

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            评估结果
        """
        y_pred = self.model.predict(X_test)

        # 基本指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # AUC（需要概率）
        try:
            y_proba = self.model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except (ValueError, AttributeError) as e:
            logger.debug(f"无法计算AUC: {e}")
            auc = 0.0

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 分类报告
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        return ModelEvaluation(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc=auc,
            confusion_matrix=cm,
            classification_report=report,
            cv_scores=np.array([])
        )

    def _cross_validate(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        交叉验证

        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        if self.config.use_time_series_split:
            cv = TimeSeriesSplit(n_splits=self.config.n_splits)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.config.n_splits, shuffle=True,
                      random_state=self.config.random_state)

        scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

        self.evaluation.cv_scores = scores

        logger.info(f"交叉验证准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    def _log_evaluation_results(self) -> None:
        """记录评估结果"""
        if not self.evaluation:
            return

        logger.info("=" * 60)
        logger.info("模型评估结果")
        logger.info("=" * 60)
        logger.info(f"准确率: {self.evaluation.accuracy:.4f}")
        logger.info(f"精确率: {self.evaluation.precision:.4f}")
        logger.info(f"召回率: {self.evaluation.recall:.4f}")
        logger.info(f"F1分数: {self.evaluation.f1_score:.4f}")
        logger.info(f"AUC: {self.evaluation.auc:.4f}")
        logger.info("\n混淆矩阵:")
        logger.info(f"\n{self.evaluation.confusion_matrix}")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        预测信号

        Args:
            features: 特征DataFrame

        Returns:
            预测的信号数组
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        X = features[self.feature_names].values
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        预测信号概率

        Args:
            features: 特征DataFrame

        Returns:
            预测的概率数组
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        X = features[self.feature_names].values
        probabilities = self.model.predict_proba(X)

        return probabilities

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性

        Args:
            top_n: 返回前N个重要特征

        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        importances = self.model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filepath: str) -> None:
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'evaluation': self.evaluation
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"模型已保存到: {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'SignalPredictionModel':
        """
        加载模型

        Args:
            filepath: 模型路径

        Returns:
            模型实例
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls(config=model_data['config'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.evaluation = model_data['evaluation']

        logger.info(f"模型已加载: {filepath}")
        return instance


def train_model_from_data(
    df: pd.DataFrame,
    feature_engineering,
    config: Optional[ModelConfig] = None
) -> SignalPredictionModel:
    """
    从数据训练模型的便捷函数

    Args:
        df: 包含OHLCV和技术指标的DataFrame
        feature_engineering: 特征工程实例
        config: 模型配置

    Returns:
        训练好的模型
    """
    # 提取特征
    features = feature_engineering.extract_features(df, fit_scaler=True)

    # 创建并训练模型
    model = SignalPredictionModel(config)
    model.train(features, df)

    return model


if __name__ == "__main__":
    # 测试模型
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from feature_engineering import FeatureEngineering, FeatureConfig, ScalingMethod, create_sample_data

    # 创建示例数据
    df = create_sample_data(500)

    # 添加技术指标
    from data_provider.base import BaseFetcher
    class TempFetcher(BaseFetcher):
        name = "TempFetcher"
        priority = 99
        def _fetch_raw_data(self, stock_code, start_date, end_date):
            return pd.DataFrame()
        def _normalize_data(self, df, stock_code):
            return df

    fetcher = TempFetcher()
    df = fetcher._calculate_indicators(df)

    # 特征工程
    fe_config = FeatureConfig(scaling_method=ScalingMethod.STANDARD)
    fe = FeatureEngineering(fe_config)
    features = fe.extract_features(df)

    # 训练模型
    model_config = ModelConfig(
        model_type=ModelType.RANDOM_FOREST,
        n_estimators=100,
        test_size=0.2,
        forward_days=5,
        profit_threshold=0.03,
        loss_threshold=-0.02
    )

    model = SignalPredictionModel(model_config)
    model.train(features, df)

    # 特征重要性
    importance = model.get_feature_importance(top_n=10)
    print("\n特征重要性 Top 10:")
    print(importance)

    # 保存模型
    model.save_model("models/signal_predictor.pkl")
