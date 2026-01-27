#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证核心功能
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("股票分析系统 - 快速功能验证")
print("=" * 60)

# 测试1: 技术指标
print("\n【测试1】技术指标计算...")
try:
    from data_provider.base import TechnicalIndicators

    # 创建测试数据
    data = {
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(95, 115, 100),
        'volume': np.random.randint(1000000, 2000000, 100)
    }
    df = pd.DataFrame(data)

    # 计算指标
    indicators = TechnicalIndicators()
    result = indicators.calculate_all(df)

    print(f"✅ MA(5,20,60): {result['MA_5'].iloc[-1]:.2f}, {result['MA_20'].iloc[-1]:.2f}")
    print(f"✅ RSI: {result['RSI'].iloc[-1]:.2f}")
    print(f"✅ MACD: {result['MACD'].iloc[-1]:.2f}")
    print(f"✅ BOLL上轨: {result['BOLL_UPPER'].iloc[-1]:.2f}")
    print(f"✅ ATR: {result['ATR'].iloc[-1]:.2f}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试2: 风险评估
print("\n【测试2】风险评估...")
try:
    from risk_analyzer import RiskAnalyzer, RiskConfig

    analyzer = RiskAnalyzer(RiskConfig())

    # 创建测试数据
    test_data = pd.DataFrame({
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.randint(1000000, 2000000, 100)
    })

    risk_score = analyzer.analyze_risk('TEST', test_data)

    print(f"✅ 总体风险评分: {risk_score['total_risk_score']:.2f}")
    print(f"✅ 风险等级: {risk_score['risk_level']}")
    print(f"✅ 技术面风险: {risk_score['technical_risk']:.2f}")
    print(f"✅ 波动性风险: {risk_score['volatility_risk']:.2f}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试3: 特征工程
print("\n【测试3】特征工程...")
try:
    from feature_engineering import FeatureEngineering

    # 创建测试数据
    data = {
        'open': np.random.uniform(100, 110, 200),
        'high': np.random.uniform(110, 120, 200),
        'low': np.random.uniform(90, 100, 200),
        'close': np.random.uniform(95, 115, 200),
        'volume': np.random.randint(1000000, 2000000, 200)
    }
    df = pd.DataFrame(data)

    # 计算技术指标
    from data_provider.base import TechnicalIndicators
    indicators = TechnicalIndicators()
    df = indicators.calculate_all(df)

    # 提取特征
    fe = FeatureEngineering()
    features = fe.extract_features(df)

    print(f"✅ 提取特征数: {features.shape[1]}")
    print(f"✅ 样本数: {features.shape[0]}")
    print(f"✅ 前5个特征: {list(features.columns)[:5]}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试4: ML预测模型
print("\n【测试4】ML预测模型...")
try:
    from ml_signal_predictor import SignalPredictionModel, ModelConfig, SignalType
    from feature_engineering import FeatureEngineering
    from data_provider.base import TechnicalIndicators

    # 创建训练数据
    np.random.seed(42)
    data = {
        'open': np.random.uniform(100, 110, 500),
        'high': np.random.uniform(110, 120, 500),
        'low': np.random.uniform(90, 100, 500),
        'close': np.random.uniform(95, 115, 500),
        'volume': np.random.randint(1000000, 2000000, 500)
    }
    df = pd.DataFrame(data)

    # 计算指标和特征
    indicators = TechnicalIndicators()
    df = indicators.calculate_all(df)

    fe = FeatureEngineering()
    features = fe.extract_features(df)

    # 训练模型
    config = ModelConfig(model_type='random_forest', n_estimators=10, max_depth=5)
    model = SignalPredictionModel(config)
    model.train(features, df)

    # 评估
    eval_result = model.evaluate(features, df)
    print(f"✅ 模型准确率: {eval_result['accuracy']:.2%}")
    print(f"✅ F1分数: {eval_result['f1_score']:.4f}")

    # 预测
    prediction = model.predict(features.iloc[[-1]])
    print(f"✅ 预测信号: {SignalType(prediction).name}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试5: 组合管理
print("\n【测试5】组合管理...")
try:
    from portfolio_manager import Portfolio, PortfolioConfig

    config = PortfolioConfig(
        name="测试组合",
        initial_capital=100000,
        max_positions=5,
        max_single_weight=0.3
    )
    portfolio = Portfolio(config)

    # 添加持仓
    portfolio.add_position("AAPL", 100, 150)
    portfolio.add_position("MSFT", 50, 300)

    # 更新价格
    portfolio.update_position("AAPL", price=155)
    portfolio.update_position("MSFT", price=310)

    # 计算指标
    metrics = portfolio.calculate_metrics()

    print(f"✅ 总市值: ${metrics['total_value']:,.2f}")
    print(f"✅ 总收益: ${metrics['total_pnl']:,.2f}")
    print(f"✅ 收益率: {metrics['total_return']:.2%}")
    print(f"✅ 持仓数: {len(portfolio.positions)}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试6: 仓位配置
print("\n【测试6】动态仓位配置...")
try:
    from position_sizer import PositionSizer, PositionSizerConfig, SizingMethod

    # 测试固定比例法
    config1 = PositionSizerConfig(
        total_capital=100000,
        max_position_pct=0.2,
        fixed_ratio=0.1
    )
    sizer1 = PositionSizer(config1)
    pos1 = sizer1.calculate_position_size(
        symbol="AAPL",
        price=150.0,
        confidence=0.8
    )
    print(f"✅ 固定比例法: {pos1.shares:.0f}股 (${pos1.dollar_amount:,.2f})")

    # 测试Kelly公式
    config2 = PositionSizerConfig(
        total_capital=100000,
        max_position_pct=0.2,
        win_rate=0.55,
        avg_win=0.03,
        avg_loss=0.02
    )
    config2.method = SizingMethod.KELLY
    sizer2 = PositionSizer(config2)
    pos2 = sizer2.calculate_position_size(
        symbol="AAPL",
        price=150.0,
        confidence=0.8
    )
    print(f"✅ Kelly公式: {pos2.shares:.0f}股 (${pos2.dollar_amount:,.2f})")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试7: Web可视化
print("\n【测试7】Web可视化...")
try:
    from web_visualization import generate_dashboard_html
    from portfolio_manager import Portfolio, PortfolioConfig

    # 创建测试组合
    config = PortfolioConfig(name="Test", initial_capital=100000)
    portfolio = Portfolio(config)
    portfolio.add_position("AAPL", 100, 150)
    portfolio.add_position("MSFT", 50, 300)
    portfolio.update_position("AAPL", price=155)
    portfolio.update_position("MSFT", price=310)

    # 生成HTML
    html = generate_dashboard_html(portfolio_data=portfolio.to_dict())

    with open("quick_test_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ 仪表板已生成: quick_test_dashboard.html")
    print(f"✅ 文件大小: {len(html)} 字节")
except Exception as e:
    print(f"❌ 失败: {e}")

# 总结
print("\n" + "=" * 60)
print("快速验证完成！")
print("=" * 60)
print("\n💡 提示:")
print("1. 查看生成的 quick_test_dashboard.html 文件")
print("2. 运行完整测试: python3 -m pytest test_*.py -v")
print("3. 查看完整测试指南: cat TESTING_GUIDE.md")
