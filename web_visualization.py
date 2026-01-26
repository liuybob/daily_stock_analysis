# -*- coding: utf-8 -*-
"""
===================================
Web可视化增强模块
===================================

功能：
1. 投资组合可视化
2. ML预测可视化
3. 仓位配置可视化
4. 交互式图表

设计原则：
- 集成现有web框架
- 丰富的图表类型
- 实时数据更新
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChartData:
    """图表数据"""
    chart_type: str           # 图表类型：line, bar, pie, scatter
    title: str                # 标题
    x_data: List[Any]         # X轴数据
    y_data: List[Any]         # Y轴数据
    labels: Optional[List[str]] = None  # 标签
    colors: Optional[List[str]] = None  # 颜色


class PortfolioVisualizer:
    """
    投资组合可视化器

    功能：
    1. 持仓分布饼图
    2. 收益曲线图
    3. 风险指标图
    """

    @staticmethod
    def generate_portfolio_allocation_html(portfolio_data: Dict[str, Any]) -> str:
        """
        生成组合配置HTML

        Args:
            portfolio_data: 组合数据

        Returns:
            HTML字符串
        """
        positions = portfolio_data.get('positions', {})

        # 提取数据
        symbols = []
        weights = []
        values = []

        for symbol, pos in positions.items():
            symbols.append(symbol)
            weights.append(pos.get('weight', 0) * 100)
            values.append(pos.get('market_value', 0))

        # 生成HTML
        html = f"""
        <div class="portfolio-visualization">
            <h2>投资组合配置</h2>

            <!-- 饼图：持仓分布 -->
            <div class="chart-container">
                <canvas id="allocationChart"></canvas>
            </div>

            <script>
                new Chart(document.getElementById('allocationChart'), {{
                    type: 'pie',
                    data: {{
                        labels: {symbols},
                        datasets: [{{
                            data: {weights},
                            backgroundColor: [
                                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                                '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                            ]
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: '持仓分布'
                            }},
                            legend: {{
                                position: 'right'
                            }}
                        }}
                    }}
                }});
            </script>

            <!-- 持仓明细表 -->
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>股票代码</th>
                        <th>股数</th>
                        <th>当前价格</th>
                        <th>市值</th>
                        <th>权重</th>
                        <th>盈亏</th>
                        <th>盈亏%</th>
                    </tr>
                </thead>
                <tbody>
        """

        for symbol, pos in positions.items():
            html += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td>{pos.get('shares', 0):.0f}</td>
                        <td>{pos.get('current_price', 0):.2f}</td>
                        <td>{pos.get('market_value', 0):.2f}</td>
                        <td>{pos.get('weight', 0) * 100:.2f}%</td>
                        <td>{pos.get('pnl', 0):.2f}</td>
                        <td>{pos.get('pnl_percentage', 0):.2f}%</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

        return html

    @staticmethod
    def generate_performance_chart_html(daily_values: List[tuple]) -> str:
        """
        生成收益曲线图HTML

        Args:
            daily_values: 每日市值列表 [(date, value), ...]

        Returns:
            HTML字符串
        """
        if not daily_values:
            return "<p>暂无数据</p>"

        dates = [d[0] for d in daily_values]
        values = [v[1] for v in daily_values]

        # 计算收益率
        initial_value = values[0] if values else 1
        returns = [(v / initial_value - 1) * 100 for v in values]

        html = f"""
        <div class="performance-chart">
            <h2>组合收益曲线</h2>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>

            <script>
                new Chart(document.getElementById('performanceChart'), {{
                    type: 'line',
                    data: {{
                        labels: {dates},
                        datasets: [{{
                            label: '累计收益率 (%)',
                            data: {returns},
                            borderColor: '#36A2EB',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            fill: true,
                            tension: 0.4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: '累计收益率'
                            }}
                        }},
                        scales: {{
                            y: {{
                                title: {{
                                    display: true,
                                    text: '收益率 (%)'
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
        """

        return html


class MLPredictionVisualizer:
    """
    ML预测可视化器

    功能：
    1. 特征重要性图
    2. 预测概率分布
    3. 模型性能指标
    """

    @staticmethod
    def generate_feature_importance_html(importance_df: pd.DataFrame) -> str:
        """
        生成特征重要性图HTML

        Args:
            importance_df: 特征重要性DataFrame

        Returns:
            HTML字符串
        """
        if importance_df.empty:
            return "<p>暂无特征重要性数据</p>"

        # 取前20个特征
        top_features = importance_df.head(20)

        features = top_features['feature'].tolist()
        importance = top_features['importance'].tolist()

        html = f"""
        <div class="feature-importance">
            <h2>特征重要性 Top 20</h2>
            <div class="chart-container">
                <canvas id="importanceChart"></canvas>
            </div>

            <script>
                new Chart(document.getElementById('importanceChart'), {{
                    type: 'bar',
                    data: {{
                        labels: {features},
                        datasets: [{{
                            label: '重要性',
                            data: {importance},
                            backgroundColor: '#36A2EB'
                        }}]
                    }},
                    options: {{
                        indexAxis: 'y',
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: '特征重要性'
                            }}
                        }},
                        scales: {{
                            x: {{
                                title: {{
                                    display: true,
                                    text: '重要性分数'
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
        """

        return html

    @staticmethod
    def generate_model_metrics_html(evaluation: Dict[str, Any]) -> str:
        """
        生成模型指标HTML

        Args:
            evaluation: 模型评估结果

        Returns:
            HTML字符串
        """
        html = """
        <div class="model-metrics">
            <h2>模型性能指标</h2>
            <div class="metrics-grid">
        """

        metrics = [
            ('准确率', evaluation.get('accuracy', 0)),
            ('精确率', evaluation.get('precision', 0)),
            ('召回率', evaluation.get('recall', 0)),
            ('F1分数', evaluation.get('f1_score', 0)),
            ('AUC', evaluation.get('auc', 0)),
        ]

        for name, value in metrics:
            html += f"""
                <div class="metric-card">
                    <div class="metric-name">{name}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
            """

        html += """
            </div>
        </div>
        """

        return html


class PositionSizerVisualizer:
    """
    仓位配置可视化器

    功能：
    1. 仓位大小对比
    2. 风险分布
    """

    @staticmethod
    def generate_position_comparison_html(positions: List[Dict[str, Any]]) -> str:
        """
        生成仓位对比图HTML

        Args:
            positions: 仓位列表

        Returns:
            HTML字符串
        """
        if not positions:
            return "<p>暂无仓位数据</p>"

        symbols = [p['symbol'] for p in positions]
        weights = [p['weight'] * 100 for p in positions]
        amounts = [p['dollar_amount'] for p in positions]

        html = f"""
        <div class="position-comparison">
            <h2>仓位配置对比</h2>

            <!-- 权重柱状图 -->
            <div class="chart-container">
                <canvas id="weightChart"></canvas>
            </div>

            <script>
                new Chart(document.getElementById('weightChart'), {{
                    type: 'bar',
                    data: {{
                        labels: {symbols},
                        datasets: [{{
                            label: '权重 (%)',
                            data: {weights},
                            backgroundColor: '#36A2EB'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: '仓位权重分布'
                            }}
                        }},
                        scales: {{
                            y: {{
                                title: {{
                                    display: true,
                                    text: '权重 (%)'
                                }}
                            }}
                        }}
                    }}
                }});
            </script>

            <!-- 仓位明细表 -->
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>股票代码</th>
                        <th>股数</th>
                        <th>权重</th>
                        <th>金额</th>
                        <th>风险金额</th>
                    </tr>
                </thead>
                <tbody>
        """

        for p in positions:
            html += f"""
                    <tr>
                        <td>{p['symbol']}</td>
                        <td>{p['shares']}</td>
                        <td>{p['weight'] * 100:.2f}%</td>
                        <td>${p['dollar_amount']:.2f}</td>
                        <td>${p['risk_amount']:.2f}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

        return html


def generate_dashboard_html(
    portfolio_data: Optional[Dict[str, Any]] = None,
    ml_evaluation: Optional[Dict[str, Any]] = None,
    positions: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    生成完整的仪表板HTML

    Args:
        portfolio_data: 组合数据
        ml_evaluation: ML评估结果
        positions: 仓位数据

    Returns:
            HTML字符串
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>股票分析系统仪表板</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .dashboard { padding: 20px; }
            .chart-container { position: relative; height: 400px; margin: 20px 0; }
            .metric-card { text-align: center; padding: 20px; background: #f8f9fa; margin: 10px; border-radius: 8px; }
            .metric-name { font-size: 14px; color: #666; }
            .metric-value { font-size: 32px; font-weight: bold; color: #333; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        </style>
    </head>
    <body>
        <div class="dashboard">
            <h1>股票分析系统仪表板</h1>
    """

    # 添加各个部分
    if portfolio_data:
        portfolio_viz = PortfolioVisualizer()
        html += portfolio_viz.generate_portfolio_allocation_html(portfolio_data)

    if ml_evaluation:
        ml_viz = MLPredictionVisualizer()
        html += ml_viz.generate_model_metrics_html(ml_evaluation)

    if positions:
        sizer_viz = PositionSizerVisualizer()
        html += sizer_viz.generate_position_comparison_html(positions)

    html += """
        </div>
    </body>
    </html>
    """

    return html


if __name__ == "__main__":
    # 测试可视化
    from portfolio_manager import Portfolio, PortfolioConfig

    # 创建测试组合
    config = PortfolioConfig(name="Test", initial_capital=100000)
    portfolio = Portfolio(config)

    portfolio.add_position("AAPL", 100, 150)
    portfolio.add_position("MSFT", 50, 300)
    portfolio.update_position("AAPL", price=155)
    portfolio.update_position("MSFT", price=310)

    portfolio_data = portfolio.to_dict()

    # 生成HTML
    html = generate_dashboard_html(portfolio_data=portfolio_data)

    # 保存HTML
    with open("test_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ 仪表板HTML已生成: test_dashboard.html")
