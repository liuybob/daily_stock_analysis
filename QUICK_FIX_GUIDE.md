# 快速修复指南

## 🔍 诊断结果汇总

✅ **通过的检查** (11/15):
- 所有核心模块可导入
- PortfolioConfig 和 ModelConfig 存在
- 技术指标模块找到 (data_provider.base)

❌ **失败的检查** (4/15):
1. `risk_analyzer.RiskConfig` 不存在
2. `position_sizer.PositionSizerConfig` 不存在
3. `portfolio_manager.Portfolio.calculate_metrics` 不存在
4. `risk_analyzer.RiskAnalyzer.analyze_risk` 不存在

---

## 🛠️ 具体修复方案

### 1. 添加 `RiskConfig` 类

**文件**: `risk_analyzer.py`
**位置**: 在文件开头的类定义区域

```python
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
```

### 2. 添加 `PositionSizerConfig` 类

**文件**: `position_sizer.py`
**位置**: 在文件开头的类定义区域

```python
@dataclass
class PositionSizerConfig:
    """
    仓位配置

    Attributes:
        total_capital: 总资金
        max_position_pct: 单个股票最大仓位百分比
        win_rate: 胜率
        avg_win: 平均盈利
        avg_loss: 平均亏损
        risk_per_trade: 单笔交易风险百分比
    """
    total_capital: float = 100000
    max_position_pct: float = 0.2
    win_rate: float = 0.5
    avg_win: float = 0.03
    avg_loss: float = 0.02
    risk_per_trade: float = 0.02
```

### 3. 添加 `calculate_metrics` 方法

**文件**: `portfolio_manager.py`
**类**: `Portfolio`

```python
def calculate_metrics(self) -> Dict[str, Any]:
    """
    计算组合绩效指标

    Returns:
        包含绩效指标的字典:
        - total_value: 总市值
        - total_pnl: 总盈亏
        - total_return: 总收益率
        - win_rate: 胜率
        - max_drawdown: 最大回撤
    """
    total_value = sum(pos.market_value for pos in self.positions.values())
    total_cost = sum(pos.cost_basis for pos in self.positions.values())
    total_pnl = total_value - total_cost
    total_return = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    # 计算胜率
    winning_positions = [p for p in self.positions.values() if p.pnl > 0]
    win_rate = len(winning_positions) / len(self.positions) if self.positions else 0

    return {
        'total_value': total_value,
        'total_pnl': total_pnl,
        'total_return': total_return / 100,  # 转为小数
        'win_rate': win_rate,
        'max_drawdown': 0.0,  # 需要历史数据计算
        'position_count': len(self.positions),
    }
```

### 4. 修复 `analyze_risk` 方法

**文件**: `risk_analyzer.py`
**问题**: 可能方法名不匹配或签名不同

检查 `RiskAnalyzer` 类中是否有 `analyze_risk` 方法，如果没有，添加：

```python
def analyze_risk(
    self,
    symbol: str,
    data: pd.DataFrame,
    lookback: int = 20
) -> Dict[str, Any]:
    """
    分析股票风险

    Args:
        symbol: 股票代码
        data: 价格数据
        lookback: 回看周期

    Returns:
        风险分析结果字典
    """
    # 实现风险分析逻辑
    return self.calculate_risk_score(symbol, data, lookback)
```

---

## 📋 修复清单

- [ ] 1. 在 `risk_analyzer.py` 中添加 `RiskConfig` 类
- [ ] 2. 在 `position_sizer.py` 中添加 `PositionSizerConfig` 类
- [ ] 3. 在 `portfolio_manager.py` 的 `Portfolio` 类中添加 `calculate_metrics` 方法
- [ ] 4. 验证 `risk_analyzer.py` 中的 `analyze_risk` 方法
- [ ] 5. 修复测试文件中的导入路径 (将 `stock_analysis.base` 改为 `data_provider.base`)
- [ ] 6. 运行 `python3 run_tests.py` 验证修复
- [ ] 7. 运行 `python3 -m pytest test_*.py -v` 运行所有测试

---

## 🚀 快速修复命令

```bash
# 1. 运行诊断
python3 run_tests.py

# 2. 应用修复（手动编辑文件或使用脚本）
# TODO: 创建自动化修复脚本

# 3. 验证修复
python3 run_tests.py

# 4. 运行测试
python3 -m pytest test_*.py -v
```

---

## 📊 预期结果

修复后，所有测试应该能够：
- ✅ 模块导入成功
- ✅ 配置类存在
- ✅ 关键方法可调用
- ✅ 测试文件能够运行

---

**下一步**: 查看完整代码审查报告
```bash
cat CODE_REVIEW_REPORT.md
```
