#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本 - 诊断和运行所有测试

运行方式:
    python3 run_tests.py              # 运行所有测试
    python3 run_tests.py --fix        # 修复并运行
    python3 run_tests.py --verbose    # 详细输出
"""

import sys
import importlib
import traceback
from pathlib import Path

def test_module_import(module_name: str) -> tuple:
    """测试模块导入"""
    try:
        module = importlib.import_module(module_name)
        return True, f"✅ {module_name} 导入成功"
    except ImportError as e:
        return False, f"❌ {module_name} 导入失败: {e}"
    except Exception as e:
        return False, f"❌ {module_name} 错误: {e}"

def test_class_exists(module_name: str, class_name: str) -> tuple:
    """测试类是否存在"""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, class_name):
            return True, f"✅ {module_name}.{class_name} 存在"
        else:
            return False, f"❌ {module_name}.{class_name} 不存在"
    except Exception as e:
        return False, f"❌ 检查 {module_name}.{class_name} 时出错: {e}"

def test_method_exists(module_name: str, class_name: str, method_name: str) -> tuple:
    """测试方法是否存在"""
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if hasattr(cls, method_name):
            return True, f"✅ {module_name}.{class_name}.{method_name} 存在"
        else:
            return False, f"❌ {module_name}.{class_name}.{method_name} 不存在"
    except Exception as e:
        return False, f"❌ 检查 {module_name}.{class_name}.{method_name} 时出错: {e}"

def main():
    """主函数"""
    verbose = "--verbose" in sys.argv

    print("""
╔══════════════════════════════════════════════════════════╗
║         A股分析系统 - 测试诊断工具                        ║
╚══════════════════════════════════════════════════════════╝
    """)

    results = []

    # 1. 测试核心模块导入
    print("="*60)
    print("【1. 核心模块导入测试】")
    print("="*60)

    modules = [
        "config",
        "storage",
        "analyzer",
        "risk_analyzer",
        "feature_engineering",
        "ml_signal_predictor",
        "portfolio_manager",
        "position_sizer",
    ]

    for module in modules:
        success, msg = test_module_import(module)
        results.append(success)
        print(msg)
        if verbose and not success:
            traceback.print_exc()

    # 2. 测试配置类
    print("\n" + "="*60)
    print("【2. 配置类测试】")
    print("="*60)

    config_tests = [
        ("risk_analyzer", "RiskConfig"),
        ("position_sizer", "PositionSizerConfig"),
        ("portfolio_manager", "PortfolioConfig"),
        ("ml_signal_predictor", "ModelConfig"),
    ]

    for module, cls in config_tests:
        success, msg = test_class_exists(module, cls)
        results.append(success)
        print(msg)

    # 3. 测试关键方法
    print("\n" + "="*60)
    print("【3. 关键方法测试】")
    print("="*60)

    method_tests = [
        ("portfolio_manager", "Portfolio", "calculate_metrics"),
        ("risk_analyzer", "RiskAnalyzer", "analyze_risk"),
    ]

    for module, cls, method in method_tests:
        success, msg = test_method_exists(module, cls, method)
        results.append(success)
        print(msg)

    # 4. 测试技术指标模块
    print("\n" + "="*60)
    print("【4. 技术指标模块测试】")
    print("="*60)

    # 尝试多种可能的导入路径
    indicator_paths = [
        "sources.base",
        "data_provider.base",
        "stock_analysis.base",
    ]

    indicator_found = False
    for path in indicator_paths:
        success, msg = test_module_import(path)
        if success:
            print(f"✅ 技术指标模块找到: {path}")
            indicator_found = True
            results.append(True)
            break
        else:
            if verbose:
                print(f"❌ 未找到: {path}")

    if not indicator_found:
        print("❌ 技术指标模块未找到，尝试了以下路径:")
        for path in indicator_paths:
            print(f"   - {path}")
        results.append(False)

    # 5. 测试文件分析
    print("\n" + "="*60)
    print("【5. 测试文件分析】")
    print("="*60)

    test_files = list(Path(".").glob("test_*.py"))
    print(f"找到 {len(test_files)} 个测试文件:")

    for test_file in test_files:
        print(f"  - {test_file.name}")

    # 统计测试函数
    total_tests = 0
    for test_file in test_files:
        content = test_file.read_text(encoding='utf-8')
        test_count = content.count('def test_')
        total_tests += test_count
        print(f"    {test_count} 个测试函数")

    print(f"\n总计: {total_tests} 个测试函数")

    # 总结
    print("\n" + "="*60)
    print("【诊断总结】")
    print("="*60)

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"总检查项: {total}")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")

    if failed > 0:
        print(f"\n⚠️  发现 {failed} 个问题")
        print(f"\n建议修复:")

        # 检查具体问题并给出建议
        success, _ = test_class_exists("risk_analyzer", "RiskConfig")
        if not success:
            print(f"  1. 在 risk_analyzer.py 中添加 RiskConfig 类")

        success, _ = test_class_exists("position_sizer", "PositionSizerConfig")
        if not success:
            print(f"  2. 在 position_sizer.py 中添加 PositionSizerConfig 类")

        success, _ = test_method_exists("portfolio_manager", "Portfolio", "calculate_metrics")
        if not success:
            print(f"  3. 在 portfolio_manager.py 的 Portfolio 类中添加 calculate_metrics 方法")

        if not indicator_found:
            print(f"  4. 确认技术指标模块的正确导入路径")

        print(f"\n查看详细报告: cat CODE_REVIEW_REPORT.md")
        return 1
    else:
        print(f"\n✅ 所有检查通过！可以运行测试")
        print(f"\n运行测试: python3 -m pytest test_*.py -v")
        return 0

if __name__ == "__main__":
    sys.exit(main())
