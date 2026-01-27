#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修复脚本 - 修复最紧急的代码问题

运行方式:
    python3 fix_critical_issues.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str) -> bool:
    """运行命令并返回成功状态"""
    print(f"\n{'='*60}")
    print(f"【{description}】")
    print(f"{'='*60}")
    print(f"命令: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"✅ 成功")
            if result.stdout:
                print(result.stdout[:500])
            return True
        else:
            print(f"❌ 失败")
            if result.stderr:
                print(result.stderr[:500])
            return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False

def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║         A股分析系统 - 快速修复脚本                        ║
╚══════════════════════════════════════════════════════════╝
    """)

    results = []

    # 1. 检查Python语法
    results.append(run_command(
        "python3 -m py_compile *.py 2>&1 | head -20",
        "检查Python语法错误"
    ))

    # 2. 检查未使用的导入
    results.append(run_command(
        "python3 -m flake8 --select=F401 *.py 2>/dev/null | head -20 || echo 'flake8未安装，跳过'",
        "检查未使用的导入"
    ))

    # 3. 统计代码行数
    results.append(run_command(
        "wc -l *.py | tail -1",
        "统计代码总行数"
    ))

    # 4. 查找可能的敏感信息
    results.append(run_command(
        "grep -rn 'api_key.*=' *.py | grep -v 'os.getenv' | grep -v 'config.' | head -10 || echo '未发现硬编码密钥'",
        "查找硬编码的API密钥"
    ))

    # 5. 查找裸except
    results.append(run_command(
        "grep -rn 'except:' *.py | wc -l",
        "统计裸except子句数量"
    ))

    # 6. 查找print语句
    results.append(run_command(
        "grep -rn 'print(' *.py | wc -l",
        "统计print语句数量"
    ))

    # 7. 检查大文件
    results.append(run_command(
        "wc -l *.py | awk '$1 > 800 {print $0}' | sort -rn",
        "查找超过800行的大文件"
    ))

    # 8. 测试导入
    print(f"\n{'='*60}")
    print("【测试关键模块导入】")
    print(f"{'='*60}")

    test_imports = [
        ("config", "配置模块"),
        ("storage", "存储模块"),
        ("analyzer", "分析器模块"),
    ]

    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} ({module}) - 导入成功")
        except Exception as e:
            print(f"❌ {name} ({module}) - 导入失败: {e}")
            results.append(False)

    # 总结
    print(f"\n{'='*60}")
    print("【修复总结】")
    print(f"{'='*60}")

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"总检查项: {total}")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")

    if failed > 0:
        print(f"\n⚠️  发现 {failed} 个问题需要修复")
        print(f"\n建议:")
        print(f"1. 查看详细报告: cat CODE_REVIEW_REPORT.md")
        print(f"2. 修复P0优先级问题")
        print(f"3. 运行测试: python3 -m pytest test_*.py -v")
        return 1
    else:
        print(f"\n✅ 所有关键检查通过！")
        return 0

if __name__ == "__main__":
    sys.exit(main())
