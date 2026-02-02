#!/usr/bin/env python
"""
测试运行脚本 - 支持单元测试和集成测试

测试环境要求：
- torch: GPU版本使用（如果可用），否则CPU
- faiss: CPU版本（faiss-cpu）
- 测试方案：使用Base64编码方式

用法：
    python run_tests.py              # 运行所有测试（跳过需要API密钥的集成测试）
    python run_tests.py --integration   # 运行包括集成测试在内的所有测试
    python run_tests.py --verbose      # 详细输出模式
    python run_tests.py --coverage     # 生成覆盖率报告
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_api_key():
    """检查是否设置了API密钥"""
    return bool(os.getenv("OPENROUTER_API_KEY"))


def check_torch_cuda():
    """检查torch是否支持CUDA"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ PyTorch CUDA可用: {torch.version.cuda}")
            print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ PyTorch CUDA不可用，将使用CPU模式")
        return cuda_available
    except ImportError:
        print("✗ PyTorch未安装")
        return False


def check_faiss_cpu():
    """检查faiss-cpu是否正确安装"""
    try:
        import faiss
        print(f"✓ FAISS版本: {faiss.__version__}")
        print("  注意：确保使用faiss-cpu（符合设计要求）")
        return True
    except ImportError:
        print("✗ FAISS未安装")
        return False


def print_environment_info():
    """打印测试环境信息"""
    print("=" * 60)
    print("测试环境检查")
    print("=" * 60)
    check_torch_cuda()
    check_faiss_cpu()
    print("=" * 60)
    print()


def run_unittest(verbose=False):
    """使用unittest运行测试"""
    args = ["-m", "unittest", "discover", "tests"]
    if verbose:
        args.append("-v")
    return subprocess.run([sys.executable, *args])


def run_pytest(verbose=False, coverage=False):
    """使用pytest运行测试"""
    args = ["-m", "pytest", "tests/"]
    if verbose:
        args.append("-v")
    if coverage:
        args.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    return subprocess.run([sys.executable, *args])


def main():
    parser = argparse.ArgumentParser(description="运行照片搜索引擎测试")
    parser.add_argument(
        "--integration",
        action="store_true",
        help="运行需要API密钥的集成测试"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出模式"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="生成代码覆盖率报告"
    )
    parser.add_argument(
        "--runner",
        choices=["unittest", "pytest"],
        default="pytest",
        help="测试运行器（默认：pytest）"
    )
    parser.add_argument(
        "--no-env-check",
        action="store_true",
        help="跳过环境检查"
    )

    args = parser.parse_args()

    print_environment_info()

    if args.integration and not check_api_key():
        print("错误：要求运行集成测试，但未设置OPENROUTER_API_KEY")
        print("\n请在.env文件中设置：")
        print("  OPENROUTER_API_KEY=your_api_key_here")
        print("\n或者使用环境变量：")
        print("  export OPENROUTER_API_KEY=your_api_key_here")
        sys.exit(1)

    if args.integration:
        print(f"运行模式：集成测试（需要API密钥）")
        print("测试方案：使用Base64编码方式")
    else:
        print(f"运行模式：单元测试（跳过需要API密钥的测试）")
        print("提示：使用 --integration 运行完整集成测试")
        print("测试方案：使用Base64编码方式")

    print()

    if args.runner == "pytest":
        result = run_pytest(verbose=args.verbose, coverage=args.coverage)
    else:
        result = run_unittest(verbose=args.verbose)

    if result.returncode == 0:
        print("\n所有测试通过！")
    else:
        print(f"\n测试失败，退出码：{result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
