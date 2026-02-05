#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
季节搜索功能测试脚本

用于验证"夏天的照片"等纯过滤查询是否正确执行。
"""

import sys
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_config
from main import initialize_services


def test_pure_filter_queries():
    """测试纯过滤查询场景"""
    print("=" * 60)
    print("开始测试季节过滤查询功能")
    print("=" * 60)
    
    # 初始化服务
    print("\n[1/4] 加载配置...")
    config = get_config()
    
    print("[2/4] 初始化服务...")
    indexer, searcher, rerank_service = initialize_services(config)
    
    print("[3/4] 加载索引...")
    if not searcher.load_index():
        print("❌ 错误：索引加载失败，请先运行 POST /init_index 构建索引")
        return False
    
    print(f"✓ 索引加载成功，共 {searcher.vector_store.get_total_items()} 张照片")
    
    # 测试用例
    test_cases = [
        {
            "name": "纯季节过滤",
            "query": "夏天的照片",
            "expected_behavior": "应该直接使用 ES 过滤，不生成 embedding",
        },
        {
            "name": "季节+场景混合",
            "query": "夏天海边的照片",
            "expected_behavior": "应该先过滤季节，再向量匹配'海边'",
        },
        {
            "name": "纯年份过滤",
            "query": "2023年的照片",
            "expected_behavior": "应该直接使用 ES 过滤，不生成 embedding",
        },
        {
            "name": "纯时段过滤",
            "query": "傍晚的照片",
            "expected_behavior": "应该直接使用 ES 过滤，不生成 embedding",
        },
    ]
    
    print("\n[4/4] 执行测试用例...")
    print("=" * 60)
    
    all_passed = True
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}/{len(test_cases)}: {case['name']}")
        print(f"查询: {case['query']}")
        print(f"预期: {case['expected_behavior']}")
        print("-" * 60)
        
        try:
            # 执行搜索
            results = searcher.search(case['query'], top_k=10)
            
            print(f"✓ 搜索完成")
            print(f"  返回结果数: {len(results)}")
            
            if results:
                print(f"  前3个结果:")
                for j, result in enumerate(results[:3], 1):
                    path = result.get('photo_path', '').split('\\')[-1]
                    score = result.get('score', 0)
                    print(f"    {j}. {path} (score: {score:.4f})")
            else:
                print(f"  ⚠️  警告：没有找到匹配的照片")
                if case['name'] == "纯季节过滤":
                    print(f"  ⚠️  可能原因：")
                    print(f"     1. ES 服务未运行或数据未同步")
                    print(f"     2. 元数据中没有该季节的照片")
                    all_passed = False
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试完成（注意检查控制台是否有 'embedding' 相关日志）")
        print("\n预期行为：")
        print("  - 纯过滤查询（如'夏天的照片'）应该打印: [DEBUG] 纯过滤查询模式")
        print("  - 不应该看到: 'generate_embedding' 或 '26个字符' 相关日志")
    else:
        print("⚠️  部分测试存在问题，请检查上面的输出")
    print("=" * 60)
    
    return all_passed


def check_metadata_seasons():
    """检查元数据中的季节分布"""
    print("\n" + "=" * 60)
    print("检查元数据中的季节信息")
    print("=" * 60)
    
    import json
    metadata_path = Path(__file__).parent / "data" / "metadata.json"
    
    if not metadata_path.exists():
        print("❌ metadata.json 不存在")
        return
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 统计季节分布
    season_count = {"春天": 0, "夏天": 0, "秋天": 0, "冬天": 0, "无": 0}
    
    for item in metadata:
        time_info = item.get("time_info", {})
        season = time_info.get("season")
        if season in season_count:
            season_count[season] += 1
        else:
            season_count["无"] += 1
    
    print(f"\n总照片数: {len(metadata)}")
    print(f"季节分布:")
    for season, count in season_count.items():
        percentage = count / len(metadata) * 100 if len(metadata) > 0 else 0
        print(f"  {season}: {count} 张 ({percentage:.1f}%)")
    
    if season_count["夏天"] > 0:
        print(f"\n✓ 确认有 {season_count['夏天']} 张夏天的照片")
    else:
        print(f"\n⚠️  警告：没有找到标记为'夏天'的照片")
    
    print("=" * 60)


if __name__ == "__main__":
    # 检查元数据
    check_metadata_seasons()
    
    # 执行测试
    print("\n")
    success = test_pure_filter_queries()
    
    # 退出码
    sys.exit(0 if success else 1)
