#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import get_config
from core.indexer import Indexer
from utils.embedding_service import DashscopeEmbeddingService
from utils.vision_llm_service import OpenRouterVisionLLMService
from utils.vector_store import VectorStore

def main():
    print("\n" + "=" * 60)
    print("测试索引构建（仅处理前3张图片）")
    print("=" * 60 + "\n")

    config = get_config()

    # 初始化服务
    print("1. 初始化服务...")

    embedding = DashscopeEmbeddingService(
        api_key=config.get("EMBEDDING_API_KEY"),
        model_name=config.get("EMBEDDING_MODEL", "text-embedding-v4"),
        base_url=config.get("EMBEDDING_BASE_URL"),
        timeout=30,
    )

    vector_store = VectorStore(
        dimension=embedding.dimension,
        index_path=config.get("INDEX_PATH"),
        metadata_path=config.get("METADATA_PATH"),
        metric=config.get("VECTOR_METRIC", "cosine"),
    )

    vision = OpenRouterVisionLLMService(
        api_key=config.get("OPENROUTER_API_KEY"),
        model_name=config.get("VISION_MODEL_NAME", "openai/gpt-4o"),
        base_url=config.get("OPENROUTER_BASE_URL"),
        timeout=60,
    )

    indexer = Indexer(
        photo_dir=config.get("PHOTO_DIR"),
        vision=vision,
        embedding=embedding,
        vector_store=vector_store,
        data_dir=config.get("DATA_DIR"),
        batch_size=3,
        max_retries=3,
        timeout=60,
    )

    print(f"   ✓ Embedding维度: {embedding.dimension}")
    print(f"   ✓ 照片目录: {config.get('PHOTO_DIR')}")

    # 扫描照片
    print("\n2. 扫描照片...")
    photos = indexer.scan_photos()
    print(f"   找到 {len(photos)} 张图片")

    if len(photos) == 0:
        print("   ✗ 没有找到有效图片")
        return

    # 仅处理前3张
    test_photos = photos[:3]
    print(f"   测试处理前3张图片:")
    for i, p in enumerate(test_photos, 1):
        print(f"     {i}. {os.path.basename(p)}")

    # 处理批
    print("\n3. 处理图片...")
    try:
        results = indexer.process_batch(test_photos)

        print(f"\n4. 处理结果:")
        for result in results:
            status = "✓ 成功" if result["status"] == "success" else "✗ 失败"
            print(f"   {status}: {os.path.basename(result['photo_path'])}")
            if result["status"] == "failed":
                print(f"      错误: {result.get('error', 'Unknown')}")

    except Exception as e:
        print(f"\n   ✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
