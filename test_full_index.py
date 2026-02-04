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
    print("测试完整索引构建（处理所有图片）")
    print("=" * 60 + "\n")

    config = get_config()

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
        batch_size=10,
        max_retries=3,
        timeout=60,
    )

    print(f"Embedding维度: {embedding.dimension}")
    print(f"照片目录: {config.get('PHOTO_DIR')}")

    result = indexer.build_index()

    print("\n" + "=" * 60)
    print("索引构建结果")
    print("=" * 60)
    print(f"状态: {result['status']}")
    print(f"消息: {result['message']}")
    print(f"总图片数: {result['total_count']}")
    print(f"成功索引数: {result['indexed_count']}")
    print(f"失败数: {result['failed_count']}")
    print(f"降级占比: {result['fallback_ratio']}")
    print(f"耗时: {result['elapsed_time']}秒")

    if result['status'] == 'success':
        print("\n✅ 索引构建成功！")
    else:
        print(f"\n❌ 索引构建失败")

if __name__ == "__main__":
    main()
