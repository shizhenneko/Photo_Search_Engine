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
from utils.embedding_service import DashscopeEmbeddingService, T5EmbeddingService
from utils.vision_llm_service import OpenRouterVisionLLMService
from utils.vector_store import VectorStore

def test_connection():
    print("=" * 60)
    print("【测试1】API连接测试")
    print("=" * 60)

    config = get_config()

    print("\n--- 测试 Vision LLM API (OpenRouter) ---")
    openrouter_key = config.get("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY 设置: {'是' if openrouter_key and openrouter_key != 'sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx' else '否'}")
    if openrouter_key and openrouter_key != 'sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx':
        try:
            vision = OpenRouterVisionLLMService(
                api_key=openrouter_key,
                model_name=config.get("VISION_MODEL_NAME", "openai/gpt-4o"),
                base_url=config.get("OPENROUTER_BASE_URL"),
                timeout=30,
            )
            print("Vision LLM服务初始化成功")
        except Exception as e:
            print(f"Vision LLM服务初始化失败: {e}")
    else:
        print("警告: OPENROUTER_API_KEY 未配置")

    # 测试 Embedding API
    print("\n--- 测试 Embedding API (阿里百炼) ---")
    embedding_key = config.get("EMBEDDING_API_KEY")
    print(f"EMBEDDING_API_KEY 设置: {'是' if embedding_key else '否'}")
    if embedding_key:
        try:
            embedding = DashscopeEmbeddingService(
                api_key=embedding_key,
                model_name=config.get("EMBEDDING_MODEL", "text-embedding-v4"),
                base_url=config.get("EMBEDDING_BASE_URL"),
                timeout=30,
            )
            print(f"Embedding服务初始化成功，向量维度: {embedding.dimension}")
        except Exception as e:
            print(f"Embedding服务初始化失败: {e}")

def test_photo_files():
    """测试照片文件"""
    print("\n" + "=" * 60)
    print("【测试2】照片文件检查")
    print("=" * 60)

    config = get_config()
    photo_dir = config.get("PHOTO_DIR")

    if not photo_dir:
        print("错误: PHOTO_DIR 未配置")
        return False

    print(f"照片目录: {photo_dir}")

    if not os.path.isdir(photo_dir):
        print(f"错误: 照片目录不存在: {photo_dir}")
        return False

    # 扫描图片文件
    from utils.image_parser import is_valid_image
    valid_photos = []
    invalid_files = []

    for root, _, files in os.walk(photo_dir):
        for name in files:
            path = os.path.abspath(os.path.join(root, name))
            if is_valid_image(path):
                valid_photos.append(path)
            else:
                invalid_files.append(path)

    print(f"\n有效图片数量: {len(valid_photos)}")
    print(f"无效文件数量: {len(invalid_files)}")

    if valid_photos:
        print("\n有效图片列表:")
        for i, photo in enumerate(valid_photos[:5], 1):
            print(f"  {i}. {os.path.basename(photo)}")
        if len(valid_photos) > 5:
            print(f"  ... 还有 {len(valid_photos) - 5} 张")

    if invalid_files and len(invalid_files) <= 5:
        print("\n无效文件列表:")
        for f in invalid_files:
            print(f"  - {os.path.basename(f)}")

    return len(valid_photos) > 0

def test_single_image():
    """测试单张图片处理"""
    print("\n" + "=" * 60)
    print("【测试3】单张图片处理测试")
    print("=" * 60)

    config = get_config()
    photo_dir = config.get("PHOTO_DIR")

    from utils.image_parser import is_valid_image

    # 找第一张有效图片
    first_photo = None
    for root, _, files in os.walk(photo_dir):
        for name in files:
            path = os.path.abspath(os.path.join(root, name))
            if is_valid_image(path):
                first_photo = path
                break
        if first_photo:
            break

    if not first_photo:
        print("错误: 没有找到有效图片")
        return

    print(f"测试图片: {first_photo}")

    try:
        # 初始化服务
        vision = OpenRouterVisionLLMService(
            api_key=config.get("OPENROUTER_API_KEY"),
            model_name=config.get("VISION_MODEL_NAME", "openai/gpt-4o"),
            base_url=config.get("OPENROUTER_BASE_URL"),
            timeout=60,  # 增加超时时间
        )
        embedding = DashscopeEmbeddingService(
            api_key=config.get("EMBEDDING_API_KEY"),
            model_name=config.get("EMBEDDING_MODEL", "text-embedding-v4"),
            base_url=config.get("EMBEDDING_BASE_URL"),
            timeout=30,
        )

        # 测试 Vision LLM
        print("\n--- 测试 Vision LLM ---")
        description = vision.generate_description(first_photo)
        print(f"✓ Vision LLM 成功")
        print(f"  描述: {description[:100]}...")

        # 测试 Embedding
        print("\n--- 测试 Embedding ---")
        vector = embedding.generate_embedding(description)
        print(f"✓ Embedding 成功")
        print(f"  向量维度: {len(vector)}")

        print("\n✅ 单张图片处理测试通过!")

    except Exception as e:
        print(f"\n❌ 单张图片处理失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "索引构建诊断工具" + " " * 23 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 测试1: API连接
    test_connection()

    # 测试2: 照片文件
    has_photos = test_photo_files()
    if not has_photos:
        print("\n❌ 没有有效图片，无法继续测试")
        return

    # 测试3: 单张图片处理
    test_single_image()

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
