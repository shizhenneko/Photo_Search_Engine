# 测试指南

## 测试类型

本项目包含两种类型的测试：

### 1. 单元测试（Unit Tests）
- **无需API密钥**
- 测试代码逻辑和功能
- 使用mock模拟外部依赖
- 快速、稳定、无成本

### 2. 集成测试（Integration Tests）
- **需要有效的 `OPENROUTER_API_KEY`**
- 调用真实的OpenRouter API
- 测试完整的系统行为
- 需要网络连接和API费用

## 快速开始

### 1. 单元测试（推荐）
```bash
# 运行所有单元测试（自动跳过需要API密钥的测试）
python run_tests.py

# 或使用pytest
pytest tests/ -v

# 或使用unittest
python -m unittest discover tests -v
```

### 2. 集成测试（需要API密钥）
```bash
# 确保在.env中设置了API密钥
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env

# 运行所有测试（包括集成测试）
python run_tests.py --integration

# 或直接使用pytest
OPENROUTER_API_KEY=sk-or-v1-... pytest tests/ -v
```

## 运行方式

### 使用便捷脚本（推荐）
```bash
# 单元测试
python run_tests.py

# 集成测试
python run_tests.py --integration

# 详细输出
python run_tests.py --verbose

# 生成覆盖率报告
python run_tests.py --coverage
```

### 使用 pytest
```bash
# 基础运行
pytest tests/

# 详细输出
pytest tests/ -v

# 单个测试文件
pytest tests/test_embedding_service.py -v

# 单个测试方法
pytest tests/test_embedding_service.py::EmbeddingServiceTests::test_config_loading -v

# 生成覆盖率
pytest tests/ --cov=. --cov-report=html
```

### 使用 unittest
```bash
# 基础运行
python -m unittest discover tests

# 详细输出
python -m unittest discover tests -v

# 单个测试文件
python -m unittest tests.test_embedding_service

# 单个测试方法
python -m unittest tests.test_embedding_service.EmbeddingServiceTests.test_config_loading
```

## 测试文件说明

| 文件 | 测试内容 | 集成测试 |
|-------|-----------|-----------|
| `test_embedding_service.py` | T5/OpenAI嵌入服务 | ✅ OpenAI |
| `test_image_parser.py` | 图片解析、EXIF、降级描述 | ❌ |
| `test_vector_store.py` | FAISS向量存储 | ❌ |
| `test_vision_llm_service.py` | Vision服务生成描述 | ✅ OpenRouter |
| `test_time_parser.py` | LLM时间解析 | ✅ OpenRouter |
| `test_searcher.py` | 搜索查询、时间过滤 | ❌ |
| `test_indexer.py` | 索引构建、验收检查 | ❌（使用LocalVision）|

## 测试覆盖范围

根据 `demo_development_doc.md` 验收要求：

### utils/ 模块
- ✅ `embedding_service.py` - T5/OpenAI嵌入生成
- ✅ `image_parser.py` - 图片验证、EXIF解析、降级描述
- ✅ `vector_store.py` - FAISS存储、搜索、持久化
- ✅ `vision_llm_service.py` - Vision服务（本地+OpenRouter）
- ✅ `time_parser.py` - LLM时间解析

### core/ 模块
- ✅ `indexer.py` - 索引构建、批处理、验收门槛（100张）
- ✅ `searcher.py` - 搜索、查询验证、时间过滤

## 环境要求

### 单元测试
```bash
pip install pytest pytest-cov torch sentence-transformers pillow piexif
```

### 集成测试
```bash
pip install pytest pytest-cov torch sentence-transformers pillow piexif openai
export OPENROUTER_API_KEY=sk-or-v1-...
...
```

### 集成测试
```bash
pip install pytest pytest-cov sentence-transformers pillow piexif openai
export OPENROUTER_API_KEY=sk-or-v1-...
```

## 常见问题

### Q: 测试被跳过？
**A**: 需要 `OPENROUTER_API_KEY` 环境变量
```bash
export OPENROUTER_API_KEY=sk-or-v1-xxx
python run_tests.py --integration
```

### Q: `sentence-transformers` 模型下载慢？
**A**: 首次使用会自动下载模型（~1-2GB），可手动设置缓存：
```bash
export TRANSFORMERS_CACHE=/path/to/cache
```

### Q: 集成测试消耗API费用？
**A**: 是的，每次调用会产生费用。建议在开发中使用单元测试，在验收前运行集成测试。

### Q: 如何只运行特定测试？
**A**: 使用pytest的选择器：
```bash
pytest tests/test_image_parser.py -v
pytest tests/test_image_parser.py::ImageParserTests::test_is_valid_image_true -v
```

## CI/CD 集成

GitHub Actions 示例：
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt torch pytest pytest-cov
      - name: Run unit tests
        run: pytest tests/ -v
```
