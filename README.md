# 照片搜索引擎 (Photo Search Engine)

基于AI的个人照片搜索引擎Demo，支持通过自然语言查询检索本地照片。系统使用Vision LLM生成图片描述，通过向量相似度检索实现智能搜索，支持场景、人物、活动、时间、情感等多类查询。

## 🌟 核心特性

- **智能描述生成**：使用GPT-4o Vision自动生成图片的中文自然语言描述
- **向量化检索**：基于FAISS的高效向量相似度搜索
- **多维度查询**：支持场景、人物、活动、时间、情感等多种查询类型
- **时间检索**：基于EXIF元数据或文件时间，支持"去年"、"冬天"等自然语言时间词
- **成本优化**：使用Base64编码+图片压缩策略，大幅降低Token消耗
- **本地化部署**：所有数据存储在本地，隐私安全

## 🏗️ 技术架构

```
用户浏览器
    ↓
templates/index.html（前端界面）
    ↓
api/routes.py（HTTP API）
    ↓
core/indexer.py（索引构建）  core/searcher.py（检索引擎）
    ↓                        ↓
utils/vision_llm_service.py（Vision LLM - OpenRouter + GPT-4o）
utils/embedding_service.py（文本Embedding - T5模型）
utils/vector_store.py（FAISS向量存储）
utils/time_parser.py（时间语义解析）
utils/image_parser.py（图片解析与优化）
```

### 技术栈

| 层次 | 技术 | 用途 |
|------|------|------|
| Web框架 | Flask 3.0.3 | HTTP服务 |
| Vision LLM | GPT-4o (via OpenRouter) | 图片描述生成 |
| 时间解析 | GPT-3.5-turbo (via OpenRouter) | 时间语义理解 |
| 文本嵌入 | sentence-t5-base | 中文语义向量 |
| 向量存储 | FAISS-cpu | 高效相似度检索 |
| 图像处理 | Pillow 10.3.0 | 图片解析与优化 |
| 深度学习 | PyTorch 2.7.0 | sentence-transformers后端 |

## 📋 环境要求

- **Python**: 3.8+ (推荐 3.12.4)
- **操作系统**: Windows / Linux / macOS
- **GPU**: 可选（PyTorch自动检测CUDA）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
# OpenRouter API密钥（必需）
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 兼容性配置（可选，如果使用原始OpenAI密钥）
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenRouter API地址（通常无需修改）
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# 照片目录（必需，使用绝对路径）
PHOTO_DIR=C:/Users/YourName/Photos

# 数据存储目录（可选，默认 ./data）
DATA_DIR=./data

# 服务器配置（可选）
根据个人网络环境和部署需求灵活配置本地服务器参数，包括主机名、端口和密钥等细节。

# 图片处理配置（可选）
# 是否使用Base64编码方式（默认 true）
USE_BASE64=true

# 图片最大边长（像素，默认 1024）
IMAGE_MAX_SIZE=1024

# 图片压缩质量（1-100，默认 85）
IMAGE_QUALITY=85

# 图片输出格式（JPEG/WEBP/PNG，默认 WEBP）
IMAGE_FORMAT=WEBP
```

### 3. 获取OpenRouter API密钥

1. 访问 https://openrouter.ai/
2. 注册/登录账号
3. 在设置中创建API密钥
4. 将密钥复制到 `.env` 文件

### 4. 准备照片

将照片放入指定的照片目录（`PHOTO_DIR`）。支持以下格式：
- JPG / JPEG
- PNG
- WEBP

建议准备至少**100张照片**以满足验收标准。

### 5. 启动服务

```bash
python main.py
```

服务将在 http://localhost:5000 启动。

### 6. 访问界面

打开浏览器访问 http://localhost:5000

## 💡 使用方法

### 索引构建流程

1. **点击"初始化索引"按钮**
2. 系统开始扫描照片目录
3. 对每张图片调用Vision LLM生成描述
4. 使用T5模型生成文本向量
5. 构建FAISS向量索引
6. 完成后显示索引统计信息

### 查询示例

| 查询类型 | 示例查询 |
|----------|---------|
| 场景查询 | "山上的照片"、"城市夜景"、"海边" |
| 人物查询 | "有人物的照片"、"集体合影"、"朋友" |
| 活动查询 | "聚餐"、"运动"、"旅行" |
| 时间查询 | "去年的照片"、"冬天的照片"、"去年夏天" |
| 情感查询 | "欢乐的场景"、"温馨的时刻" |
| 复合查询 | "去年夏天在海边的照片"、"冬天和朋友聚会的照片" |

### 搜索结果

每个搜索结果显示：
- 照片缩略图
- AI生成的自然语言描述
- 相似度分数（0-100%）
- 排名位置

## 🔌 API接口说明

### GET /
渲染前端页面

### POST /init_index
触发索引构建

**请求体**：无

**响应**：
```json
{
  "status": "success | processing | failed",
  "message": "索引构建成功/进行中/失败",
  "total_count": 105,
  "indexed_count": 100,
  "failed_count": 5,
  "fallback_ratio": 0.05,
  "elapsed_time": 45.2
}
```

### POST /search_photos
执行照片搜索

**请求体**：
```json
{
  "query": "去年夏天在海边的照片",
  "top_k": 10
}
```

**响应**：
```json
{
  "status": "success",
  "results": [
    {
      "photo_path": "/photos/vacation_beach_001.jpg",
      "photo_url": "/photo?path=/photos/vacation_beach_001.jpg",
      "description": "一群人在阳光明媚的海滩上玩耍",
      "score": 0.95,
      "rank": 1
    }
  ],
  "total_results": 10,
  "elapsed_time": 0.15
}
```

### GET /index_status
获取索引构建和加载状态

**响应**：
```json
{
  "status": "idle | processing | ready | failed",
  "message": "索引已就绪",
  "total_count": 100,
  "indexed_count": 100,
  "failed_count": 0,
  "elapsed_time": 45.2
}
```

### GET /photo
返回图片文件（供前端和Vision LLM使用）

**参数**：
- `path`: 图片绝对路径

## 📁 项目结构

```
Photo_Search_Engine/
├── api/                    # API接口层
│   ├── __init__.py
│   └── routes.py          # HTTP路由定义
├── core/                   # 核心业务模块
│   ├── __init__.py
│   ├── indexer.py          # 索引构建器
│   └── searcher.py         # 检索引擎
├── templates/              # 前端模板
│   └── index.html          # 用户界面
├── tests/                  # 测试代码
│   ├── __init__.py
│   ├── test_indexer.py
│   ├── test_searcher.py
│   └── ...
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── embedding_service.py    # 文本嵌入服务
│   ├── image_parser.py          # 图片解析
│   ├── time_parser.py           # 时间解析
│   ├── vector_store.py          # 向量存储
│   └── vision_llm_service.py    # Vision LLM服务
├── config.py              # 配置管理
├── main.py                # 应用入口
├── requirements.txt       # 依赖清单
├── demo_development_doc.md  # 开发文档
└── .env                   # 环境变量（需创建）
```

## ⚙️ 配置说明

### Vision LLM配置

**默认模型**: `openai/gpt-4o`

**可用模型**：
- `openai/gpt-4o` - GPT-4 Omni（推荐）
- `openai/gpt-4-turbo` - 速度更快
- `anthropic/claude-3-sonnet` - 多模态能力
- `google/gemini-pro-vision` - Google视觉模型

**环境变量**：
```bash
VISION_MODEL_NAME=openai/gpt-4o
```

### Embedding配置

**默认模型**: `sentence-t5-base` (768维)

**可用模型**：
- `sentence-t5-base` - 768维，速度快（推荐）
- `sentence-t5-large` - 768维，效果更佳
- `sentence-t5-xxl` - 768维，效果最佳（需更多内存）
- `paraphrase-multilingual-MiniLM-L12-v2` - 384维，资源受限场景

**环境变量**：
```bash
EMBEDDING_MODEL_NAME=sentence-t5-base
EMBEDDING_DIMENSION=768
```

### 图片访问机制

**重要说明**: OpenRouter是云端服务，无法访问本地 `localhost:5000`。因此本系统采用 **Base64编码方式**。

**成本优化策略**：
1. **自动缩放**：图片自动缩放至 `max_size`（默认1024像素）
2. **格式压缩**：使用WEBP格式（默认），比JPEG更小
3. **智能选择**：小于 `max_size` 的图片不放大

## ✅ 验收标准

为确保项目质量，以下验收指标必须达成：

| 指标 | 要求 | 说明 |
|------|------|------|
| 索引数量 | ≥100张 | 成功索引的照片数量 |
| 查询类型覆盖 | 全部 | 场景/人物/活动/时间/情感 |
| 时间检索 | 正确 | 基于EXIF或文件时间 |
| 描述质量 | >90% | 降级描述占比<10% |

## 💰 成本预估

| 服务 | 单价 | 100张照片成本 |
|------|------|---------------|
| 图像描述（GPT-4o） | ~0.003美元/张 | ~0.3美元 |
| 时间解析（GPT-3.5） | ~0.0001美元/次 | ~0.01美元 |
| 文本嵌入（T5） | 免费（本地） | 0美元 |
| **总计** | | **~0.3美元（约2元）** |

## 🔍 图片优化说明

为降低Token消耗并提高传输效率，系统对图片进行以下优化：

1. **EXIF方向校正**：自动修正图片旋转
2. **智能缩放**：等比缩放至最大边长
3. **格式转换**：支持JPEG/WebP/PNG输出
4. **质量压缩**：可调节压缩质量（1-100）

**示例**：
```python
# 优化并编码图片
image_bytes = resize_and_optimize_image(
    photo_path,
    max_size=1024,      # 最大边长
    quality=85,          # 压缩质量
    format="WEBP"        # 输出格式
)
```

## 🐛 常见问题

### 1. 索引构建失败

**可能原因**：
- `PHOTO_DIR` 未设置或路径错误
- `OPENROUTER_API_KEY` 未设置或无效
- 照片目录中没有支持的图片格式
- API调用超时

**解决方法**：
- 检查 `.env` 文件配置
- 确认照片目录存在且包含图片
- 检查网络连接和API密钥有效性

### 2. 搜索无结果

**可能原因**：
- 索引未成功构建
- 查询内容与图片描述不匹配
- 时间约束过滤掉了所有结果

**解决方法**：
- 检查索引状态（访问 /index_status）
- 尝试更通用的查询词
- 移除时间约束进行搜索

### 3. API调用超时

**可能原因**：
- 网络连接不稳定
- API服务繁忙
- 图片过大

**解决方法**：
- 增加 `TIMEOUT` 环境变量值
- 减小 `IMAGE_MAX_SIZE` 值
- 增加 `MAX_RETRIES` 重试次数

## 📝 开发文档

详细的开发文档请参考 [demo_development_doc.md](demo_development_doc.md)

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**享受智能照片搜索体验！** 📷✨
