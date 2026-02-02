# 照片搜索引擎开发计划

> 本文档用于vibe coding的逐步实现指南
>
> **模型配置**：
> - Vision LLM：OpenRouter + GPT-4 (openai/gpt-4o)
> - Embedding：本地T5模型 (sentence-t5-base, 768维)
> - Time Parser：OpenRouter + GPT-3.5-turbo
>
> **成本预估**：100张照片约2元人民币

---

## 开发阶段概览

### Phase 0: 项目初始化（按Demo开发文档）
- [ ] 创建项目文件夹结构（core/utils/api/templates/data）
- [ ] 创建Python包初始化文件（core/utils/api/__init__.py）
- [ ] 明确本地HTTP图片访问方案（/photo?path=... 用于前端与Vision LLM）

### Phase 1: 配置和基础设施（按Demo开发文档）
- [ ] 1.1 config.py - 配置管理（OpenRouter + 本地T5）
- [ ] 1.2 requirements.txt - 依赖声明（包含sentence-transformers）
- [ ] 1.3 .env.example - 环境变量模板（OpenRouter + PHOTO_DIR）

### Phase 2: 工具模块（底层到高层）
- [ ] 2.1 utils/image_parser.py - 图片解析（EXIF、元数据、格式验证）
- [ ] 2.2 utils/vector_store.py - FAISS向量存储封装
- [ ] 2.3 utils/time_parser.py - 时间解析（使用LLM）
- [ ] 2.4 utils/vision_llm_service.py - Vision服务抽象和实现
- [ ] 2.5 utils/embedding_service.py - Embedding服务抽象和实现

### Phase 3: 核心业务
- [ ] 3.1 core/indexer.py - 索引构建器
- [ ] 3.2 core/searcher.py - 搜索引擎

### Phase 4: 接口层
- [ ] 4.1 api/routes.py - HTTP API路由
- [ ] 4.2 templates/index.html - 前端界面

### Phase 5: 应用入口
- [ ] 5.1 main.py - 应用入口

---

## 详细开发步骤

### Phase 0: 项目初始化（按Demo开发文档）

#### 创建文件夹结构
```bash
mkdir -p core utils api templates data
```

#### 创建__init__.py文件
- `core/__init__.py`
- `utils/__init__.py`
- `api/__init__.py`

#### 约定本地HTTP图片访问机制
**目的**：为Vision LLM与前端提供统一访问图片的URL，避免Base64 token消耗。  
**URL格式**：`http://localhost:5000/photo?path=<图片绝对路径>`  
**注意**：路径需URL编码，且仅用于本地Demo。

---

### Phase 1: 配置和基础设施（按Demo开发文档）

#### 1.1 config.py

**文件位置**: `config.py`（项目根目录）

**核心功能**：
- 从环境变量读取配置并提供默认值
- OpenRouter + 本地T5混合模型配置
- 配置项包括：
  - PHOTO_DIR: 照片目录路径（绝对路径）
  - DATA_DIR: 数据存储目录（默认./data）
  - OPENROUTER_API_KEY: OpenRouter API密钥（Vision + 时间解析）
  - OPENAI_API_KEY: 兼容性配置（同OpenRouter密钥）
  - OPENROUTER_BASE_URL: OpenRouter API地址（默认https://openrouter.ai/api/v1）
  - VISION_MODEL_NAME: Vision模型（默认openai/gpt-4o）
  - EMBEDDING_MODEL_NAME: Embedding模型（默认sentence-t5-base）
  - EMBEDDING_DIMENSION: 向量维度（默认768）
  - TIME_PARSE_MODEL_NAME: 时间解析模型（默认openai/gpt-3.5-turbo）
  - SERVER_HOST: 服务器主机（默认localhost）
  - SERVER_PORT: 服务器端口（默认5000）
  - BATCH_SIZE: 批处理大小（默认10）
  - MAX_RETRIES: 最大重试次数（默认3）
  - TIMEOUT: 超时时间（默认30秒）
  - TOP_K: 默认返回结果数量（默认10）
  - INDEX_PATH: 索引路径（默认./data/photo_search.index）
  - METADATA_PATH: 元数据路径（默认./data/metadata.json）

**模型配置说明**：
- **Vision LLM（图像描述）**：OpenRouter接入GPT-4
  - 模型：openai/gpt-4o（GPT-4 Omni，支持视觉）
  - 备选：openai/gpt-4-turbo（速度更快）
  - 优势：高质量图像理解、支持中文、按需付费
- **Embedding（文本嵌入）**：本地T5模型
  - 模型：sentence-t5-base（768维，中文支持好）
  - 备选：sentence-t5-large（768维，效果更佳）
  - 优势：本地运行、无API成本、开源免费
- **Time Parser（时间解析）**：OpenRouter + GPT-3.5
  - 模型：openai/gpt-3.5-turbo
  - 优势：语义理解强、支持复杂中文时间表达

**关键函数**：
- `load_config()` -> dict: 加载所有配置（含默认值与路径拼接）
- `get_config()` -> dict: 获取配置单例

---

#### 1.2 requirements.txt

**内容**：
```
flask==3.0.3
numpy==1.26.4
pillow==10.3.0
faiss-cpu
openai>=1.0.0
sentence-transformers>=2.2.0
python-dotenv
piexif>=1.1.3
torch>=2.0.0
```

---

#### 1.3 .env.example

**内容**：
```
# 照片目录（绝对路径）
PHOTO_DIR=C:/Users/YourName/Photos

# OpenRouter API密钥（Vision + 时间解析）
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 兼容性配置（如果只使用OPENAI_API_KEY）
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenRouter API地址
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Vision模型名称
VISION_MODEL_NAME=openai/gpt-4o

# Embedding模型名称
EMBEDDING_MODEL_NAME=sentence-t5-base

# Embedding向量维度
EMBEDDING_DIMENSION=768

# 时间解析模型
TIME_PARSE_MODEL_NAME=openai/gpt-3.5-turbo

# 服务器配置
SERVER_HOST=localhost
SERVER_PORT=5000

# 处理参数
BATCH_SIZE=10
MAX_RETRIES=3
TIMEOUT=30
TOP_K=10
```

---

### Phase 2: 工具模块

#### 2.1 utils/image_parser.py

**文件位置**: `utils/image_parser.py`

**核心功能**：
- 图片格式验证
- EXIF元数据解析
- 文件时间获取
- 图片尺寸获取
- 降级描述生成（基于文件名）

**关键函数**：
```python
def is_valid_image(file_path: str) -> bool
    """验证文件是否为支持的图片格式"""

def extract_exif_metadata(file_path: str) -> Dict[str, Any]
    """解析EXIF元数据（拍摄时间、GPS、相机型号）"""

def get_file_time(file_path: str) -> str
    """获取文件时间（ISO 8601格式）"""

def get_image_dimensions(file_path: str) -> Tuple[int, int]
    """获取图片宽高"""

def generate_fallback_description(file_path: str) -> str
    """基于文件名生成降级描述"""
```

**依赖**：Pillow (PIL), piexif, os, re

---

#### 2.2 utils/vector_store.py

**文件位置**: `utils/vector_store.py`

**核心功能**：
- FAISS索引初始化
- 向量添加
- 相似度搜索
- 索引持久化（save/load）

**关键类**：
```python
class VectorStore:
    def __init__(self, dimension: int, index_path: str, metadata_path: str)
    def add_item(self, embedding: List[float], metadata: Dict) -> None
    def search(self, query_embedding: List[float], top_k: int) -> List[Dict]
    def save(self) -> None
    def load(self) -> bool
    def get_total_items(self) -> int
```

**依赖**：faiss, json, os

---

#### 2.3 utils/time_parser.py

**文件位置**: `utils/time_parser.py`

**核心功能**：
- 使用OpenRouter + LLM解析查询中的时间约束
- 支持"去年"、"冬天"、"上个月"等中文时间词
- 返回结构化时间范围（start_date, end_date, precision）

**关键类**：
```python
class TimeParser:
    def __init__(self, api_key: str,
                 model_name: str = "openai/gpt-3.5-turbo",
                 base_url: str = "https://openrouter.ai/api/v1")
        """
        初始化时间解析器（使用OpenRouter）
        api_key: OpenRouter API密钥
        model_name: 模型名称，默认openai/gpt-3.5-turbo
        base_url: OpenRouter API地址
        """
    def extract_time_constraints(self, query: str) -> Dict[str, Any]
        """
        返回: {
            "start_date": str | None,  # ISO 8601
            "end_date": str | None,    # ISO 8601
            "precision": str           # "year" | "month" | "season" | "none"
        }
        """
    def _infer_precision(self, start_date: str | None, end_date: str | None) -> str
```

**依赖**：openai, datetime, json

**通过OpenRouter的优势**：
- 统一API管理，使用与Vision LLM相同的密钥
- GPT-3.5-turbo语义理解能力强
- 成本极低（约0.0001-0.0002美元/次）

**Prompt设计**（参考开发文档）：
```
当前日期：{current_date}（格式：YYYY-MM-DD）

用户查询：{query}

请分析用户查询中的时间约束，返回JSON格式：
{
  "has_time_constraint": true/false,
  "start_date": "YYYY-MM-DD" 或 null,
  "end_date": "YYYY-MM-DD" 或 null,
  "reasoning": "简要说明解析逻辑"
}

规则：
1. 如果没有明确的时间词，has_time_constraint=false，其他字段为null
2. 相对时间基于当前日期计算
3. 季节定义：春(3-5月)、夏(6-8月)、秋(9-11月)、冬(12-2月)
```

---

#### 2.4 utils/vision_llm_service.py

**文件位置**: `utils/vision_llm_service.py`

**核心功能**：
- 定义Vision LLM服务抽象接口
- 实现OpenRouter + GPT-4 Vision后端
- 通过本地HTTP URL访问图片（避免Base64 token消耗）

**关键类**：
```python
class VisionLLMService(ABC):
    @abstractmethod
    def generate_description(self, image_path: str) -> str
        """生成图片描述（中文，50-200字）"""

    @abstractmethod
    def generate_description_batch(self, image_paths: List[str]) -> List[str]
        """批量生成描述"""

class OpenRouterVisionLLMService(VisionLLMService):
    def __init__(self, api_key: str,
                 model_name: str = "openai/gpt-4o",
                 base_url: str = "https://openrouter.ai/api/v1",
                 server_host: str = "localhost",
                 server_port: int = 5000)
        """
        初始化OpenRouter Vision服务
        api_key: OpenRouter API密钥
        model_name: 模型名称，默认openai/gpt-4o
        base_url: OpenRouter API地址
        """
    def generate_description(self, image_path: str) -> str
    def _get_image_url(self, image_path: str) -> str
        """生成本地HTTP图片URL"""
```

**依赖**：openai, abc, urllib.parse

**模型说明**：
- openai/gpt-4o：GPT-4 Omni，支持视觉，性价比高
- openai/gpt-4-turbo：速度更快，成本略低
- anthropic/claude-3-sonnet：多模态能力，视觉质量好

**通过OpenRouter的优势**：
- 统一API接口，方便切换模型
- 按需付费，成本透明
- 支持URL方式访问图片，避免Base64编码的token浪费

**Prompt设计**：
```
请用中文描述这张图片，包含以下要素：
1. 场景描述（室内/室外，具体地点）
2. 主要主体（人物、物体、动物等）
3. 动作或状态（在做什么）
4. 环境细节（光线、天气、背景）
5. 情绪氛围（欢乐、温馨、宁静等）

描述长度：50-200字
```

---

#### 2.5 utils/embedding_service.py

**文件位置**: `utils/embedding_service.py`

**核心功能**：
- 定义Embedding服务抽象接口
- 实现本地T5模型后端（sentence-transformers）

**关键类**：
```python
class EmbeddingService(ABC):
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]
        """生成文本嵌入向量"""

    @abstractmethod
    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]
        """批量生成嵌入向量"""

class T5EmbeddingService(EmbeddingService):
    def __init__(self, model_name: str = "sentence-t5-base", device: str = None)
        """
        初始化T5嵌入服务
        model_name: 模型名称，默认sentence-t5-base
        device: 运行设备，None表示自动检测（Cuda优先）
        """
    def generate_embedding(self, text: str) -> List[float]
        """返回768维向量"""
    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]
        """批量生成嵌入向量，提高效率"""
```

**依赖**：sentence-transformers, abc, torch（自动安装）

**模型说明**：
- sentence-t5-base：768维，速度较快，中文支持好
- sentence-t5-large：768维，效果更佳，速度稍慢
- 首次使用会自动从Hugging Face下载模型（~1-2GB）

---

### Phase 3: 核心业务

#### 3.1 core/indexer.py

**文件位置**: `core/indexer.py`

**核心功能**：
- 扫描照片目录
- 批量生成描述和嵌入
- 构建FAISS索引
- 验收门槛检查（>=100张照片，降级描述<10%）

**关键类**：
```python
class Indexer:
    def __init__(self, photo_dir: str, vision: VisionLLMService,
                 embedding: EmbeddingService, vector_store: VectorStore,
                 data_dir: str = "./data")
    def scan_photos(self) -> List[str]
        """扫描照片目录，返回有效图片路径列表"""
    def process_batch(self, photo_paths: List[str]) -> List[Dict[str, Any]]
        """批量处理照片（描述+嵌入+元数据）"""
    def generate_description(self, photo_path: str) -> str
        """调用Vision LLM生成描述（带重试和降级策略）"""
    def build_index(self) -> Dict[str, Any]
        """
        主流程：
        1. 扫描照片
        2. 批量处理（描述+嵌入）
        3. 存储到FAISS
        4. 验收检查（>=100张，降级<10%）
        5. 保存索引和marker文件
        返回: {status, total_count, success_count, failed_count, fallback_ratio, ...}
        """
    def get_status(self) -> Dict[str, Any]
        """获取索引构建状态"""
```

**依赖**：
- config
- utils/image_parser
- utils/vision_llm_service
- utils/embedding_service
- utils/vector_store

**状态文件**：
- `data/index_ready.marker`: 索引完成标记
- `data/indexing.lock`: 索引进行中锁
- `data/index_status.status`: 状态信息

---

#### 3.2 core/searcher.py

**文件位置**: `core/searcher.py`

**核心功能**：
- 加载FAISS索引
- 查询向量化
- 相似度检索
- 时间过滤
- 结果排序

**关键类**：
```python
class Searcher:
    def __init__(self, embedding: EmbeddingService, time_parser: TimeParser,
                 vector_store: VectorStore, data_dir: str = "./data")
    def load_index(self) -> bool
        """加载FAISS索引和元数据"""
    def validate_query(self, query: str) -> bool
        """验证查询有效性"""
    def _extract_time_constraints(self, query: str) -> Dict[str, Any]
        """解析时间约束（调用TimeParser）"""
    def _filter_by_time(self, results: List[Dict], constraints: Dict) -> List[Dict]
        """按时间过滤结果（EXIF优先，文件时间兜底）"""
    def _distance_to_score(self, distance: float) -> float
        """L2距离转相似度（0-1范围）"""
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]
        """
        主流程：
        1. 验证查询
        2. 解析时间约束
        3. 查询向量化
        4. FAISS Top-K检索
        5. 时间过滤
        6. 相似度排序
        返回: [{photo_path, description, score, rank}, ...]
        """
    def get_index_stats(self) -> Dict[str, Any]
        """获取索引统计"""
```

**依赖**：
- config
- utils/embedding_service
- utils/time_parser
- utils/vector_store

---

### Phase 4: 接口层

#### 4.1 api/routes.py

**文件位置**: `api/routes.py`

**核心功能**：
- 注册Flask路由
- 连接核心业务逻辑
- 处理HTTP请求/响应

**关键函数**：
```python
def register_routes(app: Flask, indexer: Indexer, searcher: Searcher, config: Dict):
    """注册所有API路由"""

路由清单：

@app.route('/')
def index():
    """渲染前端页面"""
    return render_template('index.html')

@app.route('/init_index', methods=['POST'])
def init_index():
    """触发索引构建"""
    result = indexer.build_index()
    return jsonify(result)

@app.route('/search_photos', methods=['POST'])
def search_photos():
    """执行照片搜索"""
    data = request.get_json()
    query = data.get('query', '')
    top_k = min(data.get('top_k', 10), 50)
    results = searcher.search(query, top_k)
    return jsonify({"status": "success", "results": results, ...})

@app.route('/index_status', methods=['GET'])
def index_status():
    """获取索引状态"""
    return jsonify(indexer.get_status())

@app.route('/photo')
def get_photo():
    """返回图片文件（供前端和Vision LLM使用）"""
    path = request.args.get('path')
    # 路径校验和安全检查
    return send_file(path)
```

**依赖**：
- flask
- core/indexer
- core/searcher
- config

**响应格式**：

init_index响应：
```json
{
  "status": "success" | "processing" | "failed",
  "message": "索引构建完成",
  "total_count": 105,
  "success_count": 100,
  "failed_count": 5,
  "fallback_ratio": 0.04,
  "elapsed_time": 120.5
}
```

search_photos响应：
```json
{
  "status": "success",
  "results": [
    {
      "photo_path": "/path/to/photo.jpg",
      "photo_url": "/photo?path=/path/to/photo.jpg",
      "description": "描述文本",
      "score": 0.95,
      "rank": 1
    }
  ],
  "total_results": 10,
  "elapsed_time": 0.15
}
```

index_status响应：
```json
{
  "status": "ready",
  "message": "索引就绪，共100张照片",
  "total_count": 100,
  "indexed_count": 100,
  "failed_count": 0,
  "elapsed_time": 120.5
}
```

---

#### 4.2 templates/index.html

**文件位置**: `templates/index.html`

**核心功能**：
- 前端搜索界面
- 索引初始化按钮
- 查询输入框
- 结果网格展示

**页面结构**：
```html
<!DOCTYPE html>
<html>
<head>
  <title>照片搜索引擎</title>
  <style>
    /* CSS Grid布局，响应式卡片网格 */
  </style>
</head>
<body>
  <div class="container">
    <h1>照片搜索引擎</h1>

    <!-- 索引区域 -->
    <div class="index-section">
      <button onclick="initIndex()">初始化索引</button>
      <div id="index-status">状态：未初始化</div>
    </div>

    <!-- 搜索区域 -->
    <div class="search-section">
      <input type="text" id="query-input" placeholder="输入查询（如：去年夏天在海边的照片）">
      <button onclick="searchPhotos()">搜索</button>
    </div>

    <!-- 结果区域 -->
    <div id="results-grid" class="results-grid">
      <!-- 结果卡片动态插入 -->
    </div>
  </div>

  <script>
    function initIndex() {
      // POST /init_index
      // 启动轮询 checkIndexStatus()
    }

    function checkIndexStatus() {
      // 每2秒轮询 /index_status
      // 更新状态显示
    }

    function searchPhotos() {
      // POST /search_photos
      // 渲染结果卡片
    }
  </script>
</body>
</html>
```

**交互逻辑**：
1. 点击"初始化索引" -> POST /init_index
2. 启动轮询 -> GET /index_status（每2秒）
3. 索引完成后，搜索区域启用
4. 输入查询 -> POST /search_photos
5. 渲染结果网格（图片+描述+相似度）

---

### Phase 5: 应用入口

#### 5.1 main.py

**文件位置**: `main.py`

**核心功能**：
- 加载配置
- 初始化所有服务
- 创建Flask应用
- 启动服务器

**关键函数**：
```python
def load_config() -> dict:
    """加载配置（从config.py）"""
    return get_config()

def initialize_services(config: dict) -> tuple:
    """
    初始化服务实例

    顺序：
    1. VectorStore（维度768，sentence-t5-base）
    2. VisionLLMService（OpenRouter + GPT-4o）
    3. EmbeddingService（本地T5）
    4. TimeParser（OpenRouter + GPT-3.5-turbo）
    5. Indexer
    6. Searcher

    返回: (indexer, searcher)
    """
    # VectorStore初始化（768维，sentence-t5-base）
    vector_store = VectorStore(
        dimension=config['EMBEDDING_DIMENSION'],  # 768
        index_path=os.path.join(config['DATA_DIR'], 'photo_search.index'),
        metadata_path=os.path.join(config['DATA_DIR'], 'metadata.json')
    )

    # Vision服务（OpenRouter + GPT-4o）
    vision_service = OpenRouterVisionLLMService(
        api_key=config['OPENROUTER_API_KEY'],
        model_name=config['VISION_MODEL_NAME'],  # openai/gpt-4o
        base_url=config['OPENROUTER_BASE_URL'],
        server_host=config['SERVER_HOST'],
        server_port=config['SERVER_PORT']
    )

    # Embedding服务（本地T5）
    embedding_service = T5EmbeddingService(
        model_name=config['EMBEDDING_MODEL_NAME']  # sentence-t5-base
    )

    # TimeParser（OpenRouter + GPT-3.5-turbo）
    time_parser = TimeParser(
        api_key=config['OPENROUTER_API_KEY'],
        model_name=config['TIME_PARSE_MODEL_NAME'],  # openai/gpt-3.5-turbo
        base_url=config['OPENROUTER_BASE_URL']
    )

    # Indexer
    indexer = Indexer(
        photo_dir=config['PHOTO_DIR'],
        vision=vision_service,
        embedding=embedding_service,
        vector_store=vector_store,
        data_dir=config['DATA_DIR']
    )

    # Searcher
    searcher = Searcher(
        embedding=embedding_service,
        time_parser=time_parser,
        vector_store=vector_store,
        data_dir=config['DATA_DIR']
    )

    return indexer, searcher

def create_app(indexer: Indexer, searcher: Searcher, config: dict) -> Flask:
    """创建并配置Flask应用"""
    app = Flask(__name__)
    app.secret_key = config.get('SECRET_KEY', 'dev-secret-key')

    # 注册路由
    register_routes(app, indexer, searcher, config)

    # 错误处理
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"status": "error", "message": "接口不存在"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"status": "error", "message": "服务器内部错误"}), 500

    return app

def main():
    """主入口"""
    # 加载配置
    config = load_config()

    # 初始化服务
    indexer, searcher = initialize_services(config)

    # 创建Flask应用
    app = create_app(indexer, searcher, config)

    # 启动服务器
    app.run(
        host=config['SERVER_HOST'],
        port=config['SERVER_PORT'],
        debug=False
    )

if __name__ == '__main__':
    main()
```

**依赖**：
- config
- utils/*
- core/*
- api/routes

---

## 开发注意事项

### 模型配置（已确定）
- `VISION_MODEL_NAME`: 使用"openai/gpt-4o"（OpenRouter + GPT-4 Omni）
- `EMBEDDING_MODEL_NAME`: 使用"sentence-t5-base"（本地T5，768维）
- `EMBEDDING_DIMENSION`: 固定为768（sentence-t5-base的维度）
- `TIME_PARSE_MODEL_NAME`: 使用"openai/g/gpt-3.5-turbo"（OpenRouter + GPT-3.5）

### 成本预估
- 图像描述生成（GPT-4o）：约0.002-0.005美元/张
- 时间解析（GPT-3.5-turbo）：约0.0001-0.0002美元/次
- 文本嵌入（T5本地）：免费（首次下载模型后）

**100张照片索引成本估算**：
- 图像描述：100 × 0.003美元 = 0.3美元
- 文本嵌入：0美元
- 总计：约0.3美元（约2元人民币）

### 验收要求（必须达标）
1. 索引数量：成功索引 >= 100张照片
2. 查询类型覆盖：场景/人物/活动/时间/情感查询可用
3. 时间检索：支持"去年/冬天/2023年/上个月"等查询
4. 描述质量：降级描述占比 < 10%

### 错误处理规范
- 所有API调用都要有try-except
- 单个图片失败不应中断整体流程
- 使用降级策略（文件名描述）作为fallback
- 返回清晰的错误信息

### 注释要求
- 所有类和函数都要有docstring
- 关键逻辑行要添加注释
- 使用Type Hints

---

## 测试建议

### 单元测试（可选）
- utils/image_parser.py函数测试
- utils/vector_store.py功能测试

### 集成测试
- 索引构建流程测试（使用少量测试图片）
- 搜索功能测试（准备已知描述的图片）

### 验收测试
- 准备100+张测试照片
- 测试各类查询（场景/人物/活动/时间/情感）
- 验证时间检索功能（"去年"、"冬天"等）
- 检查降级描述占比

---

## 使用流程

1. **获取OpenRouter API密钥**
   - 访问 https://openrouter.ai/
   - 注册/登录账号
   - 在设置中创建API密钥

2. **环境准备**
   ```bash
   # 安装依赖
   pip install -r requirements.txt

   # 创建环境变量文件
   cp .env.example .env

   # 编辑.env，填入以下内容：
   # OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   # PHOTO_DIR=C:/Users/YourName/Photos
   ```

3. **启动服务**
   ```bash
   python main.py
   ```
   - 首次启动会自动下载T5模型（~1-2GB）
   - 模型缓存位置：~/.cache/huggingface

4. **访问界面**
   - 打开浏览器访问 http://localhost:5000

5. **初始化索引**
   - 点击"初始化索引"按钮
   - 等待索引构建完成（100张照片约需10-20分钟）
   - 成本预估：约0.3美元（2元人民币）

6. **开始搜索**
   - 输入自然语言查询（如"去年夏天在海边的照片"）
   - 查看搜索结果

---

## 开发完成标志

完成以下所有步骤即视为开发完成：

- [x] Phase 0: 项目初始化
- [x] Phase 1: 配置和基础设施
- [ ] Phase 2: 工具模块（5个文件）
- [ ] Phase 3: 核心业务（2个文件）
- [ ] Phase 4: 接口层（2个文件）
- [ ] Phase 5: 应用入口（1个文件）

**总计：12个文件 + 3个配置文件**

---

## 优先级说明

### 高优先级（核心功能）
- config.py - OpenRouter + T5配置
- utils/vector_store.py - FAISS向量存储
- utils/embedding_service.py - T5本地嵌入
- utils/vision_llm_service.py - OpenRouter + GPT-4 Vision
- core/indexer.py - 索引构建器
- core/searcher.py - 搜索引擎
- main.py - 应用入口

### 中优先级（必要功能）
- utils/image_parser.py - 图片解析
- utils/time_parser.py - OpenRouter + 时间解析
- api/routes.py - HTTP API路由

### 低优先级（界面和配置）
- templates/index.html - 前端界面
- requirements.txt - 依赖声明
- .env.example - 环境变量模板

---

## 技术栈总结

### 后端框架
- Flask 3.0.3 - Web框架
- Python 3.12+ - 运行环境

### AI服务
- **Vision LLM**：OpenRouter + GPT-4 Omni
- **Embedding**：sentence-transformers + T5（本地）
- **Time Parser**：OpenRouter + GPT-3.5-turbo

### 核心库
- FAISS - 向量相似度检索
- NumPy - 向量计算
- Pillow - 图像处理

### 工具库
- python-dotenv - 环境变量管理
- piexif - EXIF元数据解析

---

**最后更新时间**: 2026-02-02
**适用版本**: Python 3.12+, Flask 3.0.3
**开发模式**: Vibe Coding（逐步实现）
