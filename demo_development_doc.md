# 个人照片搜索引擎Demo开发文档

## 快速开始

### 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
# 创建 .env 文件：
# PHOTO_DIR=/path/to/your/photos
# OPENROUTER_API_KEY=your_openrouter_api_key_here
# OPENAI_API_KEY=your_openrouter_api_key_here  # 兼容性配置

# 3. 启动服务
python main.py

# 4. 访问界面
# 打开浏览器访问 http://localhost:5000
```

### 使用流程

1. **准备照片**：将照片放入指定的照片目录
2. **初始化索引**：点击"初始化索引"按钮，等待系统生成图片描述和向量索引
3. **开始搜索**：输入自然语言查询（如"去年夏天在海边的照片"）进行搜索

### 查询示例

- 场景查询："山上的照片"、"城市夜景"
- 人物查询："有人物的照片"、"集体合影"
- 活动查询："聚餐"、"运动"
- 时间查询："去年的照片"、"冬天的照片"
- 情感查询："欢乐的场景"

---

## 项目概述

本项目是一个基于AI的照片搜索引擎Demo，支持通过自然语言查询检索本地照片。系统使用Vision LLM生成图片描述，通过向量相似度检索实现智能搜索，支持场景、人物、活动、时间、情感等多类查询。

### 核心特性

- **智能描述生成**：使用Vision LLM自动生成图片的中文自然语言描述
- **向量化检索**：基于FAISS的高效向量相似度搜索
- **多维度查询**：支持场景、人物、活动、时间、情感等多种查询类型
- **时间检索**：基于EXIF元数据或文件时间，支持"去年"、"冬天"等时间词
- **成本优化**：使用Base64编码+图片压缩策略，大幅降低 Token 消耗

### 图片访问机制

**重要说明**：OpenRouter 是云端服务，无法访问本地 `localhost:5000`。因此本系统采用 **Base64 编码方式**。

| 方式 | Token消耗 | 实现复杂度 | 适用场景 | OpenRouter支持 |
|------|----------|-----------|---------|--------------|
| Base64编码（优化） | **中等**（压缩后约50-200KB） | 中等 | **本地Demo（推荐）** | ✅ 支持 |
| Base64编码（原始） | 高（约图片大小/3） | 简单 | 生产环境 | ✅ 支持 |
| 本地HTTP URL | 低（仅URL字符串） | 中等 | 本地服务器 | ❌ 不支持（云端无法访问） |

**成本优化策略**：
1. **自动缩放**：图片自动缩放至 `max_size`（默认 1024 像素），等比缩放保持宽高比
2. **格式压缩**：使用 WEBP 格式（默认），比 JPEG 更小，质量参数可调（默认 85）
3. **智能选择**：小于 `max_size` 的图片不放大，避免不必要的数据传输

实现原理：
1. 读取本地图片文件
2. 使用 PIL/Pillow 进行 EXIF 方向校正和尺寸优化
3. 压缩为 WEBP/JPEG 格式（可配置）
4. Base64 编码并生成 `data:image/...;base64,...` URL
5. 发送给 OpenRouter Vision API

示例代码：
```python
# 优化并编码图片
image_bytes = resize_and_optimize_image(
    photo_path,
    max_size=1024,      # 最大边长
    quality=85,          # 压缩质量
    format="WEBP"        # 输出格式
)
base64_str = base64.b64encode(image_bytes).decode("utf-8")
image_url = f"data:image/webp;base64,{base64_str}"

# OpenRouter Vision API调用
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }]
)
```

**环境变量配置**（可选）：
```bash
# 是否使用 Base64 方式（默认 true，推荐保持）
USE_BASE64=true

# 图片最大边长（像素，默认 1024）
IMAGE_MAX_SIZE=1024

# 图片压缩质量（1-100，默认 85）
IMAGE_QUALITY=85

# 图片输出格式（JPEG/WEBP/PNG，默认 WEBP）
IMAGE_FORMAT=WEBP
```

### 技术架构

```
用户浏览器
    ↓
templates/index.html（前端界面）
    ↓
api/routes.py（HTTP API）
    ↓
core/indexer.py（索引构建）  core/searcher.py（检索引擎）
    ↓                        ↓
utils/vision_llm_service.py（Vision LLM - OpenRouter + GPT-4）
utils/embedding_service.py（文本Embedding - T5模型）
utils/vector_store.py（FAISS向量存储）
```

## 文档目录

- [0. 方案可行性分析](#0-方案可行性分析)
  - [0.1 技术依赖验证](#01-技术依赖验证)
  - [0.2 可行性结论](#02-可行性结论)
  - [0.3 验收对齐说明](#03-验收对齐说明)
- [1. 项目结构总览](#1-项目结构总览)
  - [1.1 落地文件结构与关键函数清单](#11-落地文件结构与关键函数清单)
- [2. Agent工作流程设计](#2-agent工作流程设计)
  - [2.1 单Agent工作流程（Indexer Agent）](#21-单agent工作流程indexer-agent)
  - [2.2 单Agent工作流程（Searcher Agent）](#22-单agent工作流程searcher-agent)
- [3. 文件夹详情](#3-文件夹详情)
  - [3.1 core/（核心业务模块）](#31-core核心业务模块)
  - [3.2 utils/（工具模块）](#32-utils工具模块)
  - [3.3 api/（API接口层）](#33-api接口层)
  - [3.4 templates/（前端模板）](#34-templates前端模板)
  - [3.5 config.py](#35-configpy)
  - [3.6 main.py](#36-mainpy)
- [4. 内部接口说明](#4-内部接口说明)
- [5. 代码注释要求](#5-代码注释要求)
  - [5.1 函数/类顶部注释](#51-函数类顶部注释)
  - [5.2 关键逻辑行注释](#52-关键逻辑行注释)
  - [5.3 接口定义处注释](#53-接口定义处注释)
- [6. 实现约束](#6-实现约束)
  - [6.1 禁止事项](#61-禁止事项)
  - [6.2 必做事项](#62-必做事项)
  - [6.3 测试覆盖要求](#63-测试覆盖要求)

---

## 模型配置说明

本项目采用混合模型策略，结合云端和本地模型的优势：

### Vision LLM（图像描述生成）

**配置**：
- API网关：OpenRouter（https://openrouter.ai/api/v1）
- 底层模型：GPT-4 Vision（openai/gpt-4o）
- API密钥：OPENROUTER_API_KEY（环境变量）

**选择理由**：
- OpenRouter提供统一API接口，支持多种视觉模型
- GPT-4o具备优秀的图像理解能力和中文输出质量
- 按需付费，成本透明，适合Demo阶段
- 支持URL方式访问图片，避免Base64编码的token浪费

**备选模型**（可在config.py中切换）：
- openai/gpt-4-turbo：速度更快，成本略低
- anthropic/claude-3-sonnet：多模态能力，视觉质量好
- google/gemini-pro-vision：Google的视觉模型

### Embedding模型（文本向量嵌入）

**配置**：
- 模型：sentence-t5-base（sentence-transformers）
- 向量维度：768
- 运行方式：本地运行（CPU/GPU）

**选择理由**：
- sentence-transformers基于T5，中文语义理解能力强
- 本地运行，无API调用成本，适合大批量文本处理
- 开源免费，无需额外API密钥
- 向量维度适中（768），FAISS检索效率高

**备选模型**（可在config.py中切换）：
- sentence-t5-large：768维，效果更佳，但运行速度稍慢
- sentence-t5-xxl：768维，效果最佳，需更多内存
- paraphrase-multilingual-MiniLM-L12-v2：384维，速度快，适合资源受限场景

**依赖安装**：
```bash
pip install torch sentence-transformers>=2.2.0
```

**模型首次使用**：
- 模型会自动从Hugging Face下载到本地缓存（~/.cache/huggingface）
- sentence-t5-base约需下载1-2GB数据
- 下载后本地运行，无需网络

### 环境变量配置

在项目根目录创建 `.env` 文件：

```bash
# OpenRouter API密钥（用于Vision LLM和时间解析）
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 兼容性配置（如果使用原始OpenAI密钥）
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenRouter API地址（通常无需修改）
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# 照片目录（绝对路径）
PHOTO_DIR=C:/Users/YourName/Photos
```

**获取OpenRouter API密钥**：
1. 访问 https://openrouter.ai/
2. 注册/登录账号
3. 在设置中创建API密钥
4. 将密钥复制到 .env 文件

**成本预估**：
- 图像描述生成（GPT-4o）：约0.002-0.005美元/张
- 时间解析（GPT-3.5-turbo）：约0.0001-0.0002美元/次
- 文本嵌入（T5本地）：免费（首次下载模型后）

**100张照片索引成本估算**：
- 图像描述：100 × 0.003美元 = 0.3美元
- 文本嵌入：0美元
- 总计：约0.3美元（约2元人民币）

---

## 0. 方案可行性分析

### 0.1 技术依赖验证

已验证可用的核心依赖：
- Python 3.12.4
- Flask 3.0.3（Web框架）
- numpy 1.26.4（向量计算）
- Pillow 10.3.0（图像处理）

需要额外安装的依赖：
- faiss-cpu（向量存储与检索）
- openai>=1.0.0（OpenAI API客户端，兼容OpenRouter）
- sentence-transformers（T5文本嵌入模型）
- python-dotenv（环境变量管理）
- piexif（EXIF元数据解析）
- torch（PyTorch，sentence-transformers后端）

完整的requirements.txt：
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

### 0.2 可行性结论

方案可行（面向验收目标）：
- 核心功能依赖均已具备
- FAISS可通过pip安装，支持Windows平台
- 接口抽象设计允许灵活切换后端服务
- 项目结构清晰，模块职责明确
 - 已对齐验收指标（数据规模、多类查询、时间检索、描述质量）

实现路径（验收导向）：
1. Phase 1：实现基础功能（图片解析、向量存储、简单检索）
2. Phase 2：集成AI服务（Vision LLM、Embedding），确保描述质量
3. Phase 3：完善Web接口和前端交互，支持多类查询与时间检索

### 0.3 验收对齐说明（必须达标）
为保证“基于本文档进行vibe coding即可完成验收”，以下指标必须实现：
- 索引数量：成功索引至少100张照片
- 查询类型覆盖：场景/人物/活动/时间/情感查询可用
- 时间检索：基于EXIF时间或文件时间完成“去年/冬天”等查询
- 描述质量：描述需真实反映图片内容，降级描述仅允许占比<10%


## 1. 项目结构总览

- core/：核心业务逻辑模块，包含索引构建和检索引擎
- utils/：工具类模块，支撑照片解析、图像描述生成、文本嵌入和向量存储
- api/：API接口层，提供HTTP服务接口
- templates/：前端模板，提供用户交互界面
- main.py：应用入口，初始化服务和启动Web服务器
- config.py：配置文件，管理API密钥、模型参数等

### 1.1 落地文件结构与关键函数清单（用于实现时对照）

以下清单用于“落地实现”，明确每个文件的核心职责与必须实现的关键函数/接口。

#### core/indexer.py
- 类：Indexer
- 关键函数：
  - __init__(): 依赖注入（Vision/Embedding/VectorStore）
  - scan_photos(): 递归扫描、格式过滤、排序
  - process_batch(): 串行处理一批图片（描述/嵌入/元数据）
  - generate_description(): 调用Vision LLM + 降级策略
  - build_index(): 主流程（验收门槛检查、统计报告、marker）
  - get_status(): 获取索引构建状态（供index_status API调用）

#### core/searcher.py
- 类：Searcher
- 关键函数：
  - __init__(): 初始化路径与服务，默认未加载索引
  - load_index(): 加载FAISS索引与metadata
  - validate_query(): 查询长度、空值、非法字符过滤
  - _extract_time_constraints(): 解析时间约束（见 3.1.2.2）
  - _filter_by_time(): EXIF优先、文件时间兜底
  - _distance_to_score(): L2距离 -> 相似度
  - search(): Top-K检索 + 时间过滤 + 排序
  - get_index_stats(): 统计接口（前端状态显示用）

#### utils/image_parser.py
- 函数：
  - is_valid_image(): 格式与文件头校验
  - extract_exif_metadata(): 解析拍摄时间/相机/GPS
  - get_file_time(): 文件时间兜底（创建/修改时间）
  - get_image_dimensions(): 读取宽高并考虑方向
  - generate_fallback_description(): 基于文件名的降级描述

#### utils/time_parser.py

职责：使用LLM语义理解解析查询中的时间约束

##### 3.2.1.6 类：TimeParser

职责：使用OpenAI Chat Completion API解析自然语言查询中的时间约束

属性：
- api_key: str - OpenAI API密钥
- model_name: str - 使用的模型名称（默认g'pt-3.5-turbo）
- client: OpenAI - OpenAI客户端实例
- timeout: int - API调用超时时间（秒），默认10

方法：

__init__(api_key: str, model_name: str = "gpt-3.5-turbo")
- 功能：初始化时间解析器
- 输入：
  - api_key: str - OpenAI API密钥
  - model_name: str - 模型名称，默认'gpt-3.5-turbo'
- 输出：无
- 注释要求：需说明API Key验证、客户端初始化

extract_time_constraints(query: str) -> Dict[str, Any]
- 功能：使用LLM语义理解解析查询中的时间约束
- 输入：query: str - 用户查询文本
- 输出：Dict[str, Any] - 时间约束字典，包含：
  - start_date: str | None - ISO 8601 日期起始（含）
  - end_date: str | None - ISO 8601 日期结束（含）
  - precision: str - "year" | "month" | "season" | "range" | "none"
- 注释要求：需说明Prompt设计、JSON响应解析、错误处理

_infer_precision(start_date: str | None, end_date: str | None) -> str
- 功能：根据日期范围推断精度级别
- 输入：
  - start_date: str | None - 起始日期
  - end_date: str | None - 结束日期
- 输出：str - 精度级别
- 注释要求：需说明精度推断逻辑

**实现策略**：
1. 构造包含当前日期的Prompt
2. 要求LLM返回结构化JSON
3. 解析JSON返回时间范围
4. 推断精度级别

**Prompt模板**：
```
当前日期：{current_date}（格式：YYYY-MM-DD）

用户查询：{query}

请分析用户查询中的时间约束，返回JSON格式：
{{
  "has_time_constraint": true/false,
  "start_date": "YYYY-MM-DD" 或 null,
  "end_date": "YYYY-MM-DD" 或 null,
  "reasoning": "简要说明解析逻辑"
}}

规则：
1. 如果没有明确的时间词，has_time_constraint=false，其他字段为null
2. 相对时间基于当前日期计算：
   - "去年" -> 去年全年
   - "今年" -> 今年全年
   - "上个月" -> 上个月
   - "冬天" -> 当年12月到次年2月
   - "去年冬天" -> 去年12月到今年2月
3. 季节定义月：
   - 春：3月1日-5月31日
   - 夏：6月1日-8月31日
   - 秋：9月1日-11月30日
   - 冬：12月1日-次年2月28/29日
4. 日期范围包含边界
```

**实现示例**：
```python
class TimeParser:
    """使用LLM语义理解解析时间约束"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        初始化时间解析器
        
        Args:
            api_key: OpenAI API密钥
            model_name: 模型名称，默认gpt-3.5-turbo
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.timeout = 10
    
    def extract_time_constraints(self, query: str) -> Dict[str, Any]:
        """
        使用LLM语义理解解析时间约束
        
        Args:
            query: 用户查询文本
            
        Returns:
            Dict包含：
            - start_date: str | None - ISO 8601 日期起始
            - end_date: str | None - ISO 8601 日期结束
            - precision: str - 精度级别
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""当前日期：{current_date}（格式：YYYY-MM-DD）

用户查询：{query}

请分析用户查询中的时间约束，返回JSON格式：
{{
  "has_time_constraint": true/false,
  "start_date": "YYYY-MM-DD" 或 null,
  "end_date": "YYYY-MM-DD" 或 null,
  "reasoning": "简要说明解析逻辑"
}}

规则：
1. 如果没有明确的时间词，has_time_constraint=false，其他字段为null
2. 相对时间基于当前日期计算：
   - "去年" -> 去年全年
   - "今年" -> 今年全年
   - "上个月" -> 上个月
   - "冬天" -> 当年12月到次年2月
   - "去年冬天" -> 去年12月到今年2月
3. 季节定义：
   - 春：3月1日-5月31日
   - 夏：6月1日-8月31日
   - 秋：9月1日-11月30日
   - 冬：12月1日-次年2月28/29日
4. 日期范围包含边界"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # 使用OpenRouter模型，如"openai/gpt-3.5-turbo"
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
                timeout=self.timeout
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if not result.get("has_time_constraint"):
                return {
                    "start_date": None,
                    "end_date": None,
                    "precision": "none"
                }
            
            start_date = result.get("start_date")
            end_date = result.get("end_date")
            precision = self._infer_precision(start_date, end_date)
            
            return {
                "start_date": start_date,
                "end_date": end_date,
                "precision": precision
            }
            
        except Exception as e:
            logger.error(f"时间解析失败: {e}")
            return {
                "start_date": None,
                "end_date": None,
                "precision": "none"
            }
    
    def _infer_precision(self, start_date: str | None, end_date: str | None) -> str:
        """
        根据日期范围推断精度级别
        
        Args:
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            str: 精度级别
        """
        if not start_date or not end_date:
            return "none"
        
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        delta = end - start
        
        # 跨年情况，判断是否是季节（约3个月）
        if end.year != start.year:
            if delta.days <= 90:
                return "season"
            else:
                return "range"
        
        # 同年情况
        if delta.days <= 31:
            return "month"
        elif delta.days <= 90:
            return "season"
        else:
            return "year"
```

**解析示例**：
- "去年夏天在海边的照片" -> start_date=2025-06-01, end_date=2025-08-31, precision="season"
- "2023年的照片" -> start_date=2023-01-01, end_date=2023-12-31, precision="year"
- "冬天的风景" -> start_date=2026-12-01, end_date=2027-02-28, precision="season"（假设当前是2026年）
- "海边的照片" -> start_date=null, end_date=null, precision="none"（无时间约束）

#### utils/vision_llm_service.py
- 类：VisionLLMService（抽象）
  - generate_description()
  - generate_description_batch()
- 类：OpenAIVisionLLMService（默认实现）
  - _get_image_url()
  - generate_description()

#### utils/embedding_service.py
- 类：EmbeddingService（抽象）
  - generate_embedding()
  - generate_embedding_batch()
- 类：OpenAIEmbeddingService（默认实现）
  - generate_embedding()

#### utils/vector_store.py
- 类：VectorStore
  - __init__(): 初始化FAISS索引与存储路径
  - add_item(): 写入向量与元数据
  - search(): 相似度检索
  - save()/load(): 索引持久化
  - get_total_items(): 统计数量

#### api/routes.py
- 函数：
  - register_routes(): 注册所有API路由
- API：
  - init_index(): 触发Indexer.build_index()
  - search_photos(): 触发Searcher.search()
  - index_status(): 获取索引构建和加载状态
  - get_photo(): 本地图片访问

#### templates/index.html（本地演示页面）
- 前端函数：
  - initIndex(): POST /init_index，展示状态
  - checkIndexStatus(): 轮询索引状态
  - searchPhotos(): POST /search_photos，渲染网格

#### main.py
- create_app(): 组装Flask应用、注册路由
- main(): 初始化服务并启动服务器

## 2. Agent工作流程设计

### 2.1 单Agent工作流程（Indexer Agent）

职责：完成照片索引构建的完整流程

工作步骤：

1. 初始化阶段
   - 加载配置参数（照片目录、模型选择）
   - 初始化Vision LLM服务实例
   - 初始化Embedding服务实例
   - 创建/加载FAISS向量索引

2. 扫描阶段
   - 遍历用户指定的照片目录
   - 使用glob模式匹配支持的图片格式（*.jpg, *.jpeg, *.png, *.webp）
   - 过滤掉无效或损坏的图片文件
   - 返回有效图片文件路径列表

3. 批处理阶段
   - 将图片列表分批（默认batch_size=10）
   - 对每批图片串行处理（避免API速率限制）
   - 记录处理进度，支持断点续传

4. 描述生成阶段（质量保障）
   - 对每张图片调用Vision LLM服务
   - 生成自然语言描述（中文，50-200字），需包含场景/主体/动作/环境要素
   - 处理API调用失败（重试3次，超时30秒）
   - 失败时使用降级策略（基于文件名生成简单描述）
   - 质量约束：降级描述占比必须<10%，否则索引结果标记为failed

5. 嵌入生成阶段
    - 将描述文本发送到Embedding服务（T5模型）
    - 获取固定维度的向量（sentence-t5-base为768维）
    - 验证向量维度一致性

6. 索引存储阶段
   - 将向量添加到FAISS索引
   - 存储元数据（photo_path、description、exif_data、file_time）
   - 定期保存索引到磁盘（每处理50张图片）

7. 完成阶段（验收门槛）
   - 保存完整索引文件
   - 生成索引完成信号（创建index_ready.marker）
   - 生成索引统计报告（总数量、成功数、失败数、降级描述占比）
   - 验收门槛：成功索引数量必须>=100，否则返回failed
   - 返回构建结果

### 2.2 单Agent工作流程（Searcher Agent）

职责：完成相似度检索和结果排序

工作步骤：

1. 初始化阶段
   - 加载FAISS索引文件（等待index_ready.marker文件出现）
   - 加载元数据文件
   - 初始化Embedding服务实例
   - 验证索引完整性

2. 查询处理阶段
   - 接收用户自然语言查询（中文）
   - 清洗查询文本（去除特殊字符、限制长度）
   - 验证查询非空
   - 解析时间约束（如“去年/冬天/2023年/上个月”等）
   - 解析查询类型（场景/人物/活动/情感），用于提示词与结果解释

3. 查询嵌入阶段
   - 调用Embedding服务生成查询向量
   - 验证向量维度与索引维度一致

4. 相似度检索阶段
   - 在FAISS索引中执行Top-K搜索
   - 获取距离分数（L2距离）
   - 转换为相似度分数（0-1范围）

5. 结果排序阶段
   - 按相似度分数降序排列
   - 关联元数据（photo_path、description）
   - 基于时间约束过滤结果（优先EXIF时间，其次文件时间）
   - 限制返回数量（默认top_k=10）

6. 结果返回阶段
   - 构造标准JSON响应
   - 包含每张照片的路径、描述、相似度
   - 添加检索耗时统计

## 3. 文件夹详情

### 3.1 core/（核心业务模块）

职责：实现照片索引构建和相似度检索的核心业务逻辑

#### 3.1.1 indexer.py

职责：管理照片索引构建的完整流程

##### 3.1.1.1 类：Indexer

职责：协调照片扫描、描述生成、嵌入生成和索引存储

属性：
- photo_dir: str - 照片存储目录路径，绝对路径
- vision_llm_service: VisionLLMService - Vision LLM服务实例，用于图像描述生成
- embedding_service: EmbeddingService - 嵌入服务实例，用于文本向量化
- vector_store: VectorStore - 向量存储实例，FAISS索引封装
- batch_size: int - 批处理大小，默认10，避免API速率限制
- max_retries: int - API调用最大重试次数，默认3
- timeout: int - API调用超时时间（秒），默认30
- data_dir: str - 数据存储目录，默认"./data"

方法：

__init__(photo_dir: str, vision: VisionLLMService, embedding: EmbeddingService, data_dir: str = "./data")
- 功能：初始化索引构建器
- 输入：
  - photo_dir: str - 照片目录路径
  - vision: VisionLLMService - Vision LLM服务实例
  - embedding: EmbeddingService - 嵌入服务实例
  - data_dir: str - 数据存储目录，默认"./data"
- 输出：无
- 注释要求：需说明目录创建

build_index() -> Dict[str, Any]
- 功能：构建完整的照片索引，扫描所有图片、生成描述、计算嵌入、存储索引
- 输入：无
- 输出：Dict[str, Any] - 索引构建结果字典，包含：
  - status: str - "success"或"failed"
  - total_count: int - 扫描到的图片总数
  - success_count: int - 成功索引的图片数量
  - failed_count: int - 失败的图片数量
  - fallback_ratio: float - 降级描述占比（0-1）
  - index_path: str - 索引文件保存路径
  - elapsed_time: float - 构建耗时（秒）
- 注释要求：需说明批量处理逻辑、错误恢复机制、进度报告机制、验收门槛（成功>=100且降级占比<10%）

scan_photos() -> List[str]
- 功能：扫描指定目录，返回所有照片文件路径，支持递归扫描子目录
- 输入：无
- 输出：List[str] - 照片文件路径列表，按修改时间排序
- 注释要求：需说明支持的图片格式（jpg、jpeg、png、webp）、文件过滤逻辑、路径标准化

generate_description(photo_path: str) -> str
- 功能：调用Vision LLM生成照片的自然语言描述，包含图像内容识别和场景理解
- 输入：photo_path: str - 照片文件路径（绝对路径）
- 输出：str - 生成的描述文本（中文，50-200字），描述场景、人物、动作、环境等
- 注释要求：需说明API调用超时处理、重试机制、降级策略（API失败时使用文件名生成描述）、质量约束（降级占比<10%）

process_batch(photo_paths: List[str]) -> List[Dict[str, Any]]
- 功能：批量处理一批照片，生成描述和嵌入向量
- 输入：photo_paths: List[str] - 照片文件路径列表
- 输出：List[Dict[str, Any]] - 处理结果列表，每项包含：
  - photo_path: str - 照片路径
  - description: str - 描述文本
  - embedding: List[float] - 嵌入向量
  - exif_data: Dict[str, Any] - EXIF元数据
  - file_time: str - 文件时间（ISO 8601）
  - status: str - "success"或"failed"
  - error: str - 错误信息（如果失败）
- 注释要求：需说明串行处理逻辑、错误隔离（单个失败不影响整批）、进度报告

get_status() -> Dict[str, Any]
- 功能：获取索引构建状态，供index_status API调用
- 输入：无
- 输出：Dict[str, Any] - 状态字典，包含：
  - status: str - "idle" | "processing" | "ready" | "failed"
  - message: str - 状态描述信息
  - total_count: int - 总图片数量
  - indexed_count: int - 已索引数量
  - failed_count: int - 失败数量
  - elapsed_time: float - 已耗时（秒）
- 注释要求：需说明状态机的读取逻辑、进度计算、锁文件检查

#### 3.1.2 searcher.py

职责：实现相似度检索逻辑

##### 3.1.2.1 类：Searcher

职责：执行查询向量化、相似度搜索和结果排序

属性：
- embedding_service: EmbeddingService - 嵌入服务实例，用于查询文本向量化
- vector_store: VectorStore - 向量存储实例，FAISS索引封装
- time_parser: TimeParser - 时间解析服务实例，用于提取时间约束
- index_loaded: bool - 索引是否已加载标志
- top_k: int - 默认返回结果数量，默认10
- data_dir: str - 数据存储目录，默认"./data"
- index_path: str - 索引文件路径，data/photo_search.index
- metadata_path: str - 元数据文件路径，data/metadata.json

方法：

__init__(embedding: EmbeddingService, time_parser: TimeParser, data_dir: str = "./data")
- 功能：初始化检索器
- 输入：
  - embedding: EmbeddingService - 嵌入服务实例
  - time_parser: TimeParser - 时间解析服务实例
  - data_dir: str - 数据存储目录，默认"./data"
- 输出：无
- 注释要求：需说明路径初始化、索引加载状态

load_index() -> bool
- 功能：从磁盘加载FAISS索引和元数据，验证索引完整性
- 输入：无
- 输出：bool - 加载是否成功
- 注释要求：需说明文件存在性校验、维度一致性验证、加载失败处理

search(query: str, top_k: int = 10) -> List[Dict[str, Any]]
- 功能：执行相似度检索并返回Top-K结果，支持中英文混合查询
- 输入：
  - query: str - 用户的自然语言查询（中文或英文）
  - top_k: int - 返回结果数量，默认10，最大不超过50
- 输出：List[Dict[str, Any]] - 检索结果列表，按相似度降序排列，每项包含：
  - photo_path: str - 照片文件路径
  - description: str - 照片的自然语言描述
  - score: float - 相似度分数（0-1范围，1为最相似）
  - rank: int - 排名位置
- 注释要求：需说明索引未加载时的错误处理、相似度计算方式（基于FAISS的L2距离）、时间约束解析与过滤逻辑、结果排序逻辑、空结果处理

get_index_stats() -> Dict[str, Any]
- 功能：获取索引统计信息，用于状态监控
- 输入：无
- 输出：Dict[str, Any] - 统计信息，包含：
  - total_items: int - 索引中的照片总数
  - vector_dimension: int - 向量维度
  - index_loaded: bool - 索引是否加载
  - index_path: str - 索引文件路径
- 注释要求：需说明统计信息的实时性、索引未加载时的默认值

validate_query(query: str) -> bool
- 功能：验证查询文本的有效性，过滤无效输入
- 输入：query: str - 查询文本
- 输出：bool - 查询是否有效
- 注释要求：需说明长度限制（5-500字符）、特殊字符过滤、空值处理、时间关键词合法性

_extract_time_constraints(query: str) -> Dict[str, Any]
- 功能：从查询文本中解析时间约束（年份、月份、季节、相对时间）
- 输入：query: str - 查询文本
- 输出：Dict[str, Any] - 结构化时间约束（start_date、end_date、season）
- 注释要求：需说明中文时间词解析（去年/今年/上个月/冬天）、解析失败的默认策略

##### 3.1.2.2 时间解析策略（LLM语义理解方案）

目标：利用LLM的语义理解能力，灵活解析中文时间查询，覆盖"去年/冬天/上个月/2023年/去年夏天在海边"等复杂场景。

**方案概述**：
- 使用OpenAI Chat Completion API进行时间解析
- 提供当前日期作为上下文
- 要求LLM返回结构化的JSON时间范围

**解析结果统一为**：
- start_date: str | None - ISO 8601 日期起始（含），格式："YYYY-MM-DD"
- end_date: str | None - ISO 8601 日期结束（含），格式："YYYY-MM-DD"
- precision: str - "year" | "month" | "season" | "range" | "none"

**Prompt设计**：
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
2. 相对时间基于当前日期计算：
   - "去年" -> 去年全年
   - "今年" -> 今年全年
   - "上个月" -> 上个月
   - "冬天" -> 当年12月到次年2月
   - "去年冬天" -> 去年12月到今年2月
3. 季节定义：
   - 春：3月1日-5月31日
   - 夏：6月1日-8月31日
   - 秋：9月1日-11月30日
   - 冬：12月1日-次年2月28/29日
4. 日期范围包含边界
```

**实现示例**：
```python
def _extract_time_constraints(query: str) -> Dict[str, Any]:
    """
    使用LLM语义理解解析时间约束
    
    Args:
        query: 用户查询文本
        
    Returns:
        Dict包含：
        - start_date: str | None
        - end_date: str | None
        - precision: str
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"当前日期：{current_date}\n\n用户查询：{query}\n\n请分析..."
    
    response = self.llm_client.chat.completions.create(
        model="openai/gpt-3.5-turbo",  # OpenRouter模型格式
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    if not result.get("has_time_constraint"):
        return {"start_date": None, "end_date": None, "precision": "none"}
    
    return {
        "start_date": result.get("start_date"),
        "end_date": result.get("end_date"),
        "precision": _infer_precision(result.get("start_date"), result.get("end_date"))
    }
```

**解析示例**：
- "去年夏天在海边的照片" -> start_date=2025-06-01, end_date=2025-08-31
- "2023年的照片" -> start_date=2023-01-01, end_date=2023-12-31
- "冬天的风景" -> start_date=2026-12-01, end_date=2027-02-28（假设当前是2026年）
- "海边的照片" -> start_date=null, end_date=null, precision="none"（无时间约束）

_filter_by_time(results: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]
- 功能：按时间约束过滤检索结果（优先EXIF时间，其次文件时间）
- 输入：
  - results: List[Dict[str, Any]] - 初始检索结果
  - constraints: Dict[str, Any] - 时间约束
- 输出：List[Dict[str, Any]] - 过滤后的结果
- 注释要求：需说明时间字段优先级、无时间信息的处理策略

_distance_to_score(distance: float) -> float
- 功能：将L2距离转换为相似度分数（0-1范围）
- 输入：distance: float - L2距离（FAISS IndexFlatL2 返回的是“平方L2距离”）
- 输出：float - 相似度分数
- 注释要求：需说明转换公式（示例：score = 1 / (1 + distance)）


### 3.2 utils/（工具模块）

职责：提供照片解析、AI服务调用和向量存储的基础工具

#### 3.2.1 image_parser.py

职责：解析图片元数据和验证图片格式

##### 3.2.1.1 函数：is_valid_image(file_path: str) -> bool

- 功能：验证文件是否为支持的图片格式，检查文件扩展名和实际文件头
- 输入：file_path: str - 待验证的文件路径
- 输出：bool - 是否为有效图片
- 注释要求：需说明支持的格式列表（JPG、JPEG、PNG、WEBP）、文件头验证逻辑、异常处理

##### 3.2.1.2 函数：extract_exif_metadata(file_path: str) -> Dict[str, Any]

- 功能：解析照片EXIF元数据，提取拍摄时间、地理位置、相机信息等
- 输入：file_path: str - 照片文件路径
- 输出：Dict[str, Any] - EXIF元数据字典，包含：
  - datetime: str - 拍摄时间（ISO 8601格式），默认为空字符串
  - location: Dict - GPS坐标（latitude、longitude），默认为None
  - camera: str - 相机型号，默认为"未知"
  - orientation: int - 图片方向，默认为1
- 注释要求：需说明字段缺失的默认值处理、不同厂商的EXIF标签兼容（Canon、Nikon、Sony等）、时间字段解析与规范化

##### 3.2.1.5 函数：get_file_time(file_path: str) -> str

- 功能：获取文件时间，用于时间检索的兜底
- 输入：file_path: str - 照片文件路径
- 输出：str - 文件时间（ISO 8601格式，优先创建时间，其次修改时间）
- 注释要求：需说明系统兼容（Windows/Unix）、异常处理、与EXIF时间的优先级关系

##### 3.2.1.3 函数：get_image_dimensions(file_path: str) -> Tuple[int, int]

- 功能：获取图片的宽度和高度
- 输入：file_path: str - 照片文件路径
- 输出：Tuple[int, int] - (width, height)，单位像素
- 注释要求：需说明旋转信息的处理（EXIF orientation）

##### 3.2.1.4 函数：generate_fallback_description(file_path: str) -> str

- 功能：基于文件名生成降级描述，用于Vision LLM API失败时
- 输入：file_path: str - 照片文件路径
- 输出：str - 简单描述（如"一张拍摄于2024年的照片"或"一张关于旅行的照片"）
- 注释要求：需说明文件名解析逻辑（日期、地点关键词提取）

**实现策略**：
1. 提取文件名（去除扩展名）
2. 从文件名提取日期（YYYY-MM-DD、YYYYMMDD、YYYY年MM月DD日等格式）
3. 提取主题关键词（常见的中文/英文场景词）
4. 组合生成描述

**实现示例**：
```python
def generate_fallback_description(file_path: str) -> str:
    """
    基于文件名生成降级描述
    
    Args:
        file_path: 照片文件路径
        
    Returns:
        str: 简单描述
    """
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # 提取日期
    date_match = re.search(r'(\d{4})[-_]?(\d{1,2})[-_]?(\d{1,2})', name_without_ext)
    date_str = ""
    if date_match:
        year, month, day = date_match.groups()
        date_str = f"拍摄于{year}年"
    
    # 提取主题关键词
    theme_keywords = {
        'vacation': '旅行', 'trip': '旅行', 'travel': '旅行',
        'beach': '海边', 'sea': '海边', 'ocean': '海边',
        'mountain': '山上', 'hike': '爬山', 'hiking': '爬山',
        'food': '美食', 'dinner': '聚餐', 'lunch': '聚餐',
        'party': '聚会', 'friends': '朋友', 'family': '家庭',
        'work': '工作', 'office': '办公室', 'meeting': '会议',
        'concert': '音乐会', 'music': '音乐', 'show': '演出',
        'sport': '运动', 'game': '比赛', 'gym': '健身房',
        'city': '城市', 'night': '夜景', 'sunset': '日落',
        'christmas': '圣诞节', 'birthday': '生日', 'wedding': '婚礼'
    }
    
    theme_str = ""
    name_lower = name_without_ext.lower()
    for en_key, cn_value in theme_keywords.items():
        if en_key in name_lower or cn_value in name_without_ext:
            theme_str = f"关于{cn_value}"
            break
    
    # 组合描述
    if date_str and theme_str:
        return f"一张{date_str}的{theme_str}照片"
    elif date_str:
        return f"一张{date_str}的照片"
    elif theme_str:
        return f"一张{theme_str}的照片"
    else:
        return "一张照片"
```

**解析示例**：
- "2024-05-15_vacation_beach.jpg" -> "一张拍摄于2024年的关于旅行的照片"
- "trip_to_mountain.jpg" -> "一张关于旅行的照片"
- "family_dinner_20231225.jpg" -> "一张拍摄于2023年的关于聚餐的照片"
- "IMG_20240101_123456.jpg" -> "一张拍摄于2024年的照片"
- "random_photo.jpg" -> "一张照片"

#### 3.2.2 vision_llm_service.py

职责：封装Vision LLM服务调用，支持多后端

##### 3.2.2.1 类：VisionLLMService（抽象基类）

职责：定义Vision LLM服务的统一接口，支持运行时切换不同后端

属性：
- model_name: str - 使用的模型名称
- timeout: int - API调用超时时间（秒）

方法：

generate_description(image_path: str) -> str
- 功能：生成图像的自然语言描述，分析图像内容、场景、人物、情绪等
- 输入：image_path: str - 图像文件路径（绝对路径）
- 输出：str - 生成的描述文本（中文，50-200字）
- 注释要求：需说明子类必须实现此方法、异常处理规范、描述要素约束（场景/主体/动作/环境/情绪）

generate_description_batch(image_paths: List[str]) -> List[str]
- 功能：批量生成图像描述，提高效率
- 输入：image_paths: List[str] - 图像文件路径列表
- 输出：List[str] - 描述文本列表
- 注释要求：需说明并行处理逻辑、失败隔离

##### 3.2.2.2 类：OpenAIVisionLLMService（继承VisionLLMService）

职责：使用OpenRouter接入GPT-4生成图像描述。通过本地HTTP服务URL访问图片，避免Base64编码的token消耗

**模型配置**：
- 使用OpenRouter作为API网关
- 底层模型：GPT-4 Vision（gpt-4o或gpt-4-turbo）
- 兼容OpenAI API客户端，只需修改base_url和api_key

属性：
- api_key: str - OpenRouter API密钥，从环境变量OPENROUTER_API_KEY或OPENAI_API_KEY读取
- base_url: str - OpenRouter API地址，默认"https://openrouter.ai/api/v1"
- model_name: str - 模型名称，默认"openai/gpt-4o"（GPT-4 Omni，支持视觉）
- max_tokens: int - 最大生成token数，默认300
- client: OpenAI - OpenAI客户端实例（配置为OpenRouter）
- server_host: str - 本地服务器主机地址，默认"localhost"
- server_port: int - 本地服务器端口，默认5000

方法：

__init__(api_key: str, model_name: str = "openai/gpt-4o", base_url: str = "https://openrouter.ai/api/v1")
- 功能：初始化OpenRouter Vision服务
- 输入：
  - api_key: str - OpenRouter API密钥
  - model_name: str - 模型名称，默认为OpenRouter上的GPT-4 Omni
  - base_url: str - OpenRouter API地址
- 输出：无
- 注释要求：需说明API Key验证、客户端初始化、base_url配置

**OpenRouter配置说明**：
OpenRouter是一个AI API网关，提供统一接口访问多种模型。本项目使用OpenRouter接入GPT-4 Vision模型。

优势：
- 统一API接口，方便切换模型
- 支持多种视觉模型（GPT-4o、GPT-4-turbo、Claude 3等）
- 按需付费，成本透明

generate_description(image_path: str) -> str
- 功能：调用OpenAI API生成图像描述
- 输入：image_path: str - 图像文件路径
- 输出：str - 生成的描述文本
- 注释要求：需说明通过本地HTTP服务URL访问图片、提示词工程（Prompt设计，要求包含场景/主体/动作/环境/情绪）、API响应解析

_get_image_url(image_path: str) -> str
- 功能：生成图片的可访问URL，通过本地HTTP服务提供
- 输入：image_path: str - 图片文件路径
- 输出：str - 图片的HTTP访问URL
- 注释要求：需说明URL格式为http://localhost:port/photo?path=xxx、路径编码处理

#### 3.2.3 embedding_service.py

职责：封装文本嵌入生成服务，支持多后端

##### 3.2.3.1 类：EmbeddingService（抽象基类）

职责：定义嵌入服务的统一接口

属性：
- model_name: str - 使用的模型名称

方法：

generate_embedding(text: str) -> List[float]
- 功能：生成文本的嵌入向量
- 输入：text: str - 待嵌入的文本
- 输出：List[float] - 嵌入向量
- 注释要求：需说明子类必须实现此方法、向量维度一致性

generate_embedding_batch(texts: List[str]) -> List[List[float]]
- 功能：批量生成嵌入向量
- 输入：texts: List[str] - 文本列表
- 输出：List[List[float]] - 嵌入向量列表
- 注释要求：需说明批处理效率优化

##### 3.2.3.2 类：T5EmbeddingService（继承EmbeddingService）

职责：使用Hugging Face的T5模型生成文本嵌入，作为本地嵌入方案

**模型配置**：
- 使用sentence-transformers库
- 底层模型：sentence-t5-base或sentence-t5-large
- 支持中文和英文文本嵌入
- 向量维度：768（sentence-t5-base）

属性：
- model_name: str - 模型名称，默认"sentence-t5-base"
- model: Any - sentence-transformers模型实例
- device: str - 运行设备，默认"cuda"（如果有GPU），否则"cpu"

方法：

__init__(model_name: str = "sentence-t5-base", device: str = None)
- 功能：初始化T5嵌入服务
- 输入：
  - model_name: str - 模型名称，默认"sentence-t5-base"
  - device: str - 运行设备，None表示自动检测
- 输出：无
- 注释要求：需说明模型首次下载、设备检测、内存占用

generate_embedding(text: str) -> List[float]
- 功能：调用T5模型生成嵌入向量
- 输入：text: str - 待嵌入的文本
- 输出：List[float] - 嵌入向量（维度为768）
- 注释要求：需说明文本预处理、向量归一化、异常处理

generate_embedding_batch(texts: List[str]) -> List[List[float]]
- 功能：批量生成嵌入向量（提高效率）
- 输入：texts: List[str] - 文本列表
- 输出：List[List[float]] - 嵌入向量列表
- 注释要求：需说明批处理效率、batch_size限制

**T5模型说明**：
T5（Text-to-Text Transfer Transformer）是Google提出的预训练模型，sentence-transformers基于T5提供了高质量的文本嵌入能力。

优势：
- 本地运行，无API调用成本
- 支持中文语义理解
- 开源免费，无需API密钥
- 向量维度适中（768），检索效率高

模型推荐：
- sentence-t5-base：768维，速度较快，适合一般场景
- sentence-t5-large：768维，效果更好，适合追求质量的场景


#### 3.2.4 vector_store.py

职责：封装FAISS向量存储操作

##### 3.2.4.1 类：VectorStore

职责：管理向量索引的构建、存储和检索

属性：
- index: faiss.Index - FAISS索引对象
- metadata: List[Dict] - 元数据列表（photo_path、description、exif_data、file_time）
- dimension: int - 向量维度
- index_path: str - 索引文件路径
- metadata_path: str - 元数据文件路径

方法：

__init__(dimension: int, index_path: str = None, metadata_path: str = None)
- 功能：初始化向量存储
- 输入：
  - dimension: int - 向量维度
  - index_path: str - 索引文件路径，可选
  - metadata_path: str - 元数据文件路径，可选
- 输出：无
- 注释要求：需说明FAISS索引类型选择（默认FlatL2，使用L2距离）、路径初始化

add_item(embedding: List[float], metadata: Dict) -> None
- 功能：添加向量及元数据到索引
- 输入：
  - embedding: List[float] - 嵌入向量
  - metadata: Dict - 元数据（photo_path、description、exif_data、file_time）
- 输出：无
- 注释要求：需说明向量维度校验逻辑、元数据存储

search(query_embedding: List[float], top_k: int) -> List[Dict]
- 功能：执行相似度搜索
- 输入：
  - query_embedding: List[float] - 查询向量
  - top_k: int - 返回结果数量
- 输出：List[Dict] - 检索结果列表，包含metadata和score
- 注释要求：需说明FAISS search返回值的含义（L2距离分数、索引ID）、元数据关联

save() -> None
- 功能：保存索引到磁盘
- 输入：无
- 输出：无
- 注释要求：需说明FAISS索引和元数据分开保存、文件写入异常处理

load() -> bool
- 功能：从磁盘加载索引
- 输入：无
- 输出：bool - 加载是否成功
- 注释要求：需说明文件存在性校验、维度一致性验证、加载失败处理

get_total_items() -> int
- 功能：获取索引中的项目数量
- 输入：无
- 输出：int - 项目数量
- 注释要求：需说明实时统计


### 3.3 api/（API接口层）

职责：提供HTTP服务接口，连接前端和核心业务逻辑

#### 3.3.1 routes.py

职责：定义API路由和请求处理逻辑

##### 3.3.1.1 函数：init_index()

- 功能：初始化照片索引，触发Indexer Agent的build_index方法
- 输入：无（接收POST请求）
- 输出：JSON响应，格式：
  {
    "status": "success/processing/failed",
    "message": "索引构建中/成功/失败",
    "indexed_count": 100,
    "total_count": 105,
    "failed_count": 5
  }
- 注释要求：需说明异步索引构建的处理方式（后台线程）、锁文件检查、响应格式规范

##### 3.3.1.2 函数：search_photos()

- 功能：执行照片搜索，触发Searcher Agent的search方法
- 输入：无（接收POST请求，包含query、top_k参数）
- 输出：JSON响应，格式：
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
- 注释要求：需说明参数校验和错误处理、索引未加载时的错误响应、photo_url 的生成方式

##### 3.3.1.3 函数：get_photo()

- 功能：本地演示用的图片访问接口，根据路径返回图片文件，供前端和Vision LLM使用
- 输入：GET请求，参数 path（图片绝对路径）
- 输出：图片二进制内容（由后端 send_file 返回）
- 注释要求：需说明仅供本地单人使用、路径校验（必须是绝对路径且存在）、错误返回（中文）、安全检查（路径遍历防护）

##### 3.3.1.4 函数：index_status()

- 功能：获取索引构建和加载状态，供前端轮询使用
- 输入：GET请求，无参数
- 输出：JSON响应，格式：
  ```json
  {
    "status": "idle" | "processing" | "ready" | "failed",
    "message": "未开始索引构建" | "索引构建中... (50/100)" | "索引就绪，共100张照片" | "索引构建失败",
    "total_count": 100,
    "indexed_count": 50,
    "failed_count": 0,
    "elapsed_time": 45.2
  }
- 注释要求：需说明状态机逻辑（idle->processing->ready/failed）、读取锁文件和标记文件、实时进度报告

### 3.4 templates/（前端模板）

职责：提供用户交互界面（本地演示为主，由Flask渲染页面与图片）

#### 3.4.1 index.html

职责：搜索和结果展示页面

##### 3.4.1.0 最简前端建议（本地localhost可运行）

目标：无需前端框架，使用单页HTML + 少量JS，满足“初始化索引”和“搜索展示”。

页面元素建议：
- 顶部标题与简要说明
- 索引按钮（触发 /init_index）
- 查询输入框 + 搜索按钮（触发 /search_photos）
- 状态区域（索引状态与错误提示）
- 结果展示区域（网格卡片，包含图片、描述、分数）

最简交互逻辑：
- initIndex(): POST /init_index，显示“索引构建中”，启动 checkIndexStatus() 轮询
- checkIndexStatus(): 每2秒轮询 /init_index 或 /index_status（若实现），直到成功/失败
- searchPhotos(): POST /search_photos，渲染结果卡片（img src=photo_url）

页面布局建议（纯HTML/CSS即可）：
- 使用 CSS Grid 展示结果：
  - 网格列：auto-fill，最小宽度 220px
  - 卡片内显示图片、描述、相似度

##### 3.4.1.1 函数：initIndex()

- 功能：触发索引初始化请求，显示进度条
- 输入：无
- 输出：无
- 注释要求：需说明加载状态提示、进度条更新逻辑、错误提示

##### 3.4.1.2 函数：searchPhotos()

- 功能：发送搜索请求并渲染结果
- 输入：query: str - 查询文本
- 输出：无
- 注释要求：需说明结果网格布局的渲染逻辑、使用 photo_url 直接渲染图片、相似度分数显示

##### 3.4.1.3 函数：checkIndexStatus()

- 功能：轮询检查索引状态
- 输入：无
- 输出：无
- 注释要求：需说明轮询间隔（2秒）、状态更新逻辑、停止轮询条件

### 3.5 config.py

职责：配置文件，管理API密钥、模型参数、服务器配置等

##### 3.5.1 配置项

**路径配置**
- PHOTO_DIR: str - 照片目录路径，从环境变量PHOTO_DIR读取
- DATA_DIR: str - 数据存储目录，默认"./data"

**API配置**
- OPENROUTER_API_KEY: str - OpenRouter API密钥，从环境变量OPENROUTER_API_KEY或OPENAI_API_KEY读取（优先OPENROUTER_API_KEY）
- OPENROUTER_BASE_URL: str - OpenRouter API地址，默认"https://openrouter.ai/api/v1"
- VISION_MODEL_NAME: str - Vision模型名称，默认"openai/gpt-4o"（通过OpenRouter接入）
- EMBEDDING_MODEL_NAME: str - 嵌入模型名称，默认"sentence-t5-base"（本地T5模型）
- TIME_PARSE_MODEL_NAME: str - 时间解析使用的LLM模型名称，默认"openai/gpt-3.5-turbo"（通过OpenRouter）

**模型配置说明**：
- **Vision LLM（图像描述）**：使用OpenRouter接入GPT-4 Vision模型
  - 模型：openai/gpt-4o（GPT-4 Omni，支持视觉，性价比高）
  - 备选：openai/gpt-4-turbo（GPT-4 Turbo，速度更快）
  - 优势：高质量图像理解，支持中文输出
- **Embedding（文本嵌入）**：使用本地T5模型
  - 模型：sentence-t5-base（768维，中文支持好）
  - 备选：sentence-t5-large（768维，效果更佳）
  - 优势：本地运行，无API成本，速度快

**服务器配置**
- SERVER_HOST: str - 服务器主机地址，默认"localhost"
- SERVER_PORT: int - 服务器端口，默认5000

**处理参数**
- BATCH_SIZE: int - 批处理大小，默认10
- MAX_RETRIES: int - 最大重试次数，默认3
- TIMEOUT: int - 超时时间（秒），默认30
- TOP_K: int - 默认返回结果数量，默认10

##### 3.5.2 图片访问URL说明

本系统通过本地HTTP服务提供图片访问，供前端和Vision LLM使用。

**URL格式**：
```
http://localhost:5000/photo?path=<图片绝对路径>
```

**示例**：
```
http://localhost:5000/photo?path=C:/Users/Photos/2023/vacation.jpg
```

**注意事项**：
- 路径必须为绝对路径
- 路径需要经过URL编码处理（使用urllib.parse.quote）
- 此接口仅供本地演示使用，不适用于生产环境
- 包含路径遍历安全检查


### 3.6 main.py

职责：应用入口，初始化服务和启动Web服务器

##### 3.6.1 函数：main()

- 功能：初始化服务并启动Flask应用
- 输入：无
- 输出：无
- 注释要求：需说明依赖注入的方式、服务初始化顺序、Flask配置（host、port、debug）

##### 3.6.2 函数：create_app()

- 功能：创建Flask应用实例，初始化所有服务和路由
- 输入：
  - indexer: Indexer - 索引构建器实例（已注入Vision/Embedding/VectorStore）
  - searcher: Searcher - 检索器实例（已注入Embedding/VectorStore）
  - config: dict - 配置字典（包含SERVER_HOST、SERVER_PORT等）
- 输出：Flask - Flask应用实例
- 注释要求：需说明依赖注入、路由注册、CORS配置、错误处理中间件

**实现策略**：
1. 创建Flask应用实例
2. 注册蓝图（如果使用）或直接注册路由
3. 配置CORS（如需要）
4. 配置错误处理
5. 将indexer和searcher实例通过Flask的g对象或闭包传递给路由

**路由注册机制**：
- 使用模块化路由：从api/routes.py导入路由函数
- 通过闭包或Flask.current_app共享indexer和searcher实例
- 路由列表：
  - GET/POST /init_index -> 触发索引构建
  - POST /search_photos -> 执行搜索
  - GET /index_status -> 获取索引状态
  - GET /photo -> 返回图片文件
  - GET / -> 渲染前端页面（index.html）

**实现示例**：
```python
from flask import Flask
from api.routes import register_routes

def create_app(indexer: Indexer, searcher: Searcher, config: dict) -> Flask:
    """
    创建并配置Flask应用
    
    Args:
        indexer: 索引构建器实例
        searcher: 检索器实例
        config: 配置字典
        
    Returns:
        Flask: 配置好的Flask应用实例
    """
    app = Flask(__name__)
    app.secret_key = config.get('SECRET_KEY', 'dev-secret-key')
    
    # 注册路由，传入indexer和searcher实例
    register_routes(app, indexer, searcher)
    
    # 配置错误处理
    @app.errorhandler(404)
    def not_found(error):
        return {"status": "error", "message": "接口不存在"}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {"status": "error", "message": "服务器内部错误"}, 500
    
    return app
```

**api/routes.py 中的register_routes实现**：
```python
from flask import jsonify, request, send_file

def register_routes(app: Flask, indexer: Indexer, searcher: Searcher):
    """
    注册所有API路由
    
    Args:
        app: Flask应用实例
        indexer: 索引构建器实例
        searcher: 检索器实例
    """
    
    @app.route('/')
    def index():
        """渲染前端页面"""
        return render_template('index.html')
    
    @app.route('/init_index', methods=['POST'])
    def init_index():
        """初始化照片索引"""
        result = indexer.build_index()
        return jsonify(result)
    
    @app.route('/search_photos', methods=['POST'])
    def search_photos():
        """执行照片搜索"""
        data = request.get_json()
        query = data.get('query', '')
        top_k = min(data.get('top_k', 10), 50)
        
        results = searcher.search(query, top_k)
        return jsonify(results)
    
    @app.route('/index_status', methods=['GET'])
    def index_status():
        """获取索引状态"""
        return jsonify(indexer.get_status())
    
    @app.route('/photo')
    def get_photo():
        """返回图片文件"""
        path = request.args.get('path')
        if not path or not os.path.isabs(path) or not os.path.exists(path):
            return jsonify({"status": "error", "message": "图片文件不存在"}), 404
        
        # 安全检查：防止路径遍历攻击
        if '..' in path or not path.startswith(config.get('PHOTO_DIR')):
            return jsonify({"status": "error", "message": "非法的文件路径"}), 403
        
        return send_file(path)
```

##### 3.6.3 函数：initialize_services(config: dict) -> tuple

- 功能：初始化所有服务实例（Vision LLM、Embedding、VectorStore、Indexer、Searcher）
- 输入：config: dict - 配置字典
- 输出：tuple - (indexer: Indexer, searcher: Searcher)
- 注释要求：需说明服务初始化顺序、依赖注入、错误处理

**实现示例**：
```python
def initialize_services(config: dict) -> tuple:
    """
    初始化所有服务实例
    
    Args:
        config: 配置字典
        
    Returns:
        tuple: (indexer, searcher)
    """
    # 初始化Vision LLM服务（使用OpenRouter）
    vision_service = OpenAIVisionLLMService(
        api_key=config.get('OPENROUTER_API_KEY'),
        model_name=config.get('VISION_MODEL_NAME', 'openai/gpt-4o'),
        base_url=config.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    )
    
    # 初始化Embedding服务（使用本地T5模型）
    embedding_service = T5EmbeddingService(
        model_name=config.get('EMBEDDING_MODEL_NAME', 'sentence-t5-base')
    )
    
    # 初始化时间解析器（使用OpenRouter）
    time_parser = TimeParser(
        api_key=config.get('OPENROUTER_API_KEY'),
        model_name=config.get('TIME_PARSE_MODEL_NAME', 'openai/gpt-3.5-turbo'),
        base_url=config.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    )
    
    # 初始化向量存储
    vector_store = VectorStore(
        dimension=768,  # sentence-t5-base的维度
        index_path=config.get('INDEX_PATH'),
        metadata_path=config.get('METADATA_PATH')
    )
    
    # 初始化索引构建器
    indexer = Indexer(
        photo_dir=config.get('PHOTO_DIR'),
        vision_service=vision_service,
        embedding_service=embedding_service,
        vector_store=vector_store,
        data_dir=config.get('DATA_DIR')
    )
    
    # 初始化检索器
    searcher = Searcher(
        embedding_service=embedding_service,
        time_parser=time_parser,
        vector_store=vector_store,
        data_dir=config.get('DATA_DIR')
    )
    
    return indexer, searcher
```
        api_key=config['OPENAI_API_KEY'],
        model_name=config.get('VISION_MODEL_NAME'),
        server_host=config['SERVER_HOST'],
        server_port=config['SERVER_PORT']
    )
    
    # 初始化Embedding服务
    embedding_service = OpenAIEmbeddingService(
        api_key=config['OPENAI_API_KEY'],
        model_name=config.get('EMBEDDING_MODEL_NAME')
    )
    
    # 初始化TimeParser服务（用于时间约束解析）
    time_parser = TimeParser(
        api_key=config['OPENAI_API_KEY'],
        model_name=config.get('TIME_PARSE_MODEL_NAME', 'gpt-3.5-turbo')
    )
    
    # 初始化VectorStore（首次构建时维度为None，后续从索引加载）
    vector_store = VectorStore(
        dimension=config.get('EMBEDDING_DIMENSION'),
        index_path=os.path.join(config['DATA_DIR'], 'photo_search.index'),
        metadata_path=os.path.join(config['DATA_DIR'], 'metadata.json')
    )
    
    # 初始化Indexer
    indexer = Indexer(
        photo_dir=config['PHOTO_DIR'],
        vision_service=vision_service,
        embedding_service=embedding_service,
        vector_store=vector_store,
        data_dir=config['DATA_DIR']
    )
    
    # 初始化Searcher
    searcher = Searcher(
        embedding_service=embedding_service,
        vector_store=vector_store,
        time_parser=time_parser,
        data_dir=config['DATA_DIR']
    )
    
    return indexer, searcher
```

##### 3.6.4 main() 完整执行流程

**实现示例**：
```python
def main():
    """
    主函数：初始化服务并启动Flask应用
    """
    # 加载环境变量
    load_dotenv()
    
    # 加载配置
    config = {
        'PHOTO_DIR': os.getenv('PHOTO_DIR'),
        'DATA_DIR': os.getenv('DATA_DIR', './data'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'VISION_MODEL_NAME': os.getenv('VISION_MODEL_NAME'),  # 待用户配置
        'EMBEDDING_MODEL_NAME': os.getenv('EMBEDDING_MODEL_NAME'),  # 待用户配置
        'EMBEDDING_DIMENSION': int(os.getenv('EMBEDDING_DIMENSION', '1536')),
        'SERVER_HOST': os.getenv('SERVER_HOST', 'localhost'),
        'SERVER_PORT': int(os.getenv('SERVER_PORT', '5000')),
        'BATCH_SIZE': int(os.getenv('BATCH_SIZE', '10')),
        'MAX_RETRIES': int(os.getenv('MAX_RETRIES', '3')),
        'TIMEOUT': int(os.getenv('TIMEOUT', '30')),
        'TOP_K': int(os.getenv('TOP_K', '10')),
        'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret-key')
    }
    
    # 验证必要配置
    if not config['PHOTO_DIR']:
        raise ValueError("PHOTO_DIR环境变量未设置")
    if not config['OPENAI_API_KEY']:
        raise ValueError("OPENAI_API_KEY环境变量未设置")
    if not config['VISION_MODEL_NAME']:
        raise ValueError("VISION_MODEL_NAME环境变量未设置")
    if not config['EMBEDDING_MODEL_NAME']:
        raise ValueError("EMBEDDING_MODEL_NAME环境变量未设置")
    
    # 创建数据目录
    os.makedirs(config['DATA_DIR'], exist_ok=True)
    
    # 初始化服务
    indexer, searcher = initialize_services(config)
    
    # 创建Flask应用
    app = create_app(indexer, searcher, config)
    
    # 启动服务器
    print(f"启动服务器: http://{config['SERVER_HOST']}:{config['SERVER_PORT']}")
    app.run(
        host=config['SERVER_HOST'],
        port=config['SERVER_PORT'],
        debug=False
    )

if __name__ == '__main__':
    main()
```

## 4. 内部接口说明

### 4.1 接口：build_index

- 调用场景：用户点击"初始化索引"按钮触发
- 调用者：api/routes.py -> core/indexer.py
- 请求参数：无
- 响应格式：
```json
{
  "status": "success/processing/failed",
  "message": "索引构建中/成功/失败",
  "indexed_count": 100,
  "total_count": 105,
  "failed_count": 5,
  "fallback_ratio": 0.08,
  "index_path": "./data/photo_search.index",
  "elapsed_time": 150.5
}
```
- 设计意图：使用后台线程异步构建索引，避免长时间阻塞HTTP请求；若成功数<100或降级占比>=10%，返回failed

### 4.2 接口：search_photos

- 调用场景：用户输入查询文本并点击搜索按钮
- 调用者：api/routes.py -> core/searcher.py
- 请求参数：
```json
{
  "query": "去年夏天在海边的照片",
  "top_k": 10
}
```
- 响应格式：
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
- 设计意图：返回 photo_url 由前端加载图片，避免在搜索接口中传输大文件
- 设计意图：如查询包含时间约束，优先基于EXIF时间过滤，其次文件时间

### 4.3 接口：get_photo

- 调用场景：前端渲染搜索结果图片、Vision LLM访问图片
- 调用者：templates/index.html, utils/vision_llm_service.py
- 请求参数：
```
GET /photo?path=/photos/vacation_beach_001.jpg
```
- 响应格式：图片二进制内容
- 设计意图：仅用于本地演示，方便直接渲染图片；同时供Vision LLM通过URL访问，避免Base64编码的token消耗

### 4.4 接口：generate_description（内部）

- 调用场景：Indexer Agent为每张照片生成描述
- 调用者：core/indexer.py -> utils/vision_llm_service.py
- 请求参数：
```json
{
  "image_path": "/photos/vacation_beach_001.jpg"
}
```
- 响应格式：
```json
{
  "description": "一群人在阳光明媚的海滩上玩耍",
  "model_used": "TBD",
  "elapsed_time": 2.5
}
```
- 设计意图：内部接口，仅允许indexer模块调用

### 4.5 接口：generate_embedding（内部）

- 调用场景：将描述文本转换为向量
- 调用者：core/indexer.py, core/searcher.py -> utils/embedding_service.py
- 请求参数：
```json
{
  "text": "一群人在阳光明媚的海滩上玩耍"
}
```
- 响应格式：
```json
{
  "embedding": [0.123, -0.456, ...],
  "dimension": "TBD"
}
```
- 设计意图：内部接口，支持批量处理以提高效率

### 4.6 接口：add_item（内部）

- 调用场景：索引构建时添加向量到FAISS
- 调用者：core/indexer.py -> utils/vector_store.py
- 请求参数：
```json
{
  "embedding": [0.123, -0.456, ...],
  "metadata": {
    "photo_path": "/photos/vacation_beach_001.jpg",
    "description": "一群人在阳光明媚的海滩上玩耍",
    "exif_data": {
      "datetime": "2023-07-15T10:20:30",
      "location": null,
      "camera": "未知",
      "orientation": 1
    },
    "file_time": "2023-07-16T09:00:00"
  }
}
```
- 响应格式：
```json
{
  "success": true,
  "current_total": 100
}
```
- 设计意图：内部接口，仅允许indexer模块调用

### 4.7 接口：search_vector（内部）

- 调用场景：执行相似度检索时调用
- 调用者：core/searcher.py -> utils/vector_store.py
- 请求参数：
```json
{
  "query_embedding": [0.123, -0.456, ...],
  "top_k": 10
}
```
- 响应格式：
```json
{
  "results": [
    {
      "metadata": {
        "photo_path": "/photos/vacation_beach_001.jpg",
        "description": "一群人在阳光明媚的海滩上玩耍"
      },
      "score": 0.95
    }
  ]
}
```
- 设计意图：内部接口，仅允许searcher模块调用


## 5. 代码注释要求

### 5.1 函数/类顶部注释

函数Docstring模板：
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    [功能描述]
    
    Args:
        param1 (type): [参数说明]
        param2 (type): [参数说明]
    
    Returns:
        return_type: [返回值说明]
    
    Raises:
        Exception: [可能抛出的异常]
    """
```

类Docstring模板：
```python
class ClassName:
    """
    [类的核心职责描述]
    
    Attributes:
        attr1 (type): [属性说明]
        attr2 (type): [属性说明]
    """
```

### 5.2 关键逻辑行注释

- 导入库选择：
```python
import faiss  # 使用FAISS库进行高效向量检索
```

- 数据格式转换：
```python
image_url = f"http://localhost:{port}/photo?path={urllib.parse.quote(image_path)}"  # 生成图片URL供Vision LLM访问
```

- 错误处理：
```python
try:
    description = self.vision_llm_service.generate_description(photo_path)
except Exception as e:
    logger.error(f"生成描述失败: {e}")  # 捕获API调用失败，避免中断整个索引构建流程
```

- 性能优化：
```python
batch_size = 32  # 批量处理以减少API调用次数
```

- 算法选择：
```python
self.index = faiss.IndexFlatL2(dimension)  # 使用L2距离计算相似度
```

- 锁文件管理：
```python
if not self._create_lock():
    return {"status": "failed", "message": "索引构建正在进行中"}  # 防止并发构建
```

- 向量维度校验：
```python
if len(embedding) != self.dimension:
    raise ValueError(f"向量维度不匹配: {len(embedding)} != {self.dimension}")  # 确保所有向量维度一致
```

### 5.3 接口定义处注释

```python
# 内部接口：仅允许indexer模块调用，禁止直接暴露给前端
def add_item(embedding: List[float], metadata: Dict) -> None:
    pass

# 内部接口：仅允许searcher模块调用，禁止前端直接访问向量数据库
def search(query_embedding: List[float], top_k: int) -> List[Dict]:
    pass

# API接口：用于前端触发索引构建
def init_index():
    pass

# API接口：用于前端执行搜索查询
def search_photos():
    pass
```

## 6. 实现约束

### 6.1 禁止事项
- 禁止实现用户系统、权限控制等扩展功能
- 禁止使用分布式索引或多线程优化
- 禁止在demo中实现复杂的缓存机制
- 禁止添加非必需的日志系统
- 禁止直接暴露内部接口给前端

### 6.2 必做事项
- 所有函数必须包含类型注解
- 所有外部依赖必须在requirements.txt中声明
- 错误信息必须使用中文，便于用户理解
- 向量维度必须与嵌入模型输出维度一致
- 所有API调用必须有超时和重试机制
- 文件操作必须有异常处理
 - 索引成功数量必须>=100，否则标记为failed
 - 降级描述占比必须<10%，否则标记为failed

### 6.3 测试覆盖要求

#### 6.3.1 单元测试覆盖

必须测试的函数和类：

image_parser.py：
- is_valid_image() - 测试有效和无效图片格式
- extract_exif_metadata() - 测试有EXIF和无EXIF的图片
- generate_fallback_description() - 测试不同文件名格式

vision_llm_service.py：
- OpenAIVisionLLMService.generate_description() - 测试API调用成功和失败
- LocalVisionLLMService.generate_description() - 测试本地模型推理

embedding_service.py：
- OpenAIEmbeddingService.generate_embedding() - 测试向量生成
- SentenceTransformerService.generate_embedding() - 测试本地模型推理

vector_store.py：
- VectorStore.add_item() - 测试添加单个和多个向量
- VectorStore.search() - 测试相似度检索
- VectorStore.save() / load() - 测试索引持久化

indexer.py：
- Indexer.scan_photos() - 测试目录扫描
- Indexer.process_batch() - 测试批处理
- Indexer.build_index() - 测试完整索引构建流程

searcher.py：
- Searcher.load_index() - 测试索引加载
- Searcher.search() - 测试检索功能
- Searcher.validate_query() - 测试查询验证

#### 6.3.2 集成测试覆盖

必须测试的场景：

1. 索引构建完整流程：
   - 扫描照片目录
   - 生成描述
   - 计算嵌入
   - 存储索引
   - 创建就绪标记

2. 检索完整流程：
   - 等待索引就绪
   - 加载索引
   - 执行查询
   - 返回结果

3. 双Agent协作：
   - Indexer构建索引
   - Searcher自动加载新索引
   - 检索功能正常

#### 6.3.3 测试数据要求

准备测试图片：
- 至少10张不同场景的图片（人物、风景、食物、动物等）
- 包含有EXIF和无EXIF的图片
- 包含不同格式的图片（JPG、PNG、WEBP）

准备测试查询：
- 场景查询："山上的照片"、"城市夜景"
- 人物查询："有人物的照片"、"集体合影"
- 活动查询："聚餐"、"运动"
- 时间查询："去年的照片"（如果EXIF可用）
- 情感查询："欢乐的场景"

