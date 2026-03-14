# 照片搜索引擎

基于 Flask 的本地照片搜索系统，当前运行架构已经统一为：

- gpt: 视觉描述、时间解析、视觉 Rerank、QueryFormatter
- `Qwen/Qwen3-Embedding-8B` 向量生成、`Qwen/Qwen3-Reranker-8B` 文本 Rerank
- FAISS：向量索引
- Elasticsearch：可选的关键词与时间过滤检索

系统支持文本搜图、以图搜图、上传图片搜图、文本重排、视觉重排，并在结果中返回图片描述、本地文件路径、预览链接和定位本地文件动作。

当前版本已经升级为结构化多信号检索：

- 不再只依赖单段 caption
- 每张图会提取媒介类型、标签、OCR 文本、人物角色、公众人物候选和 `retrieval_text`
- 检索结果会返回 `match_summary`，用于解释命中原因
- 文本检索会保留基础召回，并在弱结果时触发保守扩展与单轮反思
- 前端会显示“检索规划”，用于观察基础意图、扩展轮次与反思轮次

## 当前能力

- 文本搜图：自然语言检索本地图库
- 以图搜图：使用已入库图片路径检索相似图片
- 上传图片搜图：使用临时上传图片分析结果生成 embedding，再检索相似图片
- 混合检索：向量检索 + BM25 检索融合
- 结构化图像理解：外层场景、内层内容、媒介类型、标签、OCR、人物候选
- 时间过滤：支持“去年”“夏天”“傍晚”等语义过滤
- 多轮查询理解：基础意图 -> 弱结果扩展 -> 弱结果反思
- 双阶段重排：先文本 rerank，再视觉 rerank
- 本地操作：返回路径、预览原图、打开文件所在位置

## 检索与归一化

当前混合检索已经做归一化，融合逻辑是正确实现的：

- `utils/vector_store.py`：在 `cosine` 模式下先做 L2 归一化，再使用 `IndexFlatIP`
- `utils/keyword_store.py`：把 Elasticsearch BM25 分数按每次查询的 `max_score` 归一到 `0-1`
- `core/searcher.py`：只有在某一路检索信号真实命中时才参与融合，再按可用权重重归一

当前向量检索使用 `retrieval_text`，它由高置信 `media_types`、`top_tags`、`outer_scene_summary`、`inner_content_summary`、`ocr_text` 和 `identity_names` 组合而成。

这意味着：

- 向量命中但没有 BM25 命中的图片，不会因为 `keyword_score=0` 被平白压分
- 同时命中向量和 BM25 的结果，仍然会按 `VECTOR_WEIGHT` / `KEYWORD_WEIGHT` 融合
- 只有 BM25 命中的结果，也可以独立进入排序结果

如果未启用 Elasticsearch，系统会自动退化为纯向量检索，并对时间过滤走内存元数据匹配。

## 查询理解与检索规划

当前文本检索不是单轮黑盒改写，而是保守分层：

1. 第一轮：保留用户原始意图，提取 `search_text`、`media_terms`、`identity_terms` 与时间提示
2. 第二轮：只有当第一轮结果偏弱时，才触发少量替代意图扩展
3. 第三轮：如果第二轮仍偏弱，再基于弱结果做一次反思式调整

这个流程的目标是提高泛化能力，同时不破坏基础召回。高分首轮结果不会被强行改写。

前端结果区上方会展示 `search_debug` 对应的“检索规划”面板，包含：

- 基础意图
- 是否触发扩展
- 是否触发反思
- 各轮 top score 与结果数量

## 时间元数据策略

当前版本只使用 **EXIF 拍摄时间** 生成结构化时间标签：

- `year`
- `month`
- `day`
- `season`
- `time_period`
- `weekday`

`file_time` 不再用于生成这些标签，也不会再用于 ES 时间过滤。这样做的目的是避免没有 EXIF 时间的图片因为文件修改时间被错误打上“夏天”“傍晚”“2025年”等标签。

这意味着：

- 没有 EXIF `datetime` 的图片，不会被错误标记时间信息
- 这类图片仍然可以参与普通向量检索和以图搜图
- 但在“去年”“夏天”“傍晚”这类时间过滤查询中，它们不会被时间条件错误命中

如果你修改了时间标签策略、Embedding 模型、Rerank 模型、结构化分析 prompt、检索字段或 Elasticsearch mapping，应该重新构建索引。

## 索引策略

当前索引构建已经支持增量更新：

- 前端提供两个入口：`增量索引` 和 `全量重建`
- `POST /init_index` 支持通过 JSON 参数 `mode` 选择索引方式
- `mode=incremental` 或省略 `mode` 时，会优先加载现有 FAISS 索引与元数据
- 只对新增图片生成结构化分析与 embedding
- 已存在的图片不会重复分析，也不会重复写入
- `mode=full` 时会清空现有索引并执行全量重建

这意味着，日常新增照片后应该优先使用“增量索引”，它会从当前状态补齐新图，而不是每次都重扫并重建全部图片。

仍然需要全量重建的场景只有：

- 切换 `EMBEDDING_MODEL`
- 修改结构化分析字段或 `retrieval_text` 生成逻辑
- 修改 Elasticsearch mapping
- 旧索引文件损坏或与当前维度 / metric 不兼容

## 快速开始

### 1. 使用 uv 创建环境

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

### 2. 配置 `.env`

参考 `.env.example`

默认模型：

```bash
VISION_MODEL=gpt-5.4
STRUCTURED_ANALYSIS_ENABLED=true
ENHANCED_ANALYSIS_ENABLED=true
TAG_MIN_CONFIDENCE=0.65
IDENTITY_TEXT_MIN_CONFIDENCE=0.70
IDENTITY_VISUAL_MIN_CONFIDENCE=0.92
TIME_PARSE_MODEL=gpt-5.1
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
TEXT_RERANK_MODEL=Qwen/Qwen3-Reranker-8B
```

### 3. 启动服务

```bash
./.venv/bin/python main.py
```

如果 `10001` 端口被占用，可临时切换：

```bash
SERVER_PORT=10002 ./.venv/bin/python main.py
```

### Windows 本地启动

如果你是在 Windows + WSL 环境下运行，推荐直接使用单一入口脚本。你只需要运行一次脚本，脚本会自动完成以下动作：

- 自动选择前端端口，从 `10001` 开始向后寻找空闲端口，例如 `10001 -> 10002`
- 以 `-Xms1g -Xmx1g` 启动 Elasticsearch，避免 Windows 上默认自动堆过大导致 JVM 启动失败
- 启动 WSL 中的 Flask 前端
- 输出最终可访问的前端地址和 Elasticsearch 地址
- 把运行日志和状态文件写入 `artifacts/runtime/`

运行方式：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\86159\Desktop\Photo_Search_Engine\artifacts\start_stack.ps1"
```

脚本执行完成后，会在终端里打印类似下面的结果：

```text
[DONE] Elasticsearch: http://127.0.0.1:9200
[DONE] Frontend: http://127.0.0.1:10001
```

补充说明：

- 如果 `10001` 已经被占用，脚本不会报错退出，而是自动改用下一个空闲端口，例如 `10002`
- 当前唯一主入口脚本是 `artifacts/start_stack.ps1`
- 如果你想查看当前 `9200`、`10001`、`10002` 的监听和 HTTP 状态，可以额外执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\86159\Desktop\Photo_Search_Engine\artifacts\check_services.ps1"
```

### 4. 构建索引

启动后访问：

- `http://127.0.0.1:10001/`

前端按钮说明：

- `增量索引`：日常新增图片后补齐当前索引
- `全量重建`：模型、索引结构或 mapping 发生变化后重建全部索引

也可以直接调用接口。

增量索引：

```bash
curl -X POST http://127.0.0.1:10001/init_index \
  -H 'Content-Type: application/json' \
  -d '{"mode":"incremental"}'
```

全量重建：

```bash
curl -X POST http://127.0.0.1:10001/init_index \
  -H 'Content-Type: application/json' \
  -d '{"mode":"full"}'
```

注意：

- 省略 `mode` 时默认按增量索引处理
- 日常新增图片时，应优先使用 `增量索引`
- 切换 embedding 模型后仍然必须执行 `全量重建`

如果你刚更新到当前版本，必须立即重建索引，因为：

- 向量输入已从 `description` 切换为 `retrieval_text`
- 元数据新增了结构化分析字段
- Elasticsearch mapping 新增了 `retrieval_text`、`media_types`、`identity_names` 等字段
- 旧索引不会自动迁移

## 主要接口

- `POST /init_index`：建立或更新图片索引，支持 `{"mode":"incremental"}` 与 `{"mode":"full"}`
- `GET /index_status`：查看索引状态
- `POST /search_photos`：文本搜图
- `POST /search_by_image`：以图搜图
- `POST /search_by_uploaded_image`：上传图片搜图
- `POST /open_photo_location`：定位本地文件
- `GET /photo?path=...`：预览原图

搜索结果中的每条记录还会返回 `match_summary`，包含：

- `media_types`
- `top_tags`
- `identities`
- `identity_evidence`
- `ocr_excerpt`

文本搜索、以图搜图和上传图片搜图的响应中还会包含 `search_debug`，用于展示检索规划和轮次信息。

## 开发与测试

运行完整测试：

```bash
./.venv/bin/python -m pytest -q
```

运行单文件测试：

```bash
./.venv/bin/python -m pytest tests/test_routes.py -q
```

## 项目结构

```text
api/                 HTTP 路由
core/                索引与检索核心逻辑
templates/           单页前端
tests/               单元测试与集成边界测试
utils/               模型服务、向量存储、路径处理等
config.py            环境变量配置加载
main.py              应用入口
```

## 说明

- QueryFormatter 保留启用，但默认与 SU8 复用同一套中转配置
- Elasticsearch 是可选项；未启动时系统仍可运行
- 没有 EXIF 时间的图片不会生成时间标签，但不会影响普通文本检索、以图搜图和 rerank
- 前端索引入口已经拆分为“增量索引”和“全量重建”两个按钮，默认日常使用增量索引
- Figma MCP 已尝试接入，但当前提供的设计文件对当前登录账号不可访问，因此前端优化基于现有实现继续演进，而不是直接抽取设计 token
