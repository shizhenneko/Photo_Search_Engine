# Demo 开发文档

本文档只记录当前已落地、可运行的方案。

## 环境准备

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env
```

`.env` 中的核心配置：

```bash
PHOTO_DIR=C:/Users/yourname/Desktop/test_photos
SU8_API_KEY=your-su8-key
SU8_BASE_URL=https://www.su8.codes/codex/v1
VISION_MODEL=gpt-5.4
TIME_PARSE_MODEL=gpt-5.1

EMBEDDING_API_KEY=your-tumuer-key
EMBEDDING_BASE_URL=https://router.tumuer.me/v1
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
TEXT_RERANK_MODEL=Qwen/Qwen3-Reranker-8B
```

## 启动

```bash
./.venv/bin/python main.py
```

如果默认端口冲突：

```bash
SERVER_PORT=10002 ./.venv/bin/python main.py
```

如果你在 Windows + WSL 环境下做本地演示，推荐直接使用单一入口脚本：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\86159\Desktop\Photo_Search_Engine\artifacts\start_stack.ps1"
```

补充说明：

- 脚本会自动选择前端端口，从 `10001` 开始向后避让
- Elasticsearch 会固定使用 `1g/1g` 堆，避免 Windows 上自动堆过大导致启动失败
- 运行完成后，终端会打印最终的前端 URL 和 Elasticsearch URL
- 运行日志和状态文件会写入 `artifacts/runtime/`
- 现在只保留一个启动入口，不再需要分别手动启动 Elasticsearch 和 Flask
- 如需诊断当前服务状态，可运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\86159\Desktop\Photo_Search_Engine\artifacts\check_services.ps1"
```

## 已实现功能

- 文本搜图
- 以图搜图
- 上传图片搜图
- 可选文本 rerank
- 可选视觉 rerank
- 返回图片描述、图片路径、预览地址
- 定位本地文件
- 结果区显示检索规划

## 技术实现

### 视觉描述

`utils/vision_llm_service.py`

- 使用 SU8 兼容 OpenAI 的 `chat.completions`
- 图片本地读取后压缩并转成 base64 data URL
- 输出中文、可检索的描述文本

### 向量生成

`utils/embedding_service.py`

- 使用 Tumuer Router 的 OpenAI 兼容 embeddings 接口
- 当前模型为 `Qwen/Qwen3-Embedding-8B`
- 向量维度来自配置，默认 `4096`

### 混合检索

`core/searcher.py`

- 向量召回与关键词召回分别计算
- 向量分数与 BM25 分数都会先归一化
- 只有真实命中的检索通道才参与融合，再按可用权重重归一
- 第一轮弱结果时才触发查询扩展
- 第二轮仍弱时才触发一次反思式调整

这意味着：

- 只有向量命中的图片，不会因为没有 BM25 命中而被额外压分
- 同时命中向量和 BM25 的结果，仍然会按配置权重融合
- 基础首轮结果强时，不会被扩展逻辑干扰

### 增量索引

`core/indexer.py`

- 前端提供 `增量索引` 与 `全量重建` 两个按钮
- `POST /init_index` 默认或 `mode=incremental` 时优先加载当前索引与元数据
- 只处理尚未入库的新图片
- 没有新增图片时直接返回“已是最新”
- `mode=full` 时会清空现有索引并全量重建

这意味着日常新增照片后，应优先使用“增量索引”，不需要每次清空索引重建。

### 时间标签

索引阶段只使用 EXIF 拍摄时间生成结构化时间标签。

- 没有 EXIF `datetime` 的图片，不会生成 `season`、`time_period`、`year`、`month` 等标签
- `file_time` 不再用于推导时间标签，避免图片被文件修改时间误标
- 这类图片仍然可以参与普通文本检索、以图搜图和 rerank
- 但在“去年”“夏天”“傍晚”这类时间过滤查询中，不会被错误命中

### 双阶段重排

1. Tumuer `Qwen/Qwen3-Reranker-8B`
2. SU8 Vision 模型做视觉 rerank

两者都由前端复选框显式控制，不默认强开。

### 检索规划面板

前端会展示检索规划，包含：

- 基础意图
- 是否触发扩展
- 是否触发反思
- 每轮 top score 与结果数量

## 前端说明

前端为服务端模板 `templates/index.html`，当前提供：

- 文本搜图与以图搜图双 Tab
- 上传图片入口
- 索引状态卡片
- 检索规划面板
- 结果卡片中的路径展示
- 查看原图、复制路径、定位文件、搜索相似

## 验收命令

```bash
./.venv/bin/python -m pytest -q
curl -s http://127.0.0.1:10002/index_status
curl -s -X POST http://127.0.0.1:10002/search_photos -H 'Content-Type: application/json' -d '{"query":"去年夏天的海边照片","top_k":5}'
```

如果你刚升级到当前版本，且旧索引曾经使用 `file_time` 推导时间标签，建议立即执行一次“全量重建”。

如果只是日常新增图片，直接点击“增量索引”即可，它会从当前状态继续补齐新图。

如果修改了 embedding 模型、结构化字段、`retrieval_text` 生成逻辑或 Elasticsearch mapping，请使用“全量重建”。
