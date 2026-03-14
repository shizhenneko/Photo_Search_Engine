# 开发说明

## 当前运行架构

项目已经收敛到单一供应商路径，避免多套运行时并存：

- SU8：视觉描述、时间解析、QueryFormatter、视觉 rerank
- Tumuer Router：embedding、文本 rerank
- FAISS：本地向量索引
- Elasticsearch：可选混合检索与过滤

历史上的其他供应商与本地 embedding 路径已经从主流程移除。

## 推荐开发流程

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
./.venv/bin/python -m pytest -q
./.venv/bin/python main.py
```

## 检索流程

### 索引阶段

1. 遍历 `PHOTO_DIR`
2. 优先加载当前已有索引与元数据
3. 只筛出尚未入库的新图片
4. SU8 Vision 生成结构化分析与 `retrieval_text`
5. 提取 EXIF 元数据，并仅使用 EXIF `datetime` 生成时间标签
6. Tumuer `Qwen/Qwen3-Embedding-8B` 生成向量
7. 追加写入 FAISS
8. 可选写入 Elasticsearch

### 查询阶段

1. QueryFormatter 提取纯视觉语义与时间提示
2. TimeParser 解析结构化时间约束
3. 第一轮执行基础检索
4. 如果第一轮结果偏弱，触发少量意图扩展
5. 如果扩展后仍偏弱，触发单轮反思
6. 如果是纯过滤查询，直接走过滤检索
7. 否则执行向量检索或混合检索
8. 可选先做文本 rerank，再做视觉 rerank

## 关键实现约束

- 向量检索只基于纯视觉描述，不混入时间字段
- 没有 EXIF `datetime` 的图片不能被推导出季节、时段、年月标签
- `cosine` 模式下向量必须 L2 归一化
- BM25 分数必须在融合前归一到 `0-1`
- 融合时只按真实命中的检索通道重归一，不能因为无 BM25 命中而压低向量结果
- 日常新增图片走增量索引，只有模型或索引结构变化时才强制全量重建
- 文本检索扩展与反思只应在弱结果时介入，不能伤害强首轮结果
- 路径输出必须兼容 Windows 与 WSL 双环境

## 索引入口约定

- 前端显式提供 `增量索引` 与 `全量重建` 两个按钮
- `POST /init_index` 默认等价于 `{"mode":"incremental"}`
- `POST /init_index` 传入 `{"mode":"full"}` 时，会调用 `Indexer.build_index(force_rebuild=True)`
- 日常加图只补新图；只有 embedding 模型、结构化分析字段、`retrieval_text` 生成逻辑或 Elasticsearch mapping 变化时才全量重建
