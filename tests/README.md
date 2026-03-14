# 测试指南

本仓库当前以 `pytest` 为主，推荐统一使用 `uv` 环境执行。

## 测试范围

- 纯单元测试：检索逻辑、路径处理、向量归一化、路由行为
- 轻量集成测试：使用 fake service 验证 rerank、以图搜图、索引流程
- 增量索引测试：验证新增图片只补索引、不重复处理旧图片
- 真实 API 验证：不纳入默认 pytest，需要启动服务后手动调用 HTTP 接口

默认测试不会调用真实 SU8 或 Tumuer API，因此不会消耗线上额度。

## 运行方式

先准备环境：

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

运行全量测试：

```bash
./.venv/bin/python -m pytest -q
```

运行指定文件：

```bash
./.venv/bin/python -m pytest tests/test_routes.py -q
./.venv/bin/python -m pytest tests/test_searcher.py -q
```

## 重点测试文件

- `tests/test_routes.py`：接口返回结构、结果中的 `photo_path` / `photo_url`
- `tests/test_searcher.py`：混合检索、纯过滤查询、以图搜图、时间过滤边界
- `tests/test_query_formatter.py`：查询理解、扩展轮次、反思轮次
- `tests/test_embedding_service.py`：Tumuer embedding 与文本 rerank 服务
- `tests/test_time_parser.py`：SU8 时间解析
- `tests/test_vector_store.py`：FAISS 存储与归一化行为
- `tests/test_indexer.py`：索引阶段时间标签生成策略与增量索引行为
- `tests/test_path_utils.py`：Windows/WSL 路径兼容

## 手动验收建议

真实验收建议按顺序执行：

1. 启动应用
2. 调用 `POST /init_index` 的增量模式，验证新增图片补索引
3. 调用 `POST /init_index` 的全量模式，验证重建流程
4. 测试 `POST /search_photos`
5. 测试 `POST /search_by_image`
6. 测试 `POST /search_by_uploaded_image`
7. 分别开启文本 rerank 与视觉 rerank
8. 验证检索规划面板是否和后端 `search_debug` 一致
9. 验证 `/photo` 预览与 `/open_photo_location` 本地定位

推荐的索引接口调用方式：

```bash
curl -X POST http://127.0.0.1:10001/init_index \
  -H 'Content-Type: application/json' \
  -d '{"mode":"incremental"}'

curl -X POST http://127.0.0.1:10001/init_index \
  -H 'Content-Type: application/json' \
  -d '{"mode":"full"}'
```

## 注意事项

- `POST /init_index` 省略 `mode` 时默认按增量索引处理
- 切换 `EMBEDDING_MODEL` 后必须全量重建索引；日常新增图片则应走增量索引
- 如果 Elasticsearch 未启动，部分关键词过滤会退化为内存过滤，但测试仍应通过
- 当前时间标签只来自 EXIF `datetime`，无 EXIF 图片不会生成季节/时段标签
- 混合检索不会因为缺少 BM25 命中而惩罚纯向量命中的结果
- 如果本机 `10001` 被占用，可用 `SERVER_PORT=10002` 启动服务进行手动验收
