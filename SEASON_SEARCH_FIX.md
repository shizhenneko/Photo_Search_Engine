# 季节搜索功能修复说明

## 问题描述

用户输入"夏天的照片"时，后端显示对26个字符做了 embedding（向量化），导致搜索结果不正确。

### 问题根因

当 QueryFormatter (LLM 查询格式化服务) 启用时，存在以下问题：

1. **Prompt 设计缺陷**：QueryFormatter 的 Prompt 要求"search_text 只含视觉内容，禁止时间词"，但没有明确告诉 LLM 在纯过滤查询场景下应该返回空字符串
2. **LLM 误解**：对于"夏天的照片"这样的纯过滤查询，LLM 会尝试生成通用的视觉描述，如"照片 各种场景的摄影作品"
3. **判断逻辑失效**：Searcher 的纯过滤查询判断逻辑被 LLM 返回的通用描述干扰，误判为混合查询，导致执行向量检索

### 执行流程问题

```
用户输入: "夏天的照片"
    ↓
QueryFormatter (LLM)
    ↓
返回: { "search_text": "照片 各种场景的摄影作品", "season": "夏天" }
    ↓
Searcher 判断
    ↓
cleaned_query = "照片 各种场景的摄影作品" (非空)
    ↓
is_pure_filter = False (误判！)
    ↓
执行向量检索 (生成 embedding) ❌
    ↓
结果不准确
```

## 修复方案

采用**三层防护**策略，确保纯过滤查询正确执行：

### 第一层：优化 QueryFormatter Prompt

**文件**: `utils/query_formatter.py`

**修改内容**:
```python
# 在 Prompt 中添加明确规则和示例
规则：
- search_text只含视觉内容，禁止年月日季节时段
- **如果查询只包含过滤条件（时间/季节/时段）+"的照片"等通用词，search_text必须留空**
- **只有当查询包含具体场景/物体/人物/活动等视觉要素时，才填写search_text**

示例：
- "夏天的照片" → {"search_text": "", "season": "夏天"}
- "夏天海边的照片" → {"search_text": "海边沙滩海浪", "season": "夏天"}
- "2023年的照片" → {"search_text": "", "time_hint": "2023年"}
- "傍晚拍的照片" → {"search_text": "", "time_period": "傍晚"}
```

### 第二层：增强空值防护逻辑

**文件**: `utils/query_formatter.py`

**修改内容**:
```python
# 检查 LLM 返回的 search_text 是否只包含通用词汇
search_text = result.get("search_text", "")
if search_text and search_text.strip():
    # 移除通用词汇：照片、图片、相片、画面、场景、摄影、作品等
    test_text = search_text
    generic_patterns = [
        r"照片", r"图片", r"相片", r"画面", r"场景",
        r"摄影", r"作品", r"影像", r"各种", r"的", r"\s+"
    ]
    for pattern in generic_patterns:
        test_text = re.sub(pattern, "", test_text)
    test_text = test_text.strip()
    
    # 如果清理后为空或过短，设为空字符串（纯过滤查询）
    if len(test_text) < 2:
        search_text = ""

# 保持为空字符串，让 Searcher 的纯过滤逻辑处理
result["search_text"] = search_text
```

### 第三层：Searcher 兜底检查

**文件**: `core/searcher.py`

**修改内容**:
```python
# 在纯过滤查询判断后，添加额外兜底检查
if not is_pure_filter and self.query_formatter and self.query_formatter.is_enabled():
    # 检查 cleaned_query 是否只包含通用词汇
    test_query = cleaned_query
    generic_patterns = [r"照片", r"图片", r"相片", r"场景", r"画面", r"摄影", r"作品", r"影像", r"各种"]
    for pattern in generic_patterns:
        test_query = re.sub(pattern, "", test_query)
    test_query = re.sub(r"\s+", "", test_query).strip()
    
    # 如果清理后为空且原查询有过滤条件，判定为纯过滤
    if len(test_query) < 2 and (
        self._has_time_terms(query) or 
        self._has_season_terms(query) or 
        self._has_time_period_terms(query)
    ):
        is_pure_filter = True
        print(f"[DEBUG] QueryFormatter兜底：检测到纯过滤查询，跳过向量检索")
```

## 修复效果

### 预期行为（修复后）

```
用户输入: "夏天的照片"
    ↓
QueryFormatter (LLM)
    ↓
返回: { "search_text": "", "season": "夏天" }  ← LLM 返回空字符串
    ↓
(如果 LLM 失误返回了通用词汇)
    ↓
空值防护逻辑: 检测并清除通用词汇 → search_text = ""
    ↓
Searcher 判断
    ↓
cleaned_query = ""
    ↓
is_pure_filter = True ✓
    ↓
调用 _filter_only_search (直接使用 ES 过滤) ✓
    ↓
打印日志: "[DEBUG] 纯过滤查询模式"
    ↓
返回所有夏天的照片 ✓
```

### 支持的查询场景

1. **纯季节过滤**
   - 输入: "夏天的照片"
   - 行为: 直接 ES 过滤，不生成 embedding
   - 返回: 所有标记为"夏天"的照片

2. **季节+场景混合**
   - 输入: "夏天海边的照片"
   - 行为: ES 过滤季节 + 向量匹配"海边"
   - 返回: 夏天拍摄的海边照片

3. **纯时间过滤**
   - 输入: "2023年的照片"
   - 行为: 直接 ES 过滤，不生成 embedding
   - 返回: 2023年拍摄的所有照片

4. **纯时段过滤**
   - 输入: "傍晚的照片"
   - 行为: 直接 ES 过滤，不生成 embedding
   - 返回: 所有标记为"傍晚"的照片

## 测试方法

### 方法1：运行测试脚本

```bash
python test_season_search.py
```

测试脚本会：
1. 检查元数据中的季节分布
2. 执行4个测试用例（纯季节、季节+场景、纯年份、纯时段）
3. 验证是否有 "[DEBUG] 纯过滤查询模式" 日志输出

### 方法2：手动测试

启动服务后，通过前端或 API 测试：

```bash
curl -X POST http://localhost:5000/search_photos \
  -H 'Content-Type: application/json' \
  -d '{"query": "夏天的照片", "top_k": 10}'
```

**预期现象**：
- ✅ 控制台输出: `[DEBUG] 纯过滤查询模式, constraints: {'season': '夏天', ...}`
- ✅ **不应该看到**: "generate_embedding" 或 "26个字符" 相关日志
- ✅ 返回所有夏天的照片（如果 ES 数据已同步）

### 方法3：检查 ES 数据同步

如果测试返回空结果，可能是 ES 数据未同步：

```bash
# 检查 ES 索引文档数量
curl http://localhost:9200/photo_keywords/_count

# 查询夏天的照片
curl -X POST http://localhost:9200/photo_keywords/_search \
  -H 'Content-Type: application/json' \
  -d '{"query": {"term": {"season": "夏天"}}}'
```

如果 ES 中没有数据，需要重建索引：

```bash
curl -X POST http://localhost:5000/init_index
```

## 技术细节

### 季节信息提取逻辑

季节信息来源（优先级从高到低）：
1. **EXIF 拍摄时间** (`exif_data.datetime`)
2. **文件修改时间** (`file_time`)

月份到季节的映射（`indexer.py:_month_to_season`）：
- 3-5月 → 春天
- 6-8月 → 夏天
- 9-11月 → 秋天
- 12-2月 → 冬天

### 纯过滤查询判断逻辑

满足以下条件时判定为纯过滤查询：
1. 原始查询包含过滤词汇（时间/季节/时段）
2. 清理后的查询为空或只剩无意义词汇（"的照片"、"拍的"等）

### 数据流架构

```
索引阶段:
  照片 → Vision LLM (生成描述，不含季节)
      → EXIF解析 (提取季节)
      → VectorStore (向量索引)
      → KeywordStore/ES (关键字索引 + 季节字段)

搜索阶段:
  查询 → QueryFormatter (LLM拆分: search_text + season)
      → Searcher 判断是否纯过滤
      → 纯过滤: ES直接过滤 (不生成embedding)
      → 混合查询: ES过滤 + 向量检索 (生成embedding)
```

## 注意事项

1. **QueryFormatter 是可选的**：如果未配置，系统会使用正则表达式提取季节
2. **ES 是可选的**：如果未配置，系统会降级到内存过滤（`_memory_filter_search`）
3. **三层防护确保健壮性**：即使 LLM 失误，后续检查也能纠正

## 相关文件

- `utils/query_formatter.py`: QueryFormatter Prompt 和空值防护逻辑
- `core/searcher.py`: 纯过滤查询判断和兜底检查
- `core/indexer.py`: 季节信息提取逻辑
- `utils/keyword_store.py`: ES 过滤搜索实现
- `test_season_search.py`: 测试脚本

## 修复日期

2026-02-05

## 作者

AI Assistant (Claude Sonnet 4.5)
