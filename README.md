# 代码 RAG 系统

一个基于 Ollama 的本地部署代码检索增强生成 (RAG) 系统。该系统允许您：

1. 导入代码文件
2. 自动将代码分割为有意义的代码块
3. 建立可搜索索引
4. 使用自然语言查询检索相关代码块

## 功能

- **代码处理**：解析代码文件并基于语法提取有意义的代码块
- **智能分段**：按函数、类和注释块划分代码
- **向量索引**：为代码块创建嵌入向量用于语义相似度搜索
- **混合搜索**：结合语义和关键词搜索以获得更好的结果
- **查询增强**：使用 LLM 增强用户查询以获得更好的检索结果
- **完全本地部署**：使用 Ollama 模型，无需依赖外部 API

## 安装

1. 安装 [Ollama](https://ollama.com/)
2. 拉取必要的 Ollama 模型：
   ```bash
   ollama pull qwen2.5       # 用于回答问题的 LLM
   ollama pull nomic-embed-text  # 用于嵌入的模型
   ```
3. 克隆仓库
4. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
5. 基于 `env.example` 模板创建 `.env` 文件

## 使用方法

### 导入代码

导入代码文件或目录：

```bash
python main.py ingest /path/to/code
```

您可以使用不同的嵌入模型：

```bash
python main.py ingest /path/to/code --embed-model nomic-embed-text
```

### 查询索引

搜索代码块：

```bash
python main.py query "您的自然语言查询"
```

您可以指定返回结果的数量：

```bash
python main.py query "您的查询" --top_k 5
```

使用不同的 LLM 或嵌入模型：

```bash
python main.py query "您的查询" --llm-model qwen2.5 --embed-model nomic-embed-text
```

### 询问关于代码的问题

```bash
python main.py qa "Person 类是如何工作的？"
```

## 支持的语言

该系统目前具有针对以下语言的专用解析器：
- Python
- JavaScript/TypeScript

其他语言使用通用解析器支持，该解析器尝试提取有意义的代码块。

## 工作原理

1. **代码处理**：系统解析代码文件并识别逻辑片段（函数、类等）
2. **嵌入**：代码块使用 Ollama 的嵌入模型转换为向量嵌入
3. **索引**：嵌入存储在 FAISS 向量数据库中以进行高效的相似性搜索
4. **检索**：当您查询系统时，它：
   - 使用 LLM 增强您的查询
   - 使用向量嵌入执行语义搜索
   - 结合结果进行混合检索
   - 返回最相关的代码块

## 扩展系统

您可以通过以下方式扩展系统：
- 在 `CodeProcessor` 类中添加对更多编程语言的支持
- 实现不同的嵌入模型
- 添加更复杂的检索策略

## 依赖

- LangChain：用于 RAG 组件和嵌入
- FAISS：用于向量相似性搜索
- Ollama：用于本地嵌入和 LLM 推理
- Pydantic：用于数据建模

## 许可证

MIT 