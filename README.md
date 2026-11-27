# 棱镜记忆

棱镜记忆在同一 DeepSeek Harness 配置档的工作区之间共享记忆。L1、L2 和 L3 可以按文本查询相关分片，再用文档 ID 读取原文。L1 可将要保存的信息交给专职记忆智能体整理，由它提交文档、关联媒体和写入索引。

## 安装

需要 Node.js 24.9 或更新版本、DeepSeek Harness `0.2.3-alpha.2`，以及接入记忆工具的棱镜编排器。使用源码时，在仓库根目录构建并生成插件包。

```sh
npm ci
npm run build
npm test
npm pack
```

在 DSH 的 **插件 → 添加插件** 中输入生成的 `.tgz` 的绝对路径，安装并启用。记忆文档、媒体及向量保存在该配置档的 `data/dsh-prism/memory.sqlite`。记忆智能体使用同一配置档的 `data/dsh-prism/workspace/memories/`。

## 配置向量模型

默认通过 `onnxruntime-node` 在 CPU 上运行 `bge-large-zh-v1.5`。首次需要生成向量时，插件下载并校验固定版本的 ONNX 文件和 tokenizer，将其缓存到配置档的 `cache/dsh-prism/models/`。ONNX 文件约 327 MB。若已准备本机文件，可配置 `modelPath` 和 `tokenizerDirectory`。词表目录须包含 `tokenizer.json` 和 `tokenizer_config.json`。相对路径以配置档目录为基准。

使用远程模型时，在配置档的 Cordis 补丁中覆盖插件配置。下例使用 `Qwen3-VL-Embedding-8B`，可用文本、暂存图片或视频进行检索。服务器必须支持 vLLM 的 Embeddings `messages` 参数才能接收媒体。

```yaml
- id: dsh-memory
  config:
    embedding:
      backend: openai
      model: Qwen3-VL-Embedding-8B
      baseUrl: https://embedding.example.com/v1
      apiKeyEnv: PRISM_MEMORY_EMBEDDING_KEY
```

要使用远程 BGE，将 `model` 改为 `bge-large-zh-v1.5`。两种远程模型都支持 OpenAI 兼容的文本 Embeddings 请求。可用 `modelId` 设置部署端使用的模型名，用 `revision` 标识同名模型的一次更新。若服务端限制了输入长度，可设置 `maxInputTokens`。远程 Qwen 可配置 `dimensions`，值需与服务实际输出一致。选择远程后端时，待嵌入的查询、文档分片和所选媒体会发送到配置的服务。

插件按模型 tokenizer 自动分片。模型或分片配置变化后，插件会在后台为已有文档重建向量索引。重建期间继续使用原索引，完整重建后切换。重建失败时保留原索引，宿主日志会记录失败状态。完整正文和文档 ID 会保留。暂存媒体保留 24 小时。发送成功的聊天附件和记忆文档引用的媒体会继续保存。

## 验证

`npm test` 使用本地模拟的向量接口验证分片、SQLite 持久化、并发幂等、模型切换与图像和视频请求。Windows CPU 验证使用了真实 BGE ONNX 模型，完成中文文档写入、检索及全文回读。
