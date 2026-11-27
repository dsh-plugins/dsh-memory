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

打开 **设置 → 棱镜/记忆**，为当前 DSH 配置档绑定一个嵌入模型。绑定成功并完成初始化后，棱镜立即启用记忆工具和聊天媒体附件。以后每次启动 DSH 都会自动加载已绑定的模型。

- **本地部署**可选 `bge-m3`、`bge-large-zh-v1.5`、`bge-large-en-v1.5` 或 `Qwen3-Embedding-0.6B`。选择模型并确认下载后，设置页显示进度。插件优先从 HF-Mirror 下载，失败后按文件尝试 Hugging Face 官方源。校验通过的词表和 ONNX 文件缓存在 `cache/dsh-prism/models/`，模型通过 `onnxruntime-node` 在 CPU 上运行。 [HF-Mirror](https://hf-mirror.com/)
- **远程 API**填写 `BASE_URL`、`API_KEY` 和 `MODEL_ID`，确认连接后，插件会发送一条探测请求，验证模型能生成向量并展示实际维度。上下文长度优先采用服务返回的元数据，其次采用已识别模型的上限。若两者都缺失，设置页会标注保守估计值。远程绑定的密钥保存在当前 DSH 配置档的私有数据目录中。
- 下载或探测失败时可以重试。绑定成功后，来源、模型和连接配置锁定。

本机模型和通用远程模型支持文本及媒体描述。若需直接用图片或视频检索，请选择接受 vLLM Embeddings `messages` 参数的 `Qwen3-VL-Embedding-8B` 服务。待嵌入的查询、文档分片和相关媒体会发送到所配置的服务。视频按 12 秒一段、每秒一帧取样，检索结果标出命中时间范围。需要 PATH 中可用的 FFmpeg 和 ffprobe，也可配置 `video.ffmpegPath` 与 `video.ffprobePath`。

旧配置档的 Cordis `embedding` 设置继续生效，仍支持离线 `modelPath`、`tokenizerDirectory` 和远程 `apiKeyEnv`。已通过设置页绑定的配置档以绑定结果为准。旧版默认模型写入的文档与附件仍保存在原配置档，首次绑定后会按所选模型重建已有索引，保留原文和文档 ID。暂存媒体保留 24 小时，发送成功的聊天附件和记忆文档引用的媒体会继续保存。

## 验证

`npm test` 使用本地模拟的向量接口验证分片、SQLite 持久化、并发幂等、模型切换与图像和视频请求。Windows CPU 验证使用了真实 BGE ONNX 模型，完成中文文档写入、检索及全文回读。
