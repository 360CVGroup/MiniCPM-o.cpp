# MiniCPM-o.cpp

[MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6) 的 C/C++ 推理实现

<h3 align="center">
<a href="README.md"><b>English</b></a> | <b>中文</b> | <a href="./docs/install_zh.md"><b>安装教程</b></a> | <a href="./docs/report_zh.md"><b>部署报告</b></a>
</h3>

# 特性
- 基于 [ggml](https://github.com/ggml-org/ggml) 的 C/C++ 实现。
- 推理仅需 8GB 的显存。
- 支持音频和视频输入的流式处理。
- 针对 NVIDIA Jetson Orin Nano Super 优化，实现实时视频流式处理。
- 提供 Python 绑定、网页演示以及额外的集成可能性。

# 安装

克隆并初始化仓库。
```bash
# 克隆仓库
git clone https://github.com/360CVGroup/MiniCPM-o.cpp.git
cd MiniCPM-o.cpp
# 初始化并更新子模块
git submodule update --init --recursive
```

设置 Python 环境并安装包：

```bash
# 我们推荐使用 uv 进行 Python 环境和包管理
pip install uv

# 创建并激活虚拟环境
uv venv
source .venv/bin/activate
# 对于 fish shell，使用：source .venv/bin/activate.fish

# 以可编辑模式安装包
uv pip install -e . --verbose
```

有关详细的安装步骤，请参考 [安装指南](./docs/install_zh.md)。

# 快速开始

## 1. 模型准备

使用预转换和量化的 gguf 模型（推荐）。
链接：[模型文件](http://beijing.xstore.qihoo.net/cv-service/minicpmo-cpp/models.tar)

下载并将所有模型放置在 `models/` 目录中。

## 2. 模型推理
为了便于集成，我们提供了 Python 绑定。
运行脚本：
```bash
# 在项目根路径
python test/test_minicpmo.py --apm-path models/minicpmo-audio-encoder_Q4_K.gguf --vpm-path models/minicpmo-image-encoder_Q4_1.gguf --llm-path models/Model-7.6B-Q4_K_M.gguf --video-path assets/Skiing.mp4
```
我们还提供了 C/C++ 接口。有关详细信息，请参考 [C++ 接口文档](docs/install_zh.md#2-c-interface-optional)。

## 3. WebUI 演示
实时视频交互演示：

### 3.1 启动模型服务器
```shell
# 在项目根路径
uv pip install -r web_demos/minicpm-o_2.6/requirements.txt
python web_demos/minicpm-o_2.6/model_server.py
```

### 3.2 启动网页服务器
```shell
# 确保已安装 Node 和 PNPM。
sudo apt-get update
sudo apt-get install nodejs npm
npm install -g pnpm

cd web_demos/minicpm-o_2.6/web_server
# 为 https 创建 SSL 证书，https 是请求摄像头和麦克风权限所必需的。
bash ./make_ssl_cert.sh  # 输出 key.pem 和 cert.pem

pnpm install  # 安装依赖
pnpm run dev  # 启动服务器
```
在浏览器中打开 `https://localhost:8088/` 进行实时视频通话。

# 边缘设备部署
我们已在 NVIDIA Jetson Orin Nano Super 8G 嵌入式设备上部署了 MiniCPM-omni 模型。
此项目支持在 NVIDIA Jetson Orin Nano Super 8Gb 的 `MAXN SUPER 模式` 下进行实时推理。

如果您的嵌入式设备未运行 Super 系统包，请参考 [安装手册](https://www.jetson-ai-lab.com/initial_setup_jon.html) 了解如何在开发板上安装系统包。

我们录制了一段视频，展示了模型在 Jetson 设备上的实时运行情况，未应用任何加速。
[视频链接](https://beijing.xstore.qihoo.net/cv-service/minicpmo-cpp/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%AB%AF%E4%BE%A7%E9%83%A8%E7%BD%B2demo.mp4)

有关 NVIDIA Jetson Orin Nano Super 的性能，包括推理时间和首标延迟数据，请参阅 [推理性能优化](./docs/report_en.md#3-inference-performance-optimization)。

# 许可证
此项目基于 Apache 2.0 许可证授权。
对于模型的使用和分发，请遵守官方模型许可证。

# 参考
- [llama.cpp](https://github.com/ggml-org/llama.cpp): LLM inference in C/C++
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp): Port of OpenAI's Whisper model in C/C++
- [transformers](https://github.com/huggingface/transformers): Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.
- [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o): A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming on Your Phone.
