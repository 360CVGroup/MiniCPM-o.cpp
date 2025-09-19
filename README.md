# MiniCPM-o.cpp

Inference of [MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6) in plain C/C++

<h3 align="center">
<b>English</b> | <a href="README_ZH.md"> <b>中文</b> </a> | <a href="./docs/install_en.md"><b>Install</b></a> | <a href="./docs/report_en.md"><b>Report</b></a>
</h3>


# Features
- Plain C/C++ implementation based on [ggml](https://github.com/ggml-org/ggml).
- Requires only 8GB of VRAM for inference.
- Supports streaming processing for both audio and video inputs.
- Optimized for real-time video streaming on NVIDIA Jetson Orin Nano Super.
- Provides Python bindings, a web demo, and additional integration possibilities.

<!-- --- -->

<div align="center">
  <a href="https://www.bilibili.com/video/BV1fNa8zVEAy"><img src="./assets/bilibili_cover.png", width=70%></a>
</div>

# Installation

Clone and initialize the repository.
```bash
# Clone the repository
git clone https://github.com/360CVGroup/MiniCPM-o.cpp.git
cd MiniCPM-o.cpp
# Initialize and update submodules
git submodule update --init --recursive
```

Set up the Python environment and install the package:

```bash
# We recommend using uv for Python environment and package management
pip install uv

# Create and activate a virtual environment
uv venv
source .venv/bin/activate
# For fish shell, use: source .venv/bin/activate.fish

# Install the package in editable mode
uv pip install -e . --verbose
```

For detailed installation steps, please refer to the [installation guide](./docs/install_en.md).

# Quick Start

## 1. Model Prepare

Use Pre-converted and Quantized gguf Models (Recommended).
link: [Google Drive](https://drive.google.com/drive/folders/1xmkPHCzClJolUsEG_J6HZCATurjo3mDt?usp=sharing)
or [ModelScope](https://www.modelscope.cn/models/kkssss/MiniCPM-o.cpp)

Download and place all models in the `models/` directory.

## 2. Model Inference
For ease of integration, we provide a Python binding.
Run the script:
```bash
# in project root path
python test/test_minicpmo.py --apm-path models/minicpmo-audio-encoder_Q4_K.gguf --vpm-path models/minicpmo-image-encoder_Q4_1.gguf --llm-path models/Model-7.6B-Q4_K_M.gguf --video-path assets/Skiing.mp4
```
We also provide a C/C++ interface. For details, please refer to the [C++ Interface Documentation](docs/install_en.md#2-c-interface-optional).

## 3. WebUI Demo
Real-time video interaction demo:

### 3.1 Start model server
```shell
# in project root path
uv pip install -r web_demos/minicpm-o_2.6/requirements.txt
python web_demos/minicpm-o_2.6/model_server.py
```

### 3.2 Start web server
```shell
# Make sure Node and PNPM are installed.
sudo apt-get update
sudo apt-get install nodejs npm
npm install -g pnpm

cd web_demos/minicpm-o_2.6/web_server
# create ssl cert for https, https is required to request camera and microphone permissions.
bash ./make_ssl_cert.sh  # output key.pem and cert.pem

pnpm install  # install requirements
pnpm run dev  # start server
```
Open `https://localhost:8088/` in your browser for real-time video calls.

# Deployment on Edge Device
We have deployed the MiniCPM-omni model on the NVIDIA Jetson Orin Nano Super 8G embedded device.
This project supports real-time inference on NVIDIA Jetson Orin Nano Super 8Gb in `MAXN SUPER mode`.

If your embedded device is not running the Super system package, please refer to the [installation manual](https://www.jetson-ai-lab.com/initial_setup_jon.html) for instructions on installing the system package on your board.

We recorded a [video](https://www.bilibili.com/video/BV1fNa8zVEAy) of the model running on the Jetson device in real time, with no speed-up applied.

For NVIDIA Jetson Orin Nano Super performance, including inference time and first-token latency data, see [Inference Performance Optimization](./docs/report_en.md#3-inference-performance-optimization).


# License
This project is licensed under the Apache 2.0 License.
For model usage and distribution, please comply with the official model license.

# Reference
- [llama.cpp](https://github.com/ggml-org/llama.cpp): LLM inference in C/C++
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp): Port of OpenAI's Whisper model in C/C++
- [transformers](https://github.com/huggingface/transformers): Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.
- [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o): A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming on Your Phone.
