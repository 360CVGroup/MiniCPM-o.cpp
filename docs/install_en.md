# Installation & Deployment
Below are the step-by-step instructions for compiling the project and deploying the models.

## 1. Model Conversion

### 1. Use Pre-converted and Quantized gguf Models (Recommended)
Download link: [Google Drive](https://drive.google.com/drive/folders/1xmkPHCzClJolUsEG_J6HZCATurjo3mDt?usp=sharing)

### 2. Manually Convert and Quantize Models (Optional)
Coming soon...

## 2. Model Inference
### 1. Python Interface (Recommended)
For ease of integration, we provide a Python binding.
Installation:
```bash
pip install uv # Recommended to use uv for Python environment and package management

uv venv # Create virtual environment
source .venv/bin/activate # Activate virtual environment
# source .venv/bin/activate.fish  # for fish shell

uv pip install -e . --verbose # Local install
```

Run the script:
```bash
# in project root path
python test/test_minicpmo.py --apm-path models/minicpmo-audio-encoder_Q4_K.gguf --vpm-path models/minicpmo-image-encoder_Q4_1.gguf --llm-path models/Model-7.6B-Q4_K_M.gguf --video-path assets/Skiing.mp4
```

### 2. C++ Interface (Optional)
For integration, a C++ interface is provided, using ffmpeg as the video codec tool.
#### 2.1 FFmpeg Video Codec Installation
See [Ubuntu installation steps](../examples/minicpmo/README.md#video) or [FFmpeg official site](https://ffmpeg.org/download.html).

#### 2.2 CMake Build Options
Enable examples build option in cmake:
```bash
cmake --preset ggml-cuda -DBUILD_EXAMPLES=ON
cmake --build build
```

#### 2.3 CLI Usage Example
After compilation, use the following command:
```bash
# in project root path
./build/bin/minicpmo-cli assets/Skiing.mp4 models/minicpmo-image-encoder_Q4_1.gguf models/minicpmo-audio-encoder_Q4_K.gguf models/Model-7.6B-Q4_K_M.gguf
```

## 3. WebUI Demo
Real-time video interaction demo:

1. Start model server
```shell
# in project root path
uv pip install -r web_demos/minicpm-o_2.6/requirements.txt
python web_demos/minicpm-o_2.6/model_server.py
```

2. Start web server
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
Open in your browser: `https://localhost:8088/` for real-time video calls.

## Deployment on Real Device
We have deployed the MiniCPM-omni model on the NVIDIA Jetson Orin Nano Super 8G embedded device.
This project supports real-time inference on NVIDIA Jetson Orin Nano Super 8Gb in `MAXN SUPER mode`.

<img src="../assets/jetson-orin-nano-super-developer-kit-bbm-l.jpg" alt="Jetson Orin Nano Board" style="width:50%;height:auto;" />

If your embedded device is not running the Super system package, please refer to the [installation manual](https://www.jetson-ai-lab.com/initial_setup_jon.html) for steps to install the system package on your board.
### 1. Environment Preparation
After booting into the system, perform the following operations to free up more memory and improve runtime performance:
```bash
sudo init 3 # stop the desktop
sudo jetson_clocks # locks the clocks to their maximums
```

### 2. Compile & Run
Refer to the sections [Model Conversion](#1-model-conversion), [Model Inference](#2-model-inference), and [WebUI Demo](#3-webui-demo) for instructions.

### 3. Demo Video:
We recorded a video of the model running on the Jetson device, with no speed-up applied.
[Video link](https://drive.google.com/file/d/1yZulWUqaJHNCPmfNaaPKm0UlHFb4-O9Z/view?usp=drive_link)

### 4. Inference Performance
For inference time and first-token latency data, see [Inference Performance Optimization](report_en.md#3-inference-performance-optimization).
