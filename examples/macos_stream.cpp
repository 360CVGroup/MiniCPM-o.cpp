#include "macos_stream.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstdint>

// 音频录制参数
const int FRAMES_PER_BUFFER = 512;     // 修改缓冲区大小为512
const int NUM_CHANNELS = 1;            // 修改为单声道
const int SAMPLE_RATE = 16000;         // 修改采样率为16000Hz

// 视频录制参数
const int VIDEO_WIDTH = 1280;
const int VIDEO_HEIGHT = 720;
const int VIDEO_FPS = 30;

// 录制时间配置
const int RECORDING_SECONDS = 10;      // 总录制时间，秒

// 检查摄像头权限
bool checkCameraPermission() {
    std::cout << "正在检查摄像头权限..." << std::endl;
    
    // 尝试等待摄像头权限
    int attempts = 0;
    const int maxAttempts = 30; // 最多等待30秒
    
    while (attempts < maxAttempts) {
        // 尝试打开摄像头
        cv::VideoCapture test_cap(0);
        if (test_cap.isOpened()) {
            // 尝试读取一帧
            cv::Mat test_frame;
            test_cap >> test_frame;
            if (!test_frame.empty()) {
                test_cap.release();
                std::cout << "摄像头权限检查通过" << std::endl;
                return true;
            }
            test_cap.release();
        }
        
        // 未获得权限，显示提示并等待
        if (attempts == 0) {
            std::cout << "等待获取摄像头权限..." << std::endl;
            std::cout << "请在系统权限请求弹窗中选择「允许」" << std::endl;
            std::cout << "或前往：系统偏好设置 -> 安全性与隐私 -> 摄像头 中允许访问" << std::endl;
            std::cout << "正在等待权限授予（最多30秒）..." << std::endl;
        }
        
        // 每秒检查一次
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "等待中... (" << ++attempts << "/" << maxAttempts << ")" << std::endl;
    }
    
    std::cerr << "错误：无法打开摄像头，请检查以下可能的原因：" << std::endl;
    std::cerr << "1. 系统偏好设置 -> 安全性与隐私 -> 摄像头 中是否允许访问" << std::endl;
    std::cerr << "2. 摄像头是否被其他程序占用" << std::endl;
    std::cerr << "3. 摄像头硬件是否正常工作" << std::endl;
    return false;
}

// 音频录制回调函数
int recordCallback(const void *inputBuffer, void *outputBuffer,
                  unsigned long framesPerBuffer,
                  const PaStreamCallbackTimeInfo* timeInfo,
                  PaStreamCallbackFlags statusFlags,
                  void *userData) {
    AudioData *audioData = static_cast<AudioData*>(userData);
    const int16_t *in = static_cast<const int16_t*>(inputBuffer);
    
    std::lock_guard<std::mutex> lock(audioData->mutex);
    
    if (inputBuffer == nullptr) {
        for (unsigned long i = 0; i < framesPerBuffer; i++) {
            audioData->recordedSamples.push_back(0);
        }
    } else {
        for (unsigned long i = 0; i < framesPerBuffer; i++) {
            audioData->recordedSamples.push_back(in[i]);
        }
    }
    
    return paContinue;
}

// 获取当前时间戳作为文件名
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
    return ss.str();
}

// 写入WAV文件头
void writeWavHeader(std::ofstream& outFile, int sampleRate, int numChannels, size_t dataSize) {
    // RIFF Chunk
    outFile.write("RIFF", 4);
    uint32_t chunkSize = 36 + dataSize;
    outFile.write(reinterpret_cast<char*>(&chunkSize), 4);
    outFile.write("WAVE", 4);

    // fmt Subchunk
    outFile.write("fmt ", 4);
    uint32_t subchunk1Size = 16; // PCM format size
    outFile.write(reinterpret_cast<char*>(&subchunk1Size), 4);
    uint16_t audioFormat = 1; // PCM
    outFile.write(reinterpret_cast<char*>(&audioFormat), 2);
    uint16_t channels = static_cast<uint16_t>(numChannels);
    outFile.write(reinterpret_cast<char*>(&channels), 2);
    uint32_t sr = static_cast<uint32_t>(sampleRate);
    outFile.write(reinterpret_cast<char*>(&sr), 4);
    uint32_t byteRate = sampleRate * numChannels * 2; // 16位 = 2字节
    outFile.write(reinterpret_cast<char*>(&byteRate), 4);
    uint16_t blockAlign = numChannels * 2; // 16位 = 2字节
    outFile.write(reinterpret_cast<char*>(&blockAlign), 2);
    uint16_t bitsPerSample = 16;
    outFile.write(reinterpret_cast<char*>(&bitsPerSample), 2);

    // data Subchunk
    outFile.write("data", 4);
    uint32_t subchunk2Size = dataSize;
    outFile.write(reinterpret_cast<char*>(&subchunk2Size), 4);
}

// 打印音频设备信息
void printAudioDeviceInfo() {
    int numDevices = Pa_GetDeviceCount();
    std::cout << "可用的音频设备数量: " << numDevices << std::endl;
    
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        std::cout << "\n设备 " << i << ":" << std::endl;
        std::cout << "名称: " << deviceInfo->name << std::endl;
        std::cout << "最大输入通道数: " << deviceInfo->maxInputChannels << std::endl;
        std::cout << "最大输出通道数: " << deviceInfo->maxOutputChannels << std::endl;
        std::cout << "默认采样率: " << deviceInfo->defaultSampleRate << std::endl;
    }
}

// MacOSStream 类的实现
MacOSStream::MacOSStream() 
    : m_audioStream(nullptr), 
      m_audioDeviceId(-1), 
      m_frameCount(0), 
      m_savedImageCount(0), 
      m_isInitialized(false), 
      m_isRecording(false) {
}

MacOSStream::~MacOSStream() {
    if (m_isRecording) {
        stopRecording();
    }
    
    // 释放资源
    m_camera.release();
    m_videoWriter.release();
    
    if (m_audioStream) {
        Pa_CloseStream(m_audioStream);
    }
    
    Pa_Terminate();
}

bool MacOSStream::initialize() {
    if (m_isInitialized) {
        return true;
    }
    
    std::cout << "程序启动..." << std::endl;
    
    // 检查权限
    if (!checkAndRequestPermissions()) {
        return false;
    }
    
    // 初始化摄像头
    if (!initializeCamera()) {
        return false;
    }
    
    // 初始化音频
    if (!initializeAudio()) {
        return false;
    }
    
    // 创建输出目录
    if (!setupOutputDirectory()) {
        return false;
    }
    
    m_isInitialized = true;
    return true;
}

bool MacOSStream::checkAndRequestPermissions() {
    return checkCameraPermission();
}

bool MacOSStream::initializeCamera() {
    std::cout << "正在初始化摄像头..." << std::endl;
    m_camera.open(0);
    if (!m_camera.isOpened()) {
        std::cerr << "错误：无法打开摄像头" << std::endl;
        return false;
    }
    
    // 设置摄像头参数
    m_camera.set(cv::CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH);
    m_camera.set(cv::CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT);
    m_camera.set(cv::CAP_PROP_FPS, VIDEO_FPS);
    
    std::cout << "摄像头初始化成功" << std::endl;
    return true;
}

bool MacOSStream::initializeAudio() {
    std::cout << "正在初始化PortAudio..." << std::endl;
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "错误：PortAudio初始化失败: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }
    std::cout << "PortAudio初始化成功" << std::endl;
    
    // 打印音频设备信息
    printAudioDeviceInfo();
    
    // 查找音频输入设备
    if (!findAudioInputDevice()) {
        return false;
    }
    
    return true;
}

bool MacOSStream::findAudioInputDevice() {
    int numDevices = Pa_GetDeviceCount();
    
    std::cout << "\n正在查找音频输入设备..." << std::endl;
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        std::string deviceName = deviceInfo->name;
        std::cout << "检查设备 " << i << ": " << deviceName << std::endl;
        std::cout << "  输入通道数: " << deviceInfo->maxInputChannels << std::endl;
        std::cout << "  输出通道数: " << deviceInfo->maxOutputChannels << std::endl;
        std::cout << "  采样率: " << deviceInfo->defaultSampleRate << std::endl;
        
        // 查找内置麦克风
        if (deviceInfo->maxInputChannels > 0 && 
            (deviceName.find("麦克风") != std::string::npos || 
             deviceName.find("Microphone") != std::string::npos ||
             deviceName.find("Built-in") != std::string::npos)) {
            m_audioDeviceId = i;
            std::cout << "找到音频输入设备: " << deviceName << std::endl;
            break;
        }
    }
    
    if (m_audioDeviceId == -1) {
        // 如果找不到特定的设备，使用默认输入设备
        m_audioDeviceId = Pa_GetDefaultInputDevice();
        if (m_audioDeviceId == paNoDevice) {
            std::cerr << "错误：未找到音频输入设备" << std::endl;
            return false;
        }
        std::cout << "使用默认音频输入设备" << std::endl;
    }
    
    return true;
}

bool MacOSStream::setupOutputDirectory() {
    std::cout << "创建输出目录..." << std::endl;
    system("mkdir -p output");
    std::cout << "输出目录创建成功" << std::endl;
    return true;
}

bool MacOSStream::openVideoWriter(const std::string& timestamp) {
    // 为视频文件创建VideoWriter - 使用平台兼容性更好的编解码器
#ifdef _WIN32
    int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V'); // Windows
#elif __APPLE__
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // macOS (H.264)
#else
    int fourcc = cv::VideoWriter::fourcc('X', '2', '6', '4'); // Linux
#endif

    m_videoFilename = "output/video_" + timestamp + ".mp4";
    
    // 确保视频宽高是偶数（一些编解码器要求）
    int width = (VIDEO_WIDTH % 2 == 0) ? VIDEO_WIDTH : VIDEO_WIDTH - 1;
    int height = (VIDEO_HEIGHT % 2 == 0) ? VIDEO_HEIGHT : VIDEO_HEIGHT - 1;
    
    m_videoWriter.open(m_videoFilename, fourcc, VIDEO_FPS, cv::Size(width, height), true);
    
    if (!m_videoWriter.isOpened()) {
        std::cerr << "错误：无法创建视频文件，尝试使用其他编解码器..." << std::endl;
        // 尝试兼容编解码器
        fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        m_videoFilename = "output/video_" + timestamp + ".avi";
        m_videoWriter.open(m_videoFilename, fourcc, VIDEO_FPS, cv::Size(width, height), true);
        
        if (!m_videoWriter.isOpened()) {
            std::cerr << "错误：无法创建视频文件，使用默认编解码器" << std::endl;
            fourcc = -1; // 使用默认编解码器
            m_videoFilename = "output/video_" + timestamp + ".avi";
            m_videoWriter.open(m_videoFilename, fourcc, VIDEO_FPS, cv::Size(width, height), true);
            
            if (!m_videoWriter.isOpened()) {
                std::cerr << "错误：所有尝试都失败，无法创建视频文件" << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

bool MacOSStream::startRecording() {
    if (!m_isInitialized) {
        std::cerr << "错误：系统未初始化，请先调用initialize()" << std::endl;
        return false;
    }
    
    if (m_isRecording) {
        std::cerr << "警告：录制已经在进行中" << std::endl;
        return true;
    }
    
    // 清空之前的数据
    m_pcmf32_data.clear();
    m_image_list.clear();
    
    // 获取开始时间戳（用于文件名）
    std::string startTimestamp = getTimestamp();
    
    // 打开视频写入器
    if (!openVideoWriter(startTimestamp)) {
        return false;
    }
    
    // 初始化音频数据结构
    m_audioData.recordedSamples.clear();
    m_audioData.isRecording = true;
    
    // 设置音频输入参数
    PaStreamParameters inputParameters;
    inputParameters.device = m_audioDeviceId;
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(inputParameters.device);
    
    std::cout << "\n使用音频输入设备: " << deviceInfo->name << std::endl;
    std::cout << "设备最大输入通道数: " << deviceInfo->maxInputChannels << std::endl;
    std::cout << "使用采样率: " << SAMPLE_RATE << std::endl;
    
    inputParameters.channelCount = NUM_CHANNELS;
    inputParameters.sampleFormat = paInt16;
    inputParameters.suggestedLatency = deviceInfo->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;
    
    // 打开音频流
    PaError err = Pa_OpenStream(&m_audioStream,
                               &inputParameters,
                               nullptr,
                               SAMPLE_RATE,
                               FRAMES_PER_BUFFER,
                               paClipOff,
                               recordCallback,
                               &m_audioData);
    
    if (err != paNoError) {
        std::cerr << "错误：无法打开音频流: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }
    
    // 启动音频录制
    err = Pa_StartStream(m_audioStream);
    if (err != paNoError) {
        std::cerr << "错误：无法启动音频流: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(m_audioStream);
        return false;
    }
    
    std::cout << "开始录制视频和音频，总时长: " << RECORDING_SECONDS << " 秒..." << std::endl;
    
    // 重置计数器
    m_frameCount = 0;
    m_savedImageCount = 0;
    
    m_isRecording = true;
    
    // 处理视频帧
    if (!processVideoFrames()) {
        stopRecording();
        return false;
    }
    
    // 停止录制并保存
    if (!stopRecording()) {
        return false;
    }
    
    // 将int16_t音频样本转换为float格式
    convertAudioToFloat();
    
    // 保存并合并媒体文件
    if (!saveAndMergeMedia(startTimestamp)) {
        return false;
    }
    
    std::cout << "\n处理结果：" << std::endl;
    std::cout << "pcmf32_data大小: " << m_pcmf32_data.size() << " 个采样点" << std::endl;
    std::cout << "image_list大小: " << m_image_list.size() << " 帧图像" << std::endl;
    
    return true;
}

bool MacOSStream::processVideoFrames() {
    // 清空之前的图像列表
    m_image_list.clear();
    
    // 记录开始时间和每一秒的检查点时间
    auto startTime = std::chrono::steady_clock::now();
    auto nextSecondTime = startTime + std::chrono::seconds(1);
    
    // 声明frame变量
    cv::Mat frame;
    
    // 循环录制视频帧
    while (m_isRecording) {
        // 检查是否达到了录制时间
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
        
        if (elapsedSeconds >= RECORDING_SECONDS) {
            // 在结束前保存最后一秒的图像
            if (m_savedImageCount < RECORDING_SECONDS) {
                // 保存到image_list
                if (!frame.empty()) {
                    cv::Mat rgb_frame;
                    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);  // 转换为RGB格式
                    
                    // 创建image_buf结构并存储
                    image_buf<uint8_t> image;
                    image.nx = rgb_frame.cols;
                    image.ny = rgb_frame.rows;
                    image.buf.resize(rgb_frame.total() * 3);  // 3通道
                    
                    // 复制图像数据
                    std::memcpy(image.buf.data(), rgb_frame.data, image.buf.size());
                    m_image_list.push_back(image);
                    
                    // 同时保存图像文件
                    std::string timestamp = getTimestamp();
                    std::string imagePath = "output/frame_" + timestamp + ".jpg";
                    cv::imwrite(imagePath, frame);
                    m_savedImageCount++;
                    std::cout << "已保存图像 " << m_savedImageCount << "/" << RECORDING_SECONDS 
                            << ": " << imagePath << " (第 " << elapsedSeconds << " 秒)" << std::endl;
                }
            }
            break;
        }
        
        // 捕获一帧
        m_camera >> frame;
        
        if (frame.empty()) {
            std::cerr << "警告：捕获到空帧" << std::endl;
            continue;
        }
        
        // 写入视频文件
        m_videoWriter.write(frame);
        m_frameCount++;
        
        // 精确控制每秒保存一张图像
        if (currentTime >= nextSecondTime) {
            // 增加保存计数
            m_savedImageCount++;
            
            // 转换为RGB并保存到image_list
            cv::Mat rgb_frame;
            cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);  // 转换为RGB格式
            
            // 创建image_buf结构并存储
            image_buf<uint8_t> image;
            image.nx = rgb_frame.cols;
            image.ny = rgb_frame.rows;
            image.buf.resize(rgb_frame.total() * 3);  // 3通道
            
            // 复制图像数据
            std::memcpy(image.buf.data(), rgb_frame.data, image.buf.size());
            m_image_list.push_back(image);
            
            // 获取当前时间戳（每秒结尾时间）
            std::string timestamp = getTimestamp();
            
            // 保存图像
            std::string imagePath = "output/frame_" + timestamp + ".jpg";
            cv::imwrite(imagePath, frame);
            std::cout << "已保存图像 " << m_savedImageCount << "/" << RECORDING_SECONDS 
                    << ": " << imagePath << " (第 " << elapsedSeconds << " 秒)" << std::endl;
            
            // 更新下一秒检查点
            nextSecondTime = startTime + std::chrono::seconds(m_savedImageCount + 1);
        }
        
        // 更精确的帧率控制
        auto frameStartTime = std::chrono::steady_clock::now();
        auto expectedFrameTime = startTime + std::chrono::milliseconds(m_frameCount * 1000 / VIDEO_FPS);
        auto sleepTime = expectedFrameTime - frameStartTime;
        
        if (sleepTime.count() > 0) {
            std::this_thread::sleep_for(sleepTime);
        }
    }
    
    std::cout << "已采集 " << m_image_list.size() << " 张图像到image_list中" << std::endl;
    return true;
}

bool MacOSStream::stopRecording() {
    if (!m_isRecording) {
        std::cerr << "警告：没有正在进行的录制" << std::endl;
        return false;
    }
    
    // 停止音频录制
    m_audioData.isRecording = false;
    
    // 等待一小段时间确保音频缓冲区被处理完
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    PaError err = Pa_StopStream(m_audioStream);
    if (err != paNoError) {
        std::cerr << "错误：无法停止音频流: " << Pa_GetErrorText(err) << std::endl;
    }
    
    err = Pa_CloseStream(m_audioStream);
    if (err != paNoError) {
        std::cerr << "错误：无法关闭音频流: " << Pa_GetErrorText(err) << std::endl;
    }
    
    // 确保先关闭视频写入器
    m_videoWriter.release();
    std::cout << "视频文件已保存: " << m_videoFilename << std::endl;
    std::cout << "总共录制了 " << m_frameCount << " 帧视频" << std::endl;
    std::cout << "总共保存了 " << m_savedImageCount << " 张图像" << std::endl;
    std::cout << "实际录制时长: " << m_frameCount / (float)VIDEO_FPS << " 秒" << std::endl;
    
    m_isRecording = false;
    return true;
}

bool MacOSStream::saveAudioToFile(const std::string& timestamp) {
    std::string audioPath = "output/audio_" + timestamp + ".wav";
    std::ofstream outFile(audioPath, std::ios::binary);
    
    if (!outFile.is_open()) {
        std::cerr << "错误：无法创建音频文件: " << audioPath << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(m_audioData.mutex);
    // 写入WAV文件头
    int dataSize = m_audioData.recordedSamples.size() * sizeof(int16_t);
    writeWavHeader(outFile, SAMPLE_RATE, NUM_CHANNELS, dataSize);
    
    // 写入音频数据
    outFile.write(reinterpret_cast<char*>(m_audioData.recordedSamples.data()), dataSize);
    
    outFile.close();
    std::cout << "已保存音频: " << audioPath << std::endl;
    std::cout << "录制的样本数: " << m_audioData.recordedSamples.size() << std::endl;
    std::cout << "录制的时长: " << m_audioData.recordedSamples.size() / (float)SAMPLE_RATE << " 秒" << std::endl;
    
    return true;
}

bool MacOSStream::mergeAudioAndVideo(const std::string& videoFile, const std::string& audioFile, const std::string& outputFile) {
    // 合并音视频（使用ffmpeg，需要用户安装）
    std::string ffmpegCmd = "ffmpeg -y -i \"" + videoFile + "\" -i \"" + audioFile + 
                           "\" -c:v copy -c:a aac -strict experimental -shortest \"" + outputFile + "\"";
    
    std::cout << "正在合并音视频..." << std::endl;
    std::cout << "执行命令: " << ffmpegCmd << std::endl;
    int mergeResult = system(ffmpegCmd.c_str());
    
    if (mergeResult == 0) {
        std::cout << "音视频合并成功: " << outputFile << std::endl;
        return true;
    } else {
        std::cerr << "音视频合并失败，请确保已安装ffmpeg" << std::endl;
        std::cerr << "您可以手动合并音视频文件: " << videoFile << " 和 " << audioFile << std::endl;
        return false;
    }
}

bool MacOSStream::saveAndMergeMedia(const std::string& timestamp) {
    // 保存音频文件
    std::string audioPath = "output/audio_" + timestamp + ".wav";
    if (!saveAudioToFile(timestamp)) {
        return false;
    }
    
    // 合并音视频
    std::string finalVideoPath = "output/final_" + timestamp + ".mp4";
    if (!mergeAudioAndVideo(m_videoFilename, audioPath, finalVideoPath)) {
        return false;
    }
    
    return true;
}

void MacOSStream::convertAudioToFloat() {
    std::lock_guard<std::mutex> lock(m_audioData.mutex);
    
    // 清空之前的数据
    m_pcmf32_data.clear();
    m_pcmf32_data.reserve(m_audioData.recordedSamples.size());
    
    // 将int16_t音频样本转换为float格式
    // int16_t范围是-32768到32767，转换为float范围-1.0到1.0
    for (const auto& sample : m_audioData.recordedSamples) {
        float normalizedSample = static_cast<float>(sample) / 32768.0f;
        m_pcmf32_data.push_back(normalizedSample);
    }
    
    std::cout << "已将 " << m_audioData.recordedSamples.size() << " 个int16_t样本转换为float格式" << std::endl;
} 