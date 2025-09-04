#ifndef MACOS_STREAM_H
#define MACOS_STREAM_H

#include <opencv2/opencv.hpp>
#include <portaudio.h>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include "utils.h"

using namespace edge;

// 音频录制参数
extern const int FRAMES_PER_BUFFER;
extern const int NUM_CHANNELS;
extern const int SAMPLE_RATE;

// 视频录制参数
extern const int VIDEO_WIDTH;
extern const int VIDEO_HEIGHT;
extern const int VIDEO_FPS;

// 录制时间配置
extern const int RECORDING_SECONDS;

// 音频录制的共享数据结构
struct AudioData {
    std::vector<int16_t> recordedSamples;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> isRecording{false};
};

// 摄像头权限和设备相关函数
bool checkCameraPermission();
void printAudioDeviceInfo();

// 音频录制回调函数
int recordCallback(const void *inputBuffer, void *outputBuffer,
                  unsigned long framesPerBuffer,
                  const PaStreamCallbackTimeInfo* timeInfo,
                  PaStreamCallbackFlags statusFlags,
                  void *userData);

// 文件和时间工具函数
std::string getTimestamp();
void writeWavHeader(std::ofstream& outFile, int sampleRate, int numChannels, size_t dataSize);

// 流媒体相关的主要类
class MacOSStream {
public:
    MacOSStream();
    ~MacOSStream();
    
    // 初始化和检查
    bool initialize();
    bool checkAndRequestPermissions();
    
    // 设备和资源操作
    bool initializeCamera();
    bool initializeAudio();
    bool findAudioInputDevice();
    bool setupOutputDirectory();
    
    // 录制操作
    bool startRecording();
    bool stopRecording();
    bool saveAndMergeMedia(const std::string& timestamp);
    
    // 状态查询
    bool isRecording() const { return m_isRecording; }
    
    // 获取处理后的音频和图像数据
    std::vector<float> get_audio_pcmf32() const { return m_pcmf32_data; }
    std::vector<image_buf<uint8_t>> get_video_buffer() const { return m_image_list; }
    
private:
    // 摄像头相关
    cv::VideoCapture m_camera;
    cv::VideoWriter m_videoWriter;
    std::string m_videoFilename;
    int m_frameCount;
    int m_savedImageCount;
    
    // 音频相关
    PaStream* m_audioStream;
    AudioData m_audioData;
    int m_audioDeviceId;
    
    // 状态
    bool m_isInitialized;
    bool m_isRecording;
    
    // 处理后的数据
    std::vector<float> m_pcmf32_data;  // 音频数据（32位浮点格式）
    std::vector<image_buf<uint8_t>> m_image_list;  // 视频帧列表
    
    // 辅助方法
    bool openVideoWriter(const std::string& timestamp);
    bool processVideoFrames();
    bool saveAudioToFile(const std::string& timestamp);
    bool mergeAudioAndVideo(const std::string& videoFile, const std::string& audioFile, const std::string& outputFile);
    
    // 数据处理方法
    void convertAudioToFloat();  // 将int16音频转换为float格式
};

#endif // MACOS_STREAM_H 