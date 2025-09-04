#include "minicpmo.h"
#include "video_decoder.h"
#include "macos_stream.h"
#include "utils.h"

using namespace edge;

int main(int argc, const char** argv) {
    bool stream = false;

    std::string video_path   = "";
    std::string siglip_path  = "";
    std::string whisper_path = "";
    std::string llm_path     = "";
    float temperature = 0.0f; // 默认temperature值

    if (argc == 5) {
        printf("Usage using local video: ./minicpmo-cli <video_path> <siglip_path> <whisper_path> <llm_path>\n");
        video_path   = argv[1];
        siglip_path  = argv[2];
        whisper_path = argv[3];
        llm_path     = argv[4];
    }
    else if (argc == 4)
    {
        printf("Usage using stream: ./minicpmo-cli <siglip_path> <whisper_path> <llm_path>\n");
        stream = true;
        siglip_path  = argv[1];
        whisper_path = argv[2];
        llm_path     = argv[3]; 
    }else{
        printf("Usage using local video: ./minicpmo-cli <video_path> <siglip_path> <whisper_path> <llm_path>\n");
        printf("Usage using stream: ./minicpmo-cli <siglip_path> <whisper_path> <llm_path>\n");
        return 1;
    }
    
    // 获取默认参数
    auto llm_params = get_minicpmo_default_llm_params();
    // 修改temperature参数
    llm_params.sampling.temp = temperature;
    
    minicpmo_params params{
        siglip_path,
        whisper_path,
        llm_path,
        llm_params};

    MiniCPMO minicpmo(params);

    // 声明视频和音频数据变量
    std::vector<float> pcmf32_data;
    std::vector<edge::image_buf<uint8_t>> image_list;

    if (!stream) {
        VideoDecoder video_decoder(video_path);
        video_decoder.decode();
        auto audio_data = video_decoder.get_audio_pcmf32();
        auto video_data = video_decoder.get_video_buffer();
        
        pcmf32_data = audio_data;
        image_list = video_data;
    }
    else{
        MacOSStream stream;
        stream.initialize();
        stream.startRecording();
        auto audio_data = stream.get_audio_pcmf32();
        auto video_data = stream.get_video_buffer();
        
        pcmf32_data = audio_data;
        image_list = video_data;
    }

    /* dump for debugging */
    // video_decoder.write_wav("out.wav");
    // video_decoder.write_jpg("image");

    minicpmo.chat(image_list, pcmf32_data, "zh");
}
