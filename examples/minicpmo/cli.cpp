#include "minicpmo.h"
#include "video_decoder.h"

using namespace edge;

void streaming_process_video(MiniCPMO& minicpmo, std::string video_path, std::string user_prompt = "", bool eval_sys_prompt = true) {
    minicpmo.apm_streaming_mode(true);
    minicpmo.apm_kv_clear();

    VideoDecoder video_decoder(video_path);
    video_decoder.decode();
    auto pcmf32_data = video_decoder.get_audio_pcmf32();
    auto image_list  = video_decoder.get_video_buffer();

    /* dump for debugging */
    // video_decoder.write_wav("out.wav");
    // video_decoder.write_jpg("image");

    std::vector<std::vector<float>> pcmf32_list;
    for (int i = 0; i < image_list.size(); ++i) {
        std::vector<float> pcmf32_chunk(16000, 0);
        std::memcpy(pcmf32_chunk.data(), pcmf32_data.data() + i * 16000, 16000 * sizeof(float));
        pcmf32_list.emplace_back(pcmf32_chunk);
    }

    std::string lang = "en";
    if (eval_sys_prompt) {
        minicpmo.eval_system_prompt(lang);
    }

    for (int i = 0; i < image_list.size(); ++i) {
        minicpmo.streaming_prefill(image_list[i], pcmf32_list[i]);
    }
    while (true) {
        std::string tmp = minicpmo.streaming_generate(user_prompt);
        if (tmp.empty()) {
            break;
        } else {
            std::cout << tmp << std::flush;
        }
    }
}

void offline_process_video(MiniCPMO& minicpmo, std::string video_path) {
    VideoDecoder video_decoder(video_path);
    video_decoder.decode();
    auto pcmf32_data = video_decoder.get_audio_pcmf32();
    auto image_list  = video_decoder.get_video_buffer();

    /* dump for debugging */
    // video_decoder.write_wav("out.wav");
    // video_decoder.write_jpg("image");

    minicpmo.chat("", image_list, pcmf32_data, "en");
}

int main(int argc, const char** argv) {
    if (argc != 5 && argc != 8) {
        printf("Usage w/o tts: ./minicpmo-cli <video_path> <siglip_path> <whisper_path> <llm_path>\n");
        printf("OR\n");
        printf("Usage w/ tts: ./minicpmo-cli <video_path> <siglip_path> <whisper_path> <llm_path> <ttc_model_path> <cts_model_path> <audio_output_path>\n");
        return 1;
    }

    std::string video_path   = argv[1];
    std::string siglip_path  = argv[2];
    std::string whisper_path = argv[3];
    std::string llm_path     = argv[4];

    auto llm_params = get_minicpmo_default_llm_params();
    // large context in llm for support 1 minute video
    llm_params.n_ctx = 8192;
    // llm_params.n_ctx     = 32768;
    llm_params.n_keep    = 4;  // for streaming-llm settings
    llm_params.ctx_shift = true;
    // for performance
    llm_params.flash_attn = true;
    // llm_params.n_predict = 4096;
    // w/ or w/o tts
    std::string ttc_model_path    = "";
    std::string cts_model_path    = "";
    std::string audio_output_path = "";
    if (argc == 8) {
        ttc_model_path    = argv[5];
        cts_model_path    = argv[6];
        audio_output_path = argv[7];
    }

    minicpmo_params params{
        siglip_path,
        whisper_path,
        llm_path,
        ttc_model_path,
        cts_model_path,
        llm_params};

    MiniCPMO minicpmo(params);

    // std::string user_prompt = "aside from the cup facing up, can you tell me where is the other cup that has a red snake on it as well\n";
    std::string user_prompt  = "";
    std::string video_path_0 = "/devel/geelib/edge-cpp/test_clip_0.mp4";
    std::string video_path_1 = "/devel/geelib/edge-cpp/test_clip_1.mp4";
    std::string video_path_2 = "/devel/geelib/edge-cpp/test_clip_2.mp4";
    // std::string video_path_3 = "/devel/geelib/edge-cpp/video_5min.mp4";
    // offline_process_video(minicpmo, video_path_0);
    // streaming_process_video(minicpmo, video_path_0, user_prompt);

    streaming_process_video(minicpmo, video_path_0, user_prompt);
    streaming_process_video(minicpmo, video_path_1, user_prompt, false);
    streaming_process_video(minicpmo, video_path_2, user_prompt, false);

    // streaming_process_video(minicpmo, video_path_3, user_prompt);
    return 0;
}
