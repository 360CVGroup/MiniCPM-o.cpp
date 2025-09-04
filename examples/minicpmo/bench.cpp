#include <cstdio>
#include <fstream>
#include <string>
#include "common.h"
#include "minicpmo.h"
#include "video_decoder.h"

using namespace edge;

std::string get_context(std::string file_path) {
    std::ifstream inp(file_path);
    std::string context = "";
    if (inp.is_open()) {
        std::stringstream buffer;
        buffer << inp.rdbuf();
        context = buffer.str();
        inp.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }
    return context;
}

void dump_context(std::string file_path, std::string context) {
    std::ofstream out(file_path);
    if (out.is_open()) {
        out << context;
        out.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }
}

int main(int argc, const char** argv) {
    if (argc != 4) {
        printf("Usage: ./minicpmo-cli <siglip_path> <whisper_path> <llm_path>\n");
        return 1;
    }

    std::string siglip_path  = argv[1];
    std::string whisper_path = argv[2];
    std::string llm_path     = argv[3];

    auto llm_params = get_minicpmo_default_llm_params();
    // large context in llm for support 1 minute video pieces
    llm_params.n_ctx = 8192;
    // llm_params.n_ctx = 16384;

    minicpmo_params params{
        siglip_path,
        whisper_path,
        llm_path,
        "",
        "",
        llm_params};

    MiniCPMO minicpmo(params);

    // Now you have the content of the file in the 'prompt' string
    std::string prompt_dir = "/devel/github/StreamingBench/output/prompt_path/";
    std::string file_dir   = "/devel/github/StreamingBench/output/file_path/";
    for (int i = 1000; i < 1500; ++i) {
        std::string prompt_full_path = prompt_dir + std::to_string(i) + ".txt";
        std::string prompt           = get_context(prompt_full_path);

        std::string file_full_path = file_dir + std::to_string(i) + ".txt";
        std::string file           = get_context(file_full_path);

        VideoDecoder video_decoder(file);
        video_decoder.decode();
        auto pcmf32_data = video_decoder.get_audio_pcmf32();
        auto image_list  = video_decoder.get_video_buffer();

        std::string resp_str = minicpmo.chat("", image_list, pcmf32_data, "en", prompt, false, true);

        std::cout << "[" << i << "] " << "response: " << resp_str << std::endl;
        dump_context("./out/" + std::to_string(i) + ".txt", resp_str);

        minicpmo.reset();
    }
}
