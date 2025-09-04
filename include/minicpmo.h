#ifndef INCLUDE_MINICPMO_H_
#define INCLUDE_MINICPMO_H_

#include <common/common.h>
#include <cstdint>
#include "llama.h"
#include "outetts.h"
#include "siglip.h"
#include "whisper_encoder.h"

#define FREE_MODAL_HEAD(model)  \
    do {                        \
        if (model != nullptr) { \
            delete model;       \
            model = nullptr;    \
        }                       \
    } while (0);

namespace edge {
struct common_params get_minicpmo_default_llm_params();

struct minicpmo_embd_batch {
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id*> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch;
    minicpmo_embd_batch(float* embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos.resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids.resize(n_tokens + 1);
        logits.resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0]       = seq_id;
        seq_ids[n_tokens] = nullptr;
        batch             = {
            /*n_tokens       =*/n_tokens,
            /*tokens         =*/nullptr,
            /*embd           =*/embd,
            /*pos            =*/pos.data(),
            /*n_seq_id       =*/n_seq_id.data(),
            /*seq_id         =*/seq_ids.data(),
            /*logits         =*/logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos[i]      = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i]   = seq_id_0.data();
            batch.logits[i]   = false;
        }
    }
};

struct minicpmo_params {
    std::string vpm_path;
    std::string apm_path;
    std::string llm_path;
    std::string ttc_model_path;  // 文本到编码模型路径
    std::string cts_model_path;  // 编码到声音模型路径
    struct common_params llm_params;
};

class MiniCPMO {
public:
    MiniCPMO() = delete;

    MiniCPMO(minicpmo_params params);

    void single_prefill(std::vector<float>& image_embed, std::vector<float>& audio_embed);
    void streaming_prefill(image_buf<uint8_t>& image, std::vector<float>& pcmf32, int max_slice_nums = 1);
    std::string streaming_generate(std::string user_prompt);
    void chat_generate(std::string user_prompt, bool stream);
    void eval_system_prompt(std::string& language);
    void token_embed(std::vector<float>& out, std::string str, bool add_bos = false);

    std::string chat(std::string audio_output_path, std::vector<image_buf<uint8_t>>& image_bytes_list, std::vector<float>& pcmf32, std::string language = "en", std::string user_prompt = "", bool stream_out = true, bool eval_system = true);
    std::string _chat(std::string user_prompt, bool stream = true);

    void reset();
    void apm_kv_clear();
    void apm_streaming_mode(bool streaming);

    // 新增：将文本转换为语音并保存为wav文件
    bool text_to_speech(const std::string& text, const std::string& output_wav);

    ~MiniCPMO() {
        if (llama_ctx_ != nullptr) {
            llama_free(llama_ctx_);
            llama_ctx_ = nullptr;
        }
        if (llama_model_ != nullptr) {
            llama_model_free(llama_model_);
            llama_model_ = nullptr;
            llama_backend_free();
        }
        FREE_MODAL_HEAD(vpm_);
        FREE_MODAL_HEAD(apm_);
        FREE_MODAL_HEAD(tts_);
    }

protected:
    void _image_preprocess(const image_buf<uint8_t>& img, std::vector<image_buf<float>>& res_imgs, int max_slice_nums = 1);

private:
    std::vector<float> audio_embed_out_;
    std::vector<float> image_embd_out_;
    common_params params_;
    Siglip* vpm_         = nullptr;
    WhisperEncoder* apm_ = nullptr;
    Outetts* tts_        = nullptr;  // 新增：文本到语音转换模型

    struct llama_model* llama_model_ = nullptr;
    struct llama_context* llama_ctx_ = nullptr;

    // omni common-token cache
    int omni_strm_n_tokens_ = 0;
    std::vector<float> omni_strm_pre_token_;
    std::vector<float> omni_strm_mid_token_;
    std::vector<float> omni_strm_post_token_;
    std::vector<float> omni_strm_embd_inp_;

    int n_past_ = 0;

    int n_image_tokens_ = 64;
    int n_audio_tokens_ = 25;

    const int n_embedding_length_ = 3584;

    // streaming generate
    bool prefill_finished_       = false;
    int n_sample_                = 0;
    bool stop_smpl_              = false;
    std::string utf8_str_        = "";
    struct common_sampler* smpl_ = nullptr;
};
}  // namespace edge

#endif  // INCLUDE_MINICPMO_H_
