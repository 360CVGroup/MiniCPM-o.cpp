#include "minicpmo.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include "llama-model.h"
#include "llama.h"
#include "outetts.h"
#include "sampling.h"
#include "siglip.h"
#include "utils.h"
#include "whisper_encoder.h"
#ifdef USE_OPENMP
#include <omp.h>
#include <thread>
#endif

namespace edge {

struct common_params get_minicpmo_default_llm_params() {
    struct common_params params;
    params.split_mode                            = LLAMA_SPLIT_MODE_NONE;
    params.n_gpu_layers                          = 29;
    params.cpuparams.n_threads                   = 64;
    params.cpuparams_batch.n_threads             = 64;
    params.sampling.top_k                        = 100;
    params.sampling.top_p                        = 0.8;
    params.sampling.temp                         = 0.5;
    params.sampling.penalty_repeat               = 1.05;
    params.speculative.n_min                     = 5;
    params.speculative.p_min                     = 0.9;
    params.speculative.cpuparams.n_threads       = 64;
    params.speculative.cpuparams_batch.n_threads = 64;
    return params;
}

static const char* sample(struct common_sampler* smpl,
                          struct llama_context* ctx_llama,
                          int* n_past) {
    const llama_token id = common_sampler_sample(smpl, ctx_llama, -1);
    common_sampler_accept(smpl, id, true);

    const llama_model* model = llama_get_model(ctx_llama);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    static std::string ret;
    if (llama_vocab_is_eog(vocab, id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

void MiniCPMO::_image_preprocess(const image_buf<uint8_t>& img, std::vector<image_buf<float>>& res_imgs, int max_slice_nums) {
    std::vector<std::vector<image_buf<uint8_t>>> imgs = uhd_slice_image(img, max_slice_nums);

    int n_slice = 0;
    for (size_t i = 0; i < imgs.size(); ++i) {
        n_slice += imgs[i].size();
    }
    res_imgs.resize(n_slice);
    int idx = 0;
    for (size_t i = 0; i < imgs.size(); ++i) {
        for (size_t j = 0; j < imgs[i].size(); ++j) {
            image_buf<float> res;
            normalize_image_u8_to_f32(imgs[i][j], res, this->vpm_->get_image_mean().data(), this->vpm_->get_image_std().data());
            res_imgs[idx++] = res;
        }
    }
}

MiniCPMO::MiniCPMO(minicpmo_params params) {
    // needed to initialize f16 tables
    llama_backend_init();

#ifdef USE_OPENMP
    const int nthreads = std::max(1U, std::thread::hardware_concurrency() / 2);
    omp_set_num_threads(std::getenv("OMP_NUM_THREADS") ? std::stoi(std::getenv("OMP_NUM_THREADS")) : nthreads);
    setenv("OMP_PROC_BIND", "true", false);
    setenv("OMP_PLACES", "cores", false);
#endif

    vpm_ = new Siglip(params.vpm_path, false, 1);
    apm_ = new WhisperEncoder(params.apm_path, "medium");

    size_t audio_embed_size = 0;
    audio_embed_size += apm_->get_audio_ctx_length() * n_embedding_length_;
    audio_embed_size /= 30;  // default audio ctx length 30s, streaming mode only support 1s
    audio_embed_size /= 2;   // pooling output, shrink to 1/2
    audio_embed_out_.resize(audio_embed_size);
    const int num_max_patches = 1;  // TODO: check whether need 10 patches
    image_embd_out_.resize(this->vpm_->_embd_nbytes() * num_max_patches / sizeof(float));

    params_ = params.llm_params;
    llama_numa_init(params_.numa);
    llama_model_params llm_params       = common_model_params_to_llama(params_);
    llama_context_params llm_ctx_params = common_context_params_to_llama(params_);

    llama_model_ = llama_model_load_from_file(params.llm_path.c_str(), llm_params);
    llama_ctx_   = llama_init_from_model(llama_model_, llm_ctx_params);
    smpl_        = common_sampler_init(llama_model_, this->params_.sampling);

    this->token_embed(omni_strm_pre_token_, "<unit><image>");
    this->token_embed(omni_strm_mid_token_, "</image><|audio_start|>");
    this->token_embed(omni_strm_post_token_, "<|audio_end|>");
    omni_strm_embd_inp_.resize(audio_embed_out_.size() + image_embd_out_.size() + omni_strm_pre_token_.size() + omni_strm_mid_token_.size() + omni_strm_post_token_.size());
    // fill token embed here
    size_t offset = 0;
    // <unit><image>
    std::memcpy(omni_strm_embd_inp_.data() + offset, omni_strm_pre_token_.data(), omni_strm_pre_token_.size() * sizeof(float));
    offset += omni_strm_pre_token_.size();
    // image_embed
    // std::memcpy(omni_strm_embd_inp_.data() + offset, image_embd_out_.data(), image_embd_out_.size() * sizeof(float));
    offset += image_embd_out_.size();
    // </image><|audio_start|>
    std::memcpy(omni_strm_embd_inp_.data() + offset, omni_strm_mid_token_.data(), omni_strm_mid_token_.size() * sizeof(float));
    offset += omni_strm_mid_token_.size();
    // pcmf32
    // std::memcpy(omni_strm_embd_inp_.data() + offset, audio_embed_out_.data(), audio_embed_out_.size() * sizeof(float));
    offset += audio_embed_out_.size();
    // <|audio_end|>
    std::memcpy(omni_strm_embd_inp_.data() + offset, omni_strm_post_token_.data(), omni_strm_post_token_.size() * sizeof(float));

    // 初始化文本到语音转换模型
    if (!params.ttc_model_path.empty() && !params.cts_model_path.empty()) {
        tts_ = new Outetts(params.ttc_model_path, params.cts_model_path);
    }
}

// execute ggml_get_rows for embedding weight
void MiniCPMO::token_embed(std::vector<float>& out, std::string str, bool add_bos) {
    std::string str2                  = str;
    const llama_vocab* vocab          = llama_model_get_vocab(llama_model_);
    std::vector<llama_token> embd_inp = common_tokenize(vocab, str2, add_bos, true);
    int n_tokens                      = embd_inp.size();
    omni_strm_n_tokens_ += n_tokens;

    out.resize(llama_model_->tok_embd->ne[0] * n_tokens);

    // inplace running
    {
        ggml_backend_t backend = ggml_backend_cpu_init();
        std::vector<uint8_t> buf_compute_meta;
        buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
        ggml_gallocr_t compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        struct ggml_init_params params{
            buf_compute_meta.size(),
            buf_compute_meta.data(),
            true};
        ggml_context* ctx = ggml_init(params);

        ggml_cgraph* gf         = ggml_new_graph(ctx);
        ggml_tensor* inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        // ggml_set_input(inp_tokens);
        ggml_tensor* token_embedding = ggml_get_rows(ctx, llama_model_->tok_embd, inp_tokens);
        // ggml_set_output(token_embedding);
        ggml_build_forward_expand(gf, token_embedding);
        ggml_gallocr_alloc_graph(compute_alloc, gf);

        // set input
        ggml_backend_tensor_set(inp_tokens, embd_inp.data(), 0, ggml_nbytes(inp_tokens));
        // compute
        ggml_backend_graph_compute(backend, gf);
        // get output
        ggml_backend_tensor_get(token_embedding, out.data(), 0, ggml_nbytes(token_embedding));

        // free resource
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
}

void MiniCPMO::single_prefill(std::vector<float>& image_embed, std::vector<float>& audio_embed) {
    prefill_finished_ = false;  // for streaming generation
    // infinite text generation via context shifting
    const int preserve_tokens = (n_image_tokens_ + n_audio_tokens_ + 10);
    if (n_past_ + preserve_tokens >= params_.n_ctx) {
        const int n_left    = n_past_ - params_.n_keep;
        const int n_discard = n_left / 2;
        LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                n_past_, n_left, params_.n_ctx, params_.n_keep, n_discard);

        llama_kv_self_seq_rm(llama_ctx_, 0, params_.n_keep, params_.n_keep + n_discard);
        llama_kv_self_seq_add(llama_ctx_, 0, params_.n_keep + n_discard, n_past_, -n_discard);

        n_past_ -= n_discard;

        LOG_DBG("after swap: n_past = %d\n", n_past_);
    }
    // n_past_ += omni_strm_n_tokens_;

    size_t offset = 0;
    // <unit><image>
    // std::memcpy(omni_strm_embd_inp_.data() + offset, omni_strm_pre_token_.data(), omni_strm_pre_token_.size() * sizeof(float));
    offset += omni_strm_pre_token_.size();
    // image_embed
    std::memcpy(omni_strm_embd_inp_.data() + offset, image_embed.data(), image_embed.size() * sizeof(float));
    offset += image_embed.size();
    // </image><|audio_start|>
    // std::memcpy(omni_strm_embd_inp_.data() + offset, omni_strm_mid_token_.data(), omni_strm_mid_token_.size() * sizeof(float));
    offset += omni_strm_mid_token_.size();
    // pcmf32
    std::memcpy(omni_strm_embd_inp_.data() + offset, audio_embed.data(), audio_embed.size() * sizeof(float));
    // offset += audio_embed.size();
    // <|audio_end|>
    // std::memcpy(omni_strm_embd_inp_.data() + offset, omni_strm_post_token_.data(), omni_strm_post_token_.size() * sizeof(float));

    minicpmo_embd_batch omni_embd_batch = minicpmo_embd_batch(omni_strm_embd_inp_.data(), n_image_tokens_ + n_audio_tokens_ + omni_strm_n_tokens_, n_past_, 0);
    llama_decode(llama_ctx_, omni_embd_batch.batch);
    n_past_ += (n_image_tokens_ + n_audio_tokens_ + omni_strm_n_tokens_);
}

void MiniCPMO::streaming_prefill(image_buf<uint8_t>& image, std::vector<float>& pcmf32, int max_slice_nums) {
    prefill_finished_ = false;  // for streaming generation
    // n_ctx for 1s streaming mode
    this->apm_->set_exp_n_audio_ctx(n_audio_tokens_ * 2);  // 50 tokens

    const int64_t t_img_enc_start_us = ggml_time_us();
    // TODO: support large slice number
    std::vector<edge::image_buf<float>> image_inp;
    this->_image_preprocess(image, image_inp, max_slice_nums);
    for (size_t i = 0; i < image_inp.size(); ++i) {
        auto patched = reshape_by_patch(image_inp[i], vpm_->patch_size_);
        this->vpm_->forward(patched, image_inp[i].nx, image_inp[i].ny, image_embd_out_);
    }
    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms             = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;
    LOG_INF("image encoded in %8.2f ms by Siglip (%8.2f ms per image patch)\n", t_img_enc_ms, t_img_enc_ms / vpm_->n_patches_);

    const int sample_rate = 16000;  // 16k
    // streaming mode only receive 1s audio
    pcmf32.resize(sample_rate);

    this->apm_->forward(pcmf32, audio_embed_out_);
    // this->prefill(image_embd_out_, audio_embed_out_);
    this->single_prefill(image_embd_out_, audio_embed_out_);
}

static bool is_valid_utf8(const std::string& str) {
    int expected_continuations = 0;

    for (char c : str) {
        unsigned char byte = static_cast<unsigned char>(c);

        if (expected_continuations > 0) {
            // check if it is a valid continuation byte (0x80-0xBF)
            if ((byte & 0xC0) != 0x80) {
                return false;
            }
            expected_continuations--;
        } else {
            // Check for new UTF-8 start bytes
            if ((byte & 0x80) == 0) {
                // single-byte characters (0x00-0x7F)
                continue;
            } else if ((byte & 0xE0) == 0xC0) {
                // double-byte characters (0xC0-0xDF)
                expected_continuations = 1;
            } else if ((byte & 0xF0) == 0xE0) {
                // three-byte characters (0xE0-0xEF)
                expected_continuations = 2;
            } else if ((byte & 0xF8) == 0xF0) {
                // four-byte characters (0xF0-0xF7)
                expected_continuations = 3;
            } else {
                // invalid start byte
                return false;
            }
        }
    }

    return expected_continuations == 0;
}

std::string MiniCPMO::streaming_generate(std::string user_prompt) {
    if (prefill_finished_ == false) {
        prefill_finished_ = true;

        const std::string builtin_prompt = "<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>";
        if (user_prompt.length() == 0) {
            user_prompt = "assistant\n";
        }
        eval_string(llama_ctx_, ("<|im_end|>\n<|im_start|>" + user_prompt + builtin_prompt).c_str(), params_.n_batch, &n_past_, false);
        n_sample_  = 0;
        stop_smpl_ = false;
        smpl_      = common_sampler_init(llama_model_, this->params_.sampling);
    }

    const int max_tgt_len = params_.n_predict < 0 ? 256 : params_.n_predict;
    std::string response  = "";
    if (!stop_smpl_ && n_sample_ < max_tgt_len) {
        const char* tmp = sample(smpl_, llama_ctx_, &n_past_);
        n_sample_++;
        // TODO: improve utf8 handling
        utf8_str_ += tmp;
        while (!is_valid_utf8(utf8_str_)) {
            if (!stop_smpl_ && n_sample_ < max_tgt_len) {
                tmp = sample(smpl_, llama_ctx_, &n_past_);
                n_sample_++;
                utf8_str_ += tmp;
            } else {
                break;
            }
        }
        if (strcmp(utf8_str_.c_str(), "</s>") == 0 || strstr(utf8_str_.c_str(), "###")  // Yi-VL behavior
            || strstr(utf8_str_.c_str(), "<user>")                                      // minicpm-v
        ) {
            stop_smpl_ = true;
            // response   = "\n";  // end of sentence
        } else {
            response = utf8_str_;
        }
        utf8_str_.clear();
    }
    return response;
}

void MiniCPMO::chat_generate(std::string user_prompt, bool stream) {
    this->_chat(user_prompt, stream);
}

std::string MiniCPMO::chat(std::string audio_output_path, std::vector<image_buf<uint8_t>>& image_buf_list, std::vector<float>& pcmf32, std::string language, std::string user_prompt, bool stream_out, bool eval_system) {
    std::vector<std::vector<float>> image_embed_list;
    for (auto const& image_buf : image_buf_list) {
        const int64_t t_img_enc_start_us = ggml_time_us();

        std::vector<edge::image_buf<float>> image_inp;
        // TODO: support large slice number
        this->_image_preprocess(image_buf, image_inp, 1);

        for (size_t i = 0; i < image_inp.size(); ++i) {
            auto patched = reshape_by_patch(image_inp[i], vpm_->patch_size_);
            this->vpm_->forward(patched, image_inp[i].nx, image_inp[i].ny, image_embd_out_);
        }
        image_embed_list.emplace_back(image_embd_out_);
        const int64_t t_img_enc_end_us = ggml_time_us();
        float t_img_enc_ms             = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;
        LOG_INF("image encoded in %8.2f ms by Siglip (%8.2f ms per image patch)\n", t_img_enc_ms, t_img_enc_ms / vpm_->n_patches_);
    }

    const int sample_rate         = 16000;  // 16k
    const float silence_pad_value = 0.f;
    // whisper need trim pcmf32 to chunks of 30s
    const int chunk_seconds = 30;
    const size_t chunk_size = sample_rate * chunk_seconds;
    const size_t chunks     = (pcmf32.size() - 1) / chunk_size + 1;

    std::vector<float> audio_embed_out_data(audio_embed_out_.size() * chunks, 0);

    size_t n_pieces     = pcmf32.size() / chunk_size;
    size_t epilogue     = pcmf32.size() % chunk_size;
    size_t offset_data  = 0;
    size_t offset_embed = 0;

    for (size_t i = 0; i < n_pieces; ++i) {
        std::vector<float> piece_data(chunk_size, silence_pad_value);
        std::memcpy(piece_data.data(), pcmf32.data() + offset_data, chunk_size * sizeof(float));
        this->apm_->forward(piece_data, audio_embed_out_);
        std::memcpy(audio_embed_out_data.data() + offset_embed, audio_embed_out_.data(), audio_embed_out_.size() * sizeof(float));
        offset_data += chunk_size;
        offset_embed += audio_embed_out_.size();
    }
    if (epilogue > 0) {
        std::vector<float> piece_data(chunk_size, silence_pad_value);
        std::memcpy(piece_data.data(), pcmf32.data() + offset_data, epilogue * sizeof(float));
        this->apm_->forward(piece_data, audio_embed_out_);
        std::memcpy(audio_embed_out_data.data() + offset_embed, audio_embed_out_.data(), audio_embed_out_.size() * sizeof(float));
    }

    int n_frame    = image_embed_list.size();  // matching the image list length
    int n_aud_embd = n_audio_tokens_ * n_embedding_length_;
    std::vector<std::vector<float>> audio_embed_list;
    for (int i = 0; i < n_frame; ++i) {
        std::vector<float> audio_embed(n_aud_embd, 0);
        std::memcpy(audio_embed.data(), audio_embed_out_data.data() + i * n_aud_embd, n_aud_embd * sizeof(float));
        audio_embed_list.emplace_back(audio_embed);
    }

    // llm processing
    if (eval_system) {
        this->eval_system_prompt(language);
    }
    for (size_t i = 0; i < image_embed_list.size(); ++i) {
        this->single_prefill(image_embed_list[i], audio_embed_list[i]);
    }
    std::string resp_str = this->_chat(user_prompt, stream_out);

    // 如果TTS模型已初始化，则将生成的文本转换为语音
    if (tts_ != nullptr) {
        LOG_INF("Converting generated text to speech: %s\n", audio_output_path.c_str());
        if (text_to_speech(resp_str, audio_output_path)) {
            LOG_INF("Speech generated successfully: %s\n", audio_output_path.c_str());
        } else {
            LOG_ERR("Failed to generate speech from text\n");
        }
    }
    return resp_str;
}

void MiniCPMO::eval_system_prompt(std::string& language) {
    std::string system_prompt = "<|im_start|>user\nYou are a helpful assistant. You can accept video, audio and text input and output voice and text. <|im_end|>\n<|im_start|>user\n";
    if (language == "zh") {
        system_prompt = "<|im_start|>user\n你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本，并使用中文进行回答。<|im_end|>\n<|im_start|>user\n";
    }
    eval_string(llama_ctx_, system_prompt.c_str(), params_.n_batch, &n_past_, false);
}

std::string MiniCPMO::_chat(std::string user_prompt, bool stream) {
    const std::string builtin_prompt = "<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>";
    if (user_prompt.length() == 0) {
        user_prompt = "assistant\n";
    }
    eval_string(llama_ctx_, ("<|im_end|>\n<|im_start|>" + user_prompt + builtin_prompt).c_str(), params_.n_batch, &n_past_, false);

    struct common_sampler* smpl = common_sampler_init(llama_model_, this->params_.sampling);
    const int max_tgt_len       = params_.n_predict < 0 ? 256 : params_.n_predict;
    std::string response;
    bool have_tmp = false;
    printf("\n");
    for (int i = 0; i < max_tgt_len; i++) {
        const char* tmp = sample(smpl, llama_ctx_, &n_past_);

        response += tmp;
        if (strcmp(tmp, "</s>") == 0) {
            if (!have_tmp) {
                continue;
            }
            break;
        }
        if (strstr(tmp, "###"))
            break;  // Yi-VL behavior
        have_tmp = true;
        if (stream) {
            printf("%s", tmp);
        }
        if (strstr(response.c_str(), "<user>"))
            break;  // minicpm-v

        fflush(stdout);
    }
    if (stream) {
        printf("\n");
    }
    return response;
}

void MiniCPMO::reset() {
    n_past_ = 0;
    llama_kv_self_clear(llama_ctx_);
}

void MiniCPMO::apm_kv_clear() {
    this->apm_->kv_cache_clear();
}

void MiniCPMO::apm_streaming_mode(bool streaming) {
    this->apm_->set_streaming_mode(streaming);
}

// 文本到语音转换
bool MiniCPMO::text_to_speech(const std::string& text, const std::string& output_wav) {
    if (!tts_) {
        LOG_ERR("TTS model not initialized\n");
        return false;
    }

    LOG_INF("Converting text to speech: %s\n", text.c_str());

    // 使用outetts来处理文本到语音转换
    std::vector<float> audio_data;
    if (!tts_->text_to_speech(text, audio_data)) {
        LOG_ERR("Failed to convert text to speech\n");
        return false;
    }

    // 保存生成的音频数据为WAV文件
    if (!tts_->save_wav(output_wav, audio_data)) {
        LOG_ERR("Failed to save audio to %s\n", output_wav.c_str());
        return false;
    }

    LOG_INF("Speech generated and saved to %s\n", output_wav.c_str());
    return true;
}

}  // namespace edge
