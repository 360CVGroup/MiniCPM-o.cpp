#ifndef INCLUDE_WHISPER_ENCODER_N_
#define INCLUDE_WHISPER_ENCODER_N_

#include <map>
#include <set>
#include <string>
#include <vector>
#include "ggml-backend.h"
#include "utils.h"

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT 400
#define WHISPER_HOP_LENGTH 160
#define WHISPER_CHUNK_SIZE 30
#define WHISPER_MAX_NODES 4096

namespace edge {

enum whisper_alignment_heads_preset {
    WHISPER_AHEADS_NONE,
    WHISPER_AHEADS_N_TOP_MOST,  // All heads from the N-top-most text-layers
    WHISPER_AHEADS_CUSTOM,
    WHISPER_AHEADS_TINY_EN,
    WHISPER_AHEADS_TINY,
    WHISPER_AHEADS_BASE_EN,
    WHISPER_AHEADS_BASE,
    WHISPER_AHEADS_SMALL_EN,
    WHISPER_AHEADS_SMALL,
    WHISPER_AHEADS_MEDIUM_EN,
    WHISPER_AHEADS_MEDIUM,
    WHISPER_AHEADS_LARGE_V1,
    WHISPER_AHEADS_LARGE_V2,
    WHISPER_AHEADS_LARGE_V3,
    WHISPER_AHEADS_LARGE_V3_TURBO,
};

struct whisper_context_params {
    int32_t n_threads;
    bool use_gpu;
    bool flash_attn;
    int gpu_device;  // CUDA device

    // [EXPERIMENTAL] Token-level timestamps with DTW
    bool dtw_token_timestamps;
    enum whisper_alignment_heads_preset dtw_aheads_preset;

    int dtw_n_top;
    // struct whisper_aheads dtw_aheads;

    size_t dtw_mem_size;  // TODO: remove
};

struct whisper_context_params whisper_context_default_params();

// ggml_backend_sched wrapper for whisper usage
struct whisper_sched {
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> meta;
};

// default hparams (Whisper tiny)
struct whisper_hparams {
    int32_t n_vocab       = 51864;
    int32_t n_audio_ctx   = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head  = 6;
    int32_t n_audio_layer = 4;
    int32_t n_text_ctx    = 448;
    int32_t n_text_state  = 384;
    int32_t n_text_head   = 6;
    int32_t n_text_layer  = 4;
    int32_t n_mels        = 80;
    int32_t ftype         = 1;
    float eps             = 1e-5f;
};

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

// audio encoding layer
struct whisper_layer_encoder {
    // encoder.blocks.*.attn_ln
    struct ggml_tensor* attn_ln_0_w;
    struct ggml_tensor* attn_ln_0_b;

    // encoder.blocks.*.attn.out
    struct ggml_tensor* attn_ln_1_w;
    struct ggml_tensor* attn_ln_1_b;

    // encoder.blocks.*.attn.query
    struct ggml_tensor* attn_q_w;
    struct ggml_tensor* attn_q_b;

    // encoder.blocks.*.attn.key
    struct ggml_tensor* attn_k_w;

    // encoder.blocks.*.attn.value
    struct ggml_tensor* attn_v_w;
    struct ggml_tensor* attn_v_b;

    // encoder.blocks.*.mlp_ln
    struct ggml_tensor* mlp_ln_w;
    struct ggml_tensor* mlp_ln_b;

    // encoder.blocks.*.mlp.0
    struct ggml_tensor* mlp_0_w;
    struct ggml_tensor* mlp_0_b;

    // encoder.blocks.*.mlp.2
    struct ggml_tensor* mlp_1_w;
    struct ggml_tensor* mlp_1_b;
};

// available whisper models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_TINY,
    MODEL_BASE,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
};

struct whisper_encoder_model {
    e_model type = MODEL_UNKNOWN;

    whisper_hparams hparams;
    whisper_filters filters;

    // encoder.positional_embedding
    struct ggml_tensor* e_pe;

    // encoder.conv1
    struct ggml_tensor* e_conv_1_w;
    struct ggml_tensor* e_conv_1_b;

    // encoder.conv2
    struct ggml_tensor* e_conv_2_w;
    struct ggml_tensor* e_conv_2_b;

    // encoder.ln_post
    struct ggml_tensor* e_ln_w;
    struct ggml_tensor* e_ln_b;

    std::vector<whisper_layer_encoder> layers_encoder;

    // ggml context that contains all the meta information about the model tensors
    struct ggml_context* ctx = nullptr;

    // the model backend data is read-only and can be shared between processors
    ggml_backend_buffer_t buffer = nullptr;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor*> tensors;

    // audio projector
    struct ggml_tensor* proj_1_w;  // linear1 weight
    struct ggml_tensor* proj_1_b;  // linear1 bias
    struct ggml_tensor* proj_2_w;  // linear2 weight
    struct ggml_tensor* proj_2_b;  // linear2 bias
};

struct whisper_kv_cache {
    uint32_t size = 0;

    // total tokens in kv cache
    uint32_t n = 0;

    std::vector<ggml_tensor*> k_l;  // per layer
    std::vector<ggml_tensor*> v_l;

    ggml_backend_buffer_t buffer = nullptr;

    std::vector<uint8_t> ctx_buf;
};

class WhisperEncoder {
public:
    WhisperEncoder(std::string model_path, std::string dtw_type = "medium", size_t num_audio_ctx = 1500);
    void forward(std::vector<float>& samples, std::vector<float>& embed_enc_data_out);
    size_t get_audio_ctx_length();
    void set_exp_n_audio_ctx(int32_t exp_n_ctx);

    void kv_cache_init(int64_t n_state, int64_t n_layer, int n_kv_ctx);
    void kv_cache_free();
    void kv_cache_clear();

    void set_streaming_mode(bool streaming) {
        streaming_ = streaming;
    }

    WhisperEncoder() = delete;
    ~WhisperEncoder() {
        if (model_ != nullptr) {
            ggml_free(model_->ctx);
            ggml_backend_buffer_free(model_->buffer);
            model_->ctx    = nullptr;
            model_->buffer = nullptr;
            delete model_;
            model_ = nullptr;
        }

        ggml_backend_sched_free(sched_conv_.sched);
        ggml_backend_sched_free(sched_encode_.sched);
        for (auto& backend : backends_) {
            ggml_backend_free(backend);
        }
        GGUF_FREE(ctx_gguf_);
        GGML_FREE(ctx_ggml_);
        if (compute_alloc_ != nullptr) {
            ggml_gallocr_free(compute_alloc_);
            compute_alloc_ = nullptr;
        }
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
        if (params_buffer_ != nullptr) {
            ggml_backend_buffer_free(params_buffer_);
            params_buffer_ = nullptr;
        }
        // kv_self buffer free
        if (kv_self_.buffer != nullptr) {
            ggml_backend_buffer_free(kv_self_.buffer);
        }
        GGML_FREE(ctx_kv_self_);
    }

protected:
    void load_model(const std::string& model_path, const int verbosity, size_t num_audio_ctx);
    void build_graph();
    struct ggml_cgraph* _whisper_build_graph_conv();
    struct ggml_cgraph* _whisper_build_graph_encoder();
    struct ggml_cgraph* _stream_whisper_build_graph_encoder();
    bool whisper_encode_internal(const int mel_offset, const int n_threads);

private:
    bool streaming_                 = false;
    whisper_context_params cparams_ = whisper_context_default_params();
    // result of the encoder
    struct ggml_tensor* embd_conv_ = nullptr;
    struct ggml_tensor* embd_enc_  = nullptr;

    // struct ggml_tensor* dump_key_tensor_        = nullptr;
    // struct ggml_tensor* dump_origin_key_tensor_ = nullptr;

    // self-attention KV cache for streaming input
    whisper_kv_cache kv_self_;
    struct ggml_context* ctx_kv_self_ = nullptr;

    int iter_ = -1;

    // helpers for GPU offloading
    std::vector<float> inp_mel_;
    whisper_mel mel_;

    // - stores meta info about the intermediate tensors into the `meta` buffers
    whisper_sched sched_conv_;
    whisper_sched sched_encode_;

    // [EXPERIMENTAL] speed-up techniques
    // now for streaming usage
    int32_t exp_n_audio_ctx_ = 0;  // 0 - use default

    // TODO: check kv format
    // ggml_type wtype_ = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
    // ggml_type itype_ = ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)
    ggml_type wtype_ = ggml_type::GGML_TYPE_F32;  // weight type (FP32 / FP16 / QX)
    ggml_type itype_ = ggml_type::GGML_TYPE_F32;  // intermediate type (FP32 or FP16)

    whisper_encoder_model* model_ = nullptr;

    struct gguf_context* ctx_gguf_ = nullptr;
    struct ggml_context* ctx_ggml_ = nullptr;

    std::vector<uint8_t> buf_compute_meta_;

    // memory buffers to evaluate the model
    ggml_backend_buffer_t params_buffer_ = nullptr;

    ggml_backend_t backend_ = nullptr;
    std::vector<ggml_backend_t> backends_;
    ggml_gallocr_t compute_alloc_ = nullptr;
};

void fft(float* in, int N, float* out);
void dft(const float* in, int N, float* out);
void log_mel_spectrogram_worker_thread(int ith, const float* hann, const std::vector<float>& samples, int n_samples, int frame_size, int frame_step, int n_threads, const whisper_filters& filters, whisper_mel& mel);
bool log_mel_spectrogram(const float* samples, const int n_samples, const int /*sample_rate*/, const int frame_size, const int frame_step, const int n_mel, const int n_threads, const whisper_filters& filters, const bool debug, whisper_mel& mel);
bool whisper_sched_graph_init(struct whisper_sched& allocr, std::vector<ggml_backend_t> backends, std::function<struct ggml_cgraph*()>&& get_graph);
bool ggml_graph_compute_helper(ggml_backend_sched_t sched, struct ggml_cgraph* graph, int n_threads);
size_t whisper_sched_size(struct whisper_sched& allocr);
ggml_backend_t whisper_backend_init_gpu(const whisper_context_params& params);
std::vector<ggml_backend_t> whisper_backend_init(const whisper_context_params& params);

}  // namespace edge

#endif  // INCLUDE_WHISPER_ENCODER_N_
