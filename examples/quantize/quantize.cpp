#include "arg.h"
#include "base64.hpp"
#include "common.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include "log.h"
#include "sampling.h"
#include "whisper_encoder.h"

#include <cassert>
#include <fstream>
#include <regex>

using namespace edge;
static void print_usage(int argc, char** argv) {
    (void)argc;

    fprintf(stderr, "usage: %s /path/to/ggml-model-f32.gguf /path/to/ggml-model-quantized.gguf type\n", argv[0]);
    fprintf(stderr, "  type = 2 - q4_0\n");
    fprintf(stderr, "  type = 3 - q4_1\n");
    fprintf(stderr, "  type = 6 - q5_0\n");
    fprintf(stderr, "  type = 7 - q5_1\n");
    fprintf(stderr, "  type = 8 - q8_0\n");
}

gguf_context* ctx_gguf_;
ggml_context* ctx_ggml_;

void whisper_encoder_init(std::string fname_inp, int verbosity = 1) {
    auto model_ = new edge::whisper_encoder_model();

    const char* fname         = fname_inp.c_str();
    struct ggml_context* meta = nullptr;
    struct gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = &meta;

    ctx_gguf_ = gguf_init_from_file(fname, params);

    if (verbosity >= 1) {
        const int n_tensors           = gguf_get_n_tensors(ctx_gguf_);
        const int n_kv                = gguf_get_n_kv(ctx_gguf_);
        const int ftype               = get_u32(ctx_gguf_, "general.file_type");
        const std::string ftype_str   = get_ftype(ftype);
        const int idx_desc            = get_key_idx(ctx_gguf_, "general.description");
        const std::string description = gguf_get_val_str(ctx_gguf_, idx_desc);
        const int idx_name            = gguf_find_key(ctx_gguf_, "general.name");
        if (idx_name != -1) {  // make name optional temporarily as some of the uploaded models missing it due to a bug
            const std::string name = gguf_get_val_str(ctx_gguf_, idx_name);
            LOG_INF("%s: model name:   %s\n", __func__, name.c_str());
        }
        LOG_INF("%s: description:  %s\n", __func__, description.c_str());
        LOG_INF("%s: GGUF version: %d\n", __func__, gguf_get_version(ctx_gguf_));
        LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx_gguf_));
        LOG_INF("%s: n_tensors:    %d\n", __func__, n_tensors);
        LOG_INF("%s: n_kv:         %d\n", __func__, n_kv);
        LOG_INF("%s: ftype:        %s\n", __func__, ftype_str.c_str());
        LOG_INF("\n");
    }

    const int n_tensors = gguf_get_n_tensors(ctx_gguf_);

    // kv
    const int n_kv = gguf_get_n_kv(ctx_gguf_);
    LOG_INF("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n",
            __func__, n_kv, n_tensors, fname);
    {
        std::map<enum ggml_type, uint32_t> n_type;

        for (int i = 0; i < n_tensors; i++) {
            enum ggml_type type = gguf_get_tensor_type(ctx_gguf_, i);

            n_type[type]++;
        }

        LOG_INF("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);
        for (int i = 0; i < n_kv; i++) {
            const char* name          = gguf_get_key(ctx_gguf_, i);
            const enum gguf_type type = gguf_get_kv_type(ctx_gguf_, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                    ? format("%s[%s,%ld]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(ctx_gguf_, i)), gguf_get_arr_n(ctx_gguf_, i))
                    : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(ctx_gguf_, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LOG_INF("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto& kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LOG_INF("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    // data
    size_t model_size = 0;
    {
        for (int i = 0; i < n_tensors; ++i) {
            const char* name        = gguf_get_tensor_name(ctx_gguf_, i);
            const size_t offset     = gguf_get_tensor_offset(ctx_gguf_, i);
            enum ggml_type type     = gguf_get_tensor_type(ctx_gguf_, i);
            struct ggml_tensor* cur = ggml_get_tensor(meta, name);
            size_t tensor_size      = ggml_nbytes(cur);
            model_size += tensor_size;
            if (verbosity >= 3) {
                LOG_INF("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                        __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    auto backend_ = ggml_backend_cpu_init();

    if (verbosity >= 1) {
        LOG_INF("%s: model size:     %.2f MB\n", __func__, model_size / 1024.0 / 1024.0);
        LOG_INF("%s: metadata size:  %.2f MB\n", __func__, ggml_get_mem_size(meta) / 1024.0 / 1024.0);
    }

    LOG_INF("%s: params backend buffer size = % 6.2f MB (%i tensors)\n", __func__, model_size / (1024.0 * 1024.0), n_tensors);

    // load tensors
    {
        std::vector<uint8_t> read_buf;
        struct ggml_init_params params;
        params.mem_size   = (n_tensors + 1) * ggml_tensor_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        ctx_ggml_ = ggml_init(params);
        if (!ctx_ggml_) {
            // LOG_ERR("%s: ggml_init() failed\n", __func__);
            gguf_free(ctx_gguf_);
            throw std::runtime_error(format("%s: ggml_init() failed\n", __func__));
        }

        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            gguf_free(ctx_gguf_);
            // LOG_ERR("cannot open model file for loading tensors\n");
            throw std::runtime_error(format("%s: cannot open model file for loading tensors\n", __func__));
        }

        // add tensors to context
        for (int i = 0; i < n_tensors; ++i) {
            const char* name        = gguf_get_tensor_name(ctx_gguf_, i);
            struct ggml_tensor* t   = ggml_get_tensor(meta, name);
            struct ggml_tensor* cur = ggml_dup_tensor(ctx_ggml_, t);
            ggml_set_name(cur, name);
        }

        // alloc memory and offload data
        ggml_backend_buffer_t params_buffer_ = ggml_backend_alloc_ctx_tensors(ctx_ggml_, backend_);
        for (int i = 0; i < n_tensors; ++i) {
            const char* name        = gguf_get_tensor_name(ctx_gguf_, i);
            struct ggml_tensor* cur = ggml_get_tensor(ctx_ggml_, name);
            const size_t offset     = gguf_get_data_offset(ctx_gguf_) + gguf_get_tensor_offset(ctx_gguf_, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                // LOG_ERR("%s: failed to seek for tensor %s\n", __func__, name);
                gguf_free(ctx_gguf_);
                throw std::runtime_error(format("%s: failed to seek for tensor %s\n", __func__, name));
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(params_buffer_)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char*>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char*>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        fin.close();
    }
    auto& hparams = model_->hparams;
    // hparams.n_vocab = get_u32(ctx_gguf_, "vocab_size");
    // hparams.n_audio_ctx = get_u32(ctx_gguf_, "max_source_positions");
    hparams.n_audio_state = get_u32(ctx_gguf_, "d_model");
    hparams.n_audio_head  = get_u32(ctx_gguf_, "encoder_attention_heads");
    hparams.n_audio_layer = get_u32(ctx_gguf_, "encoder_layers");
    // hparams.n_text_ctx = get_u32(ctx_gguf_, "max_length");
    hparams.n_text_state = get_u32(ctx_gguf_, "d_model");
    // hparams.n_text_head = get_u32(ctx_gguf_, "decoder_attention_heads");
    // hparams.n_text_layer = get_u32(ctx_gguf_, "decoder_layers");
    // hparams.n_mels = get_u32(ctx_gguf_, "num_mel_bins");
    hparams.ftype = get_u32(ctx_gguf_, "use_f16");

    // load mel filters
    {
        auto& filters = model_->filters;
        filters.n_mel = get_u32(ctx_gguf_, "n_mel");
        filters.n_fft = get_u32(ctx_gguf_, "n_fft");

        int idx_flt_data              = get_key_idx(ctx_gguf_, "filters");
        const float* filter_data_addr = (const float*)gguf_get_arr_data(ctx_gguf_, idx_flt_data);
        int n_flts                    = filters.n_mel * filters.n_fft;
        filters.data.resize(n_flts);
        std::memcpy(filters.data.data(), filter_data_addr, n_flts * sizeof(float));
    }

    // clang-format off
    std::string mver = "";
    if (hparams.n_audio_layer == 4) { model_->type = e_model::MODEL_TINY; }
    if (hparams.n_audio_layer == 6) { model_->type = e_model::MODEL_BASE; }
    if (hparams.n_audio_layer == 12) { model_->type = e_model::MODEL_SMALL; }
    if (hparams.n_audio_layer == 24) { model_->type = e_model::MODEL_MEDIUM; }
    if (hparams.n_audio_layer == 32) { model_->type = e_model::MODEL_LARGE; if (hparams.n_vocab == 51866) { mver = " v3"; } }
    // clang-format on

    const int n_audio_layer = hparams.n_audio_layer;
    model_->layers_encoder.resize(n_audio_layer);

    // encoder
    {
        model_->e_pe = get_tensor(ctx_ggml_, "encoder.positional_embedding");

        model_->e_conv_1_w = get_tensor(ctx_ggml_, "encoder.conv1.weight");
        model_->e_conv_1_b = get_tensor(ctx_ggml_, "encoder.conv1.bias");

        model_->e_conv_2_w = get_tensor(ctx_ggml_, "encoder.conv2.weight");
        model_->e_conv_2_b = get_tensor(ctx_ggml_, "encoder.conv2.bias");

        model_->e_ln_w = get_tensor(ctx_ggml_, "encoder.ln_post.weight");
        model_->e_ln_b = get_tensor(ctx_ggml_, "encoder.ln_post.bias");

        for (int i = 0; i < n_audio_layer; ++i) {
            auto& layer = model_->layers_encoder[i];

            layer.mlp_ln_w = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".mlp_ln.weight");
            layer.mlp_ln_b = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".mlp_ln.bias");

            layer.mlp_0_w = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".mlp.0.weight");
            layer.mlp_0_b = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".mlp.0.bias");

            layer.mlp_1_w = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".mlp.2.weight");
            layer.mlp_1_b = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".mlp.2.bias");

            layer.attn_ln_0_w = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".attn_ln.weight");
            layer.attn_ln_0_b = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".attn_ln.bias");

            layer.attn_q_w = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".attn.query.weight");
            layer.attn_q_b = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".attn.query.bias");

            layer.attn_k_w = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".attn.key.weight");

            layer.attn_v_w = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".attn.value.weight");
            layer.attn_v_b = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".attn.value.bias");

            layer.attn_ln_1_w = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".attn.out.weight");
            layer.attn_ln_1_b = get_tensor(ctx_ggml_, "encoder.blocks." + std::to_string(i) + ".attn.out.bias");
        }
    }
    // audio projector
    model_->proj_1_w = get_tensor(ctx_ggml_, "audio_projector.linear1.weight");
    model_->proj_1_b = get_tensor(ctx_ggml_, "audio_projector.linear1.bias");
    model_->proj_2_w = get_tensor(ctx_ggml_, "audio_projector.linear2.weight");
    model_->proj_2_b = get_tensor(ctx_ggml_, "audio_projector.linear2.bias");

    ggml_free(meta);
}

bool whisper_encoder_model_quantize(const char* fname_inp, const char* fname_out, const int itype) {
    assert(itype < GGML_TYPE_COUNT);
    ggml_type type = static_cast<ggml_type>(itype);

    whisper_encoder_init(fname_inp);

    const auto& ctx_src  = ctx_gguf_;
    const auto& ctx_data = ctx_ggml_;

    auto* ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_src);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", itype);

    auto fout = std::ofstream(fname_out, std::ios::binary);

    const int n_tensors = gguf_get_n_tensors(ctx_src);

    for (int i = 0; i < n_tensors; ++i) {
        const char* name        = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor* cur = ggml_get_tensor(ctx_data, name);
        gguf_add_tensor(ctx_out, cur);
    }

    const size_t meta_size = gguf_get_meta_size(ctx_out);
    for (size_t i = 0; i < meta_size; ++i) {
        fout.put(0);
    }

    // regexes of tensor names to be quantized
    const std::vector<std::string> k_names = {
        ".*weight",
    };
    const std::vector<std::string> k_skip = {
        // ".*weight",
        "encoder.conv*",
        "encoder.positional_embedding*",
        "audio_projector.linear*",
    };

    std::vector<uint8_t> work(512);
    std::vector<float> conv_buf(512);
    size_t total_size_org = 0;
    size_t total_size_new = 0;

    for (int i = 0; i < n_tensors; ++i) {
        const std::string name  = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor* cur = ggml_get_tensor(ctx_data, name.c_str());

        enum ggml_type new_type;
        void* new_data;
        size_t new_size;

        bool quantize = false;
        for (const auto& s : k_names) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }
        for (const auto& s : k_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }

        // quantize only 2D tensors and bigger than block size
        quantize &= (ggml_n_dims(cur) == 2) && cur->ne[0] > ggml_blck_size(type);

        if (quantize) {
            new_type = type;
            if (new_type >= GGML_TYPE_Q2_K && name.find("embd") != std::string::npos) {
                new_type = GGML_TYPE_Q8_0;  // ggml_get_rows needs non K type
                // LOG_ERR("%s: quantizing %s to %s\n", __func__, name.c_str(), ggml_type_name(new_type));
            }
            const size_t n_elms = ggml_nelements(cur);
            float* f32_data;

            switch (cur->type) {
                case GGML_TYPE_F32:
                    f32_data = (float*)cur->data;
                    break;
                case GGML_TYPE_F16:
                    if (conv_buf.size() < n_elms) {
                        conv_buf.resize(n_elms);
                    }
                    for (size_t j = 0; j < n_elms; ++j) {
                        conv_buf[j] = ggml_fp16_to_fp32(((ggml_fp16_t*)cur->data)[j]);
                    }
                    f32_data = (float*)conv_buf.data();
                    break;
                default:
                    LOG_ERR("%s: Please use an input file in f32 or f16\n", __func__);
                    gguf_free(ctx_out);
                    return false;
            }

            if (work.size() < n_elms * 4) {
                work.resize(n_elms * 4);
            }
            new_data = work.data();

            new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, n_elms / cur->ne[0], cur->ne[0], nullptr);
        } else {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }
        const size_t orig_size = ggml_nbytes(cur);
        total_size_org += orig_size;
        total_size_new += new_size;
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        GGML_ASSERT(gguf_get_tensor_size(ctx_out, gguf_find_tensor(ctx_out, name.c_str())) == new_size);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data);
        fout.write((const char*)new_data, new_size);
        size_t pad = GGML_PAD(new_size, gguf_get_alignment(ctx_out)) - new_size;
        for (size_t j = 0; j < pad; ++j) {
            fout.put(0);
        }

        LOG_INF("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), ggml_n_dims(cur), quantize,
                orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
    }

    // go back to beginning of file and write the updated metadata
    fout.seekp(0, std::ios::beg);
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(ctx_out, meta.data());
    fout.write((const char*)meta.data(), meta_size);

    fout.close();

    gguf_free(ctx_out);

    {
        LOG_INF("%s: original  size = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        LOG_INF("%s: quantized size = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);
    }

    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        print_usage(argc, argv);
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const int itype = atoi(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!whisper_encoder_model_quantize(fname_inp.c_str(), fname_out.c_str(), itype)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    return 0;
}
