#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include "log.h"
#include "utils.h"
#include "whisper_encoder.h"

namespace edge {

struct whisper_context_params whisper_context_default_params() {
    struct whisper_context_params result = {
        /*.n_threads            =*/std::min(4, (int32_t)std::thread::hardware_concurrency()),
        /*.use_gpu              =*/true,
        /*.flash_attn           =*/false,
        /*.gpu_device           =*/0,

        /*.dtw_token_timestamps =*/false,
        /*.dtw_aheads_preset    =*/WHISPER_AHEADS_NONE,
        /*.dtw_n_top            =*/-1,
        /*.dtw_mem_size         =*/1024 * 1024 * 128,
    };
    return result;
}

#define SIN_COS_N_COUNT WHISPER_N_FFT
namespace {
struct whisper_global_cache {
    // In FFT, we frequently use sine and cosine operations with the same values.
    // We can use precalculated values to speed up the process.
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];

    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    float hann_window[WHISPER_N_FFT];

    whisper_global_cache() {
        fill_sin_cos_table();
        fill_hann_window(sizeof(hann_window) / sizeof(hann_window[0]), true, hann_window);
    }

    void fill_sin_cos_table() {
        for (int i = 0; i < SIN_COS_N_COUNT; i++) {
            double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
            sin_vals[i]  = sinf(theta);
            cos_vals[i]  = cosf(theta);
        }
    }

    void fill_hann_window(int length, bool periodic, float* output) {
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }
} global_cache;
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
void dft(const float* in, int N, float* out) {
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT);  // t = 2*M_PI*k*n/N
            re += in[n] * global_cache.cos_vals[idx];              // cos(t)
            im -= in[n] * global_cache.sin_vals[idx];              // sin(t)
        }

        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
void fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i] = in[2 * i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2 * i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < half_N; k++) {
        int idx  = k * sin_cos_step;             // t = 2*M_PI*k/N
        float re = global_cache.cos_vals[idx];   // cos(t)
        float im = -global_cache.sin_vals[idx];  // sin(t)

        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + half_N) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

void log_mel_spectrogram_worker_thread(int ith, const float* hann, const std::vector<float>& samples, int n_samples, int frame_size, int frame_step, int n_threads, const whisper_filters& filters, whisper_mel& mel) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft = filters.n_fft;
    int i     = ith;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    assert(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                    fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                    fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                    fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                    fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }
            sum                         = log10(std::max(sum, 1e-10));
            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
bool log_mel_spectrogram(
    const float* samples,
    const int n_samples,
    const int /*sample_rate*/,
    const int frame_size,
    const int frame_step,
    const int n_mel,
    const int n_threads,
    const whisper_filters& filters,
    const bool debug,
    whisper_mel& mel) {
    // Hann window
    assert(frame_size == WHISPER_N_FFT && "Unsupported frame_size");
    const float* hann = global_cache.hann_window;

    // Calculate the length of padding
    int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30;
    int64_t stage_2_pad = frame_size / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

    mel.n_mel = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(
                log_mel_spectrogram_worker_thread, iw + 1, hann, std::cref(samples_padded),
                n_samples + stage_2_pad, frame_size, frame_step, n_threads,
                std::cref(filters), std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, n_threads, filters, mel);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0) / 4.0;
    }

    // Dump log_mel_spectrogram
    if (debug) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
            outFile << mel.data[i] << ", ";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

// measure the memory usage of a graph and prepare the allocr's internal data buffer
bool whisper_sched_graph_init(struct whisper_sched& allocr, std::vector<ggml_backend_t> backends, std::function<struct ggml_cgraph*()>&& get_graph) {
    auto& sched = allocr.sched;
    auto& meta  = allocr.meta;

    sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), WHISPER_MAX_NODES, false);

    meta.resize(ggml_tensor_overhead() * WHISPER_MAX_NODES + ggml_graph_overhead());

    // since there are dependencies between the different graphs,
    // we need to allocate them instead of only reserving to get the correct compute buffer size
    if (!ggml_backend_sched_alloc_graph(sched, get_graph())) {
        // failed to allocate the compute buffer
        LOG_ERR("%s: failed to allocate the compute buffer\n", __func__);
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_graph_compute_helper(
    ggml_backend_sched_t sched,
    struct ggml_cgraph* graph,
    int n_threads) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

        auto* fn_set_n_threads = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn_set_n_threads) {
            fn_set_n_threads(backend, n_threads);
        }
    }

    bool t = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_reset(sched);
    return t;
}

size_t whisper_sched_size(struct whisper_sched& allocr) {
    size_t size = allocr.meta.size();
    for (int i = 0; i < ggml_backend_sched_get_n_backends(allocr.sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(allocr.sched, i);
        size += ggml_backend_sched_get_buffer_size(allocr.sched, backend);
    }
    return size;
}

ggml_backend_t whisper_backend_init_gpu(const whisper_context_params& params) {
    // ggml_log_set(g_state.log_callback, g_state.log_callback_user_data);

    // whisper_load_backends();

    ggml_backend_dev_t dev = nullptr;

    int cnt = 0;
    if (params.use_gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev_cur = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev_cur) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                if (cnt == 0 || cnt == params.gpu_device) {
                    dev = dev_cur;
                }

                if (++cnt > params.gpu_device) {
                    break;
                }
            }
        }
    }

    if (dev == nullptr) {
        LOG_INF("%s: no GPU found\n", __func__);
        return nullptr;
    }

    LOG_INF("%s: using %s backend\n", __func__, ggml_backend_dev_name(dev));
    ggml_backend_t result = ggml_backend_dev_init(dev, nullptr);
    if (!result) {
        LOG_ERR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
    }

    return result;
}

std::vector<ggml_backend_t> whisper_backend_init(const whisper_context_params& params) {
    std::vector<ggml_backend_t> result;

    ggml_backend_t backend_gpu = whisper_backend_init_gpu(params);

    if (backend_gpu) {
        result.push_back(backend_gpu);
    }

    // ACCEL backends
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            LOG_INF("%s: using %s backend\n", __func__, ggml_backend_dev_name(dev));
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (!backend) {
                LOG_ERR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
                continue;
            }
            result.push_back(backend);
        }
    }

    GGML_UNUSED(params);

    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend_cpu == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    result.push_back(backend_cpu);

    return result;
}

WhisperEncoder::WhisperEncoder(std::string model_path, std::string dtw_type, size_t num_audio_ctx) {
    cparams_ = whisper_context_default_params();

    // clang-format off
    auto set_dtw_from_string = [&](std::string dtw_type) {
        if (dtw_type == "tiny")      cparams_.dtw_aheads_preset = WHISPER_AHEADS_TINY;
        if (dtw_type == "tiny.en")   cparams_.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
        if (dtw_type == "base")      cparams_.dtw_aheads_preset = WHISPER_AHEADS_BASE;
        if (dtw_type == "base.en")   cparams_.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
        if (dtw_type == "small")     cparams_.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
        if (dtw_type == "small.en")  cparams_.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
        if (dtw_type == "medium")    cparams_.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
        if (dtw_type == "medium.en") cparams_.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
        if (dtw_type == "large.v1")  cparams_.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
        if (dtw_type == "large.v2")  cparams_.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
        if (dtw_type == "large.v3")  cparams_.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;
        if (dtw_type == "large.v3.turbo")  cparams_.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3_TURBO;
    };
    // clang-format on

    set_dtw_from_string(dtw_type);
    if (cparams_.dtw_aheads_preset == WHISPER_AHEADS_NONE) {
        fprintf(stderr, "error: unknown DTW preset '%s'\n", dtw_type.c_str());
    }

    this->load_model(model_path, 1, num_audio_ctx);
    this->kv_cache_init(model_->hparams.n_audio_state, model_->layers_encoder.size(), model_->hparams.n_audio_ctx);
    this->build_graph();
}

void WhisperEncoder::load_model(const std::string& model_path, const int verbosity, size_t num_audio_ctx) {
    model_                      = new whisper_encoder_model();
    model_->hparams.n_audio_ctx = num_audio_ctx;

    const char* fname         = model_path.c_str();
    struct ggml_context* meta = nullptr;
    struct gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = &meta;

    ctx_gguf_ = gguf_init_from_file(fname, params);
    if (!ctx_gguf_) {
        throw std::runtime_error(format("%s: failed to load Whisper model from %s. Does this file exist?\n", __func__, fname));
    }
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

#ifdef GGML_USE_CUDA
    backend_ = ggml_backend_cuda_init(0);
#endif

    if (backend_ == nullptr) {
        backend_ = ggml_backend_cpu_init();
        LOG_INF("%s: Whisper using CPU backend\n", __func__);
    }

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
        params_buffer_ = ggml_backend_alloc_ctx_tensors(ctx_ggml_, backend_);
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

struct ggml_cgraph* WhisperEncoder::_whisper_build_graph_conv() {
    const auto& hparams = model_->hparams;

    const int n_ctx   = exp_n_audio_ctx_ > 0 ? exp_n_audio_ctx_ : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    GGML_UNUSED(n_state);

    const int n_mels = hparams.n_mels;

    struct ggml_init_params params = {
        /*.mem_size   =*/sched_conv_.meta.size(),
        /*.mem_buffer =*/sched_conv_.meta.data(),
        /*.no_alloc   =*/true,
    };

    struct ggml_context* ctx0 = ggml_init(params);

    ggml_cgraph* gf = ggml_new_graph(ctx0);

    struct ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2 * n_ctx, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    struct ggml_tensor* cur = nullptr;

    // convolution + gelu
    {
        cur = ggml_conv_1d_ph(ctx0, model_->e_conv_1_w, mel, 1, 1);
        cur = ggml_add(ctx0, cur, model_->e_conv_1_b);

        cur = ggml_gelu(ctx0, cur);

        cur = ggml_conv_1d_ph(ctx0, model_->e_conv_2_w, cur, 2, 1);
        cur = ggml_add(ctx0, cur, model_->e_conv_2_b);

        cur = ggml_gelu(ctx0, cur);
    }

    ggml_set_name(cur, "embd_conv_");
    embd_conv_ = cur;

    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

struct ggml_cgraph* WhisperEncoder::_whisper_build_graph_encoder() {
    const auto& hparams = model_->hparams;

    const int n_ctx   = exp_n_audio_ctx_ > 0 ? exp_n_audio_ctx_ : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    const int n_head  = hparams.n_audio_head;
    const int n_layer = hparams.n_audio_layer;

    const int n_state_head = n_state / n_head;

    // flash-attn padding
    // const int n_ctx_pad = GGML_PAD(n_ctx, 256);

    struct ggml_init_params params = {
        /*.mem_size   =*/sched_encode_.meta.size(),
        /*.mem_buffer =*/sched_encode_.meta.data(),
        /*.no_alloc   =*/true,
    };

    struct ggml_context* ctx0 = ggml_init(params);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, WHISPER_MAX_NODES, false);

    struct ggml_tensor* cur = ggml_view_tensor(ctx0, embd_conv_);

    const float KQscale = 1.0f / sqrtf(float(n_state_head));

    // ===================================================================
    // NOTE: experimenting with partial evaluation of the encoder (ignore)
    // static int iter = -1;
    // const int n_iter = 1500/n_ctx;

    // iter = (iter + 1) % n_iter;

    // if (iter == 0) {
    //     memset(model_->memory_cross_k->data, 0, ggml_nbytes(model_->memory_cross_k));
    //     memset(model_->memory_cross_v->data, 0, ggml_nbytes(model_->memory_cross_v));
    // }

    static int iter = 0;

    const size_t e_pe_stride = model_->e_pe->ne[0] * ggml_element_size(model_->e_pe);
    const size_t e_pe_offset = model_->e_pe->ne[0] * ggml_element_size(model_->e_pe) * n_ctx * iter;

    struct ggml_tensor* e_pe = ggml_view_2d(ctx0, model_->e_pe, model_->e_pe->ne[0], n_ctx, e_pe_stride, e_pe_offset);
    cur                      = ggml_add(ctx0, e_pe, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));

    // ===================================================================

    // original:
    // cur = ggml_add(ctx0, model_->e_pe, ggml_transpose(ctx0, cur));

    struct ggml_tensor* inpL = cur;

    for (int il = 0; il < n_layer; ++il) {
        const auto& layer = model_->layers_encoder[il];

        // norm
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);

            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0, cur, layer.attn_ln_0_w),
                           layer.attn_ln_0_b);
        }

        // self-attention
        {
            struct ggml_tensor* Qcur = ggml_mul_mat(ctx0,
                                                    layer.attn_q_w,
                                                    cur);

            Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);

            // Qcur = ggml_scale(ctx0, Qcur, pow(float(n_state_head), -0.25));

            // note: no bias for Key
            struct ggml_tensor* Kcur = ggml_mul_mat(ctx0,
                                                    layer.attn_k_w,
                                                    cur);

            // Kcur = ggml_scale(ctx0, Kcur, pow(float(n_state_head), -0.25));

            struct ggml_tensor* Vcur = ggml_mul_mat(ctx0,
                                                    layer.attn_v_w,
                                                    cur);

            Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);

            // ------

            struct ggml_tensor* Q =
                ggml_permute(ctx0,
                             ggml_reshape_3d(ctx0, Qcur, n_state_head, n_head, n_ctx),
                             0, 2, 1, 3);

            if (cparams_.flash_attn) {
                throw std::runtime_error("flash attention not supported yet");
                // ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, ggml_view_1d(ctx0, kv_pad.k, n_ctx*n_state, 0)));
                // ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, ggml_view_1d(ctx0, kv_pad.v, n_ctx*n_state, 0)));

                // struct ggml_tensor * K =
                //     ggml_view_3d(ctx0, kv_pad.k,
                //             n_state_head, n_ctx_pad, n_head,
                //             ggml_element_size(kv_pad.k)*n_state,
                //             ggml_element_size(kv_pad.k)*n_state_head,
                //             0);

                // struct ggml_tensor * V =
                //     ggml_view_3d(ctx0, kv_pad.v,
                //             n_state_head, n_ctx_pad, n_head,
                //             ggml_element_size(kv_pad.v)*n_state,
                //             ggml_element_size(kv_pad.v)*n_state_head,
                //             0);

                // cur = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, KQscale, 0.0f, 0.0f);

                // cur = ggml_reshape_2d(ctx0, cur, n_state, n_ctx);
            } else {
                struct ggml_tensor* K =
                    ggml_permute(ctx0,
                                 ggml_cast(ctx0,
                                           ggml_reshape_3d(ctx0, Kcur, n_state_head, n_head, n_ctx),
                                           itype_),
                                 0, 2, 1, 3);

                // K * Q
                struct ggml_tensor* KQ = ggml_mul_mat(ctx0, K, Q);

                struct ggml_tensor* KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);

                struct ggml_tensor* V =
                    ggml_cast(ctx0,
                              ggml_permute(ctx0,
                                           ggml_reshape_3d(ctx0,
                                                           Vcur,
                                                           n_state_head, n_head, n_ctx),
                                           1, 2, 0, 3),
                              itype_);

                struct ggml_tensor* KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

                struct ggml_tensor* KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

                cur = ggml_cont_2d(ctx0, KQV_merged, n_state, n_ctx);
            }
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0,
                               layer.attn_ln_1_w,
                               cur);

            cur = ggml_add(ctx0, cur, layer.attn_ln_1_b);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor* inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_add(ctx0,
                               ggml_mul(ctx0, cur, layer.mlp_ln_w),
                               layer.mlp_ln_b);
            }

            // fully connected
            cur = ggml_mul_mat(ctx0,
                               layer.mlp_0_w,
                               cur);

            cur = ggml_add(ctx0, cur, layer.mlp_0_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0,
                               layer.mlp_1_w,
                               cur);

            cur = ggml_add(ctx0, cur, layer.mlp_1_b);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    // norm
    {
        cur = ggml_norm(ctx0, cur, hparams.eps);

        // cur = ln_f_g*cur + ln_f_b
        cur = ggml_add(ctx0,
                       ggml_mul(ctx0, cur, model_->e_ln_w),
                       model_->e_ln_b);
    }

    // audio projector and average pooling
    {
        cur = ggml_mul_mat(ctx0, model_->proj_1_w, cur);
        cur = ggml_add(ctx0, cur, model_->proj_1_b);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_mul_mat(ctx0, model_->proj_2_w, cur);
        cur = ggml_add(ctx0, cur, model_->proj_2_b);

        // average pooling
        // HACK: without ggml_cpy it will cause segmentation fault in ggml_backend_sched_graph_compute
        cur = ggml_cpy(ctx0,
                       ggml_permute(ctx0, cur, 1, 0, 2, 3),
                       ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[1], cur->ne[0]));
        cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 2, 2, 0);
        cur = ggml_cpy(ctx0,
                       ggml_permute(ctx0, cur, 1, 0, 2, 3),
                       ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[1], cur->ne[0]));
    }

    ggml_build_forward_expand(gf, cur);

    embd_enc_ = cur;

    // ggml_graph_print(gf);

    ////////////////////////////////////////////////////////////////////////////

    // printf("%s: used_mem = %f MB, %f MB, %f MB %f MB %f MB\n", __func__,
    //         ggml_used_mem(ctx0)/1e6,
    //         wstate.get_buf_max_mem(0)/1e6,
    //         wstate.get_buf_max_mem(1)/1e6,
    //         wstate.get_buf_max_mem(2)/1e6,
    //         wstate.get_buf_max_mem(3)/1e6);

    ggml_free(ctx0);

    return gf;
}

struct ggml_cgraph* WhisperEncoder::_stream_whisper_build_graph_encoder() {
    const auto& hparams = model_->hparams;

    const int n_tokens = exp_n_audio_ctx_ > 0 ? exp_n_audio_ctx_ : hparams.n_audio_ctx;
    const int n_state  = hparams.n_audio_state;
    const int n_head   = hparams.n_audio_head;
    const int n_layer  = hparams.n_audio_layer;

    const int n_state_head = n_state / n_head;

    // flash-attn padding
    // const int n_ctx_pad = GGML_PAD(n_ctx, 256);

    struct ggml_init_params params = {
        /*.mem_size   =*/sched_encode_.meta.size(),
        /*.mem_buffer =*/sched_encode_.meta.data(),
        /*.no_alloc   =*/true,
    };

    struct ggml_context* ctx0 = ggml_init(params);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, WHISPER_MAX_NODES, false);

    struct ggml_tensor* cur = ggml_view_tensor(ctx0, embd_conv_);

    const float KQscale = 1.0f / sqrtf(float(n_state_head));

    // embed_pos = embed_pos[past_key_values_length : inputs_embeds.shape[1] + past_key_values_length, :]
    const int n_iter = 1500 / n_tokens;
    iter_            = (iter_ + 1) % n_iter;
    // LLAMA_LOG("iter: %d\n", iter_);

    const size_t e_pe_stride = model_->e_pe->ne[0] * ggml_element_size(model_->e_pe);
    const size_t e_pe_offset = model_->e_pe->ne[0] * ggml_element_size(model_->e_pe) * n_tokens * iter_;

    struct ggml_tensor* e_pe = ggml_view_2d(ctx0, model_->e_pe, model_->e_pe->ne[0], n_tokens, e_pe_stride, e_pe_offset);
    cur                      = ggml_add(ctx0, e_pe, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));

    // ===================================================================
    struct ggml_tensor* inpL = cur;

    for (int il = 0; il < n_layer; ++il) {
        const auto& layer = model_->layers_encoder[il];

        // norm
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0, cur, layer.attn_ln_0_w),
                           layer.attn_ln_0_b);
        }

        // self-attention
        {
            struct ggml_tensor* Qcur = ggml_mul_mat(ctx0, layer.attn_q_w, cur);

            Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);

            // note: no bias for Key
            struct ggml_tensor* Kcur = ggml_mul_mat(ctx0, layer.attn_k_w, cur);

            // Kcur = ggml_scale(ctx0, Kcur, pow(float(n_state_head), -0.25));

            struct ggml_tensor* Vcur = ggml_mul_mat(ctx0, layer.attn_v_w, cur);

            Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_state_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_state_head, n_head, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_state_head, n_head, n_tokens);

            const bool v_trans = !cparams_.flash_attn;
            // store key and value to memory
            {
                ggml_tensor* k_cache_view = ggml_view_1d(ctx0, kv_self_.k_l[il], n_tokens * n_state, ggml_row_size(kv_self_.k_l[il]->type, n_state) * (iter_ * n_tokens));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));

                Vcur = ggml_reshape_2d(ctx0, Vcur, n_state, n_tokens);

                ggml_tensor* v_cache_view = nullptr;

                if (!v_trans) {
                    // ggml_tensor* v_cache_view = ggml_view_1d(ctx0, kv_self_.v_l[il], n_ctx * n_state, ggml_element_size(kv_self_.v_l[il]) * n_state * n_ctx * n_kv_shift);
                    throw std::runtime_error("not supported yet");
                } else {
                    v_cache_view = ggml_view_2d(ctx0, kv_self_.v_l[il], n_tokens, n_state,
                                                (kv_self_.size) * ggml_element_size(kv_self_.v_l[il]),
                                                // (n_kv_offset)*ggml_element_size(kv_self_.v_l[il]));
                                                iter_ * n_tokens * ggml_element_size(kv_self_.v_l[il]));
                    Vcur         = ggml_transpose(ctx0, Vcur);
                }
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
            }

            ggml_tensor* Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

            ggml_tensor* K =
                ggml_view_3d(ctx0, kv_self_.k_l[il],
                             // n_state_head, n_tokens, n_head,
                             n_state_head, n_tokens * (iter_ + 1), n_head,
                             ggml_row_size(kv_self_.k_l[il]->type, n_state),
                             ggml_row_size(kv_self_.k_l[il]->type, n_state_head),
                             0);
            // cb(k, "k", il);

            ggml_tensor* V =
                ggml_view_3d(ctx0, kv_self_.v_l[il],
                             //  n_tokens, n_state_head, n_head,
                             n_tokens * (iter_ + 1), n_state_head, n_head,
                             ggml_element_size(kv_self_.v_l[il]) * kv_self_.size,
                             ggml_element_size(kv_self_.v_l[il]) * kv_self_.size * n_state_head,
                             0);

            GGML_ASSERT(v_trans == true);

            if (cparams_.flash_attn) {
                throw std::runtime_error("flash attention not supported yet");
            } else {
                struct ggml_tensor* KQ = ggml_mul_mat(ctx0, K, Q);
                // KQ = ggml_diag_mask_inf_inplace(ctx0, KQ, 1500);
                // KQ = ggml_diag_mask_inf_inplace(ctx0, KQ, n_kv_offset);
                // KQ = ggml_diag_mask_inf_inplace(ctx0, KQ, iter * 50);
                // KQ = ggml_diag_mask_inf_inplace(ctx0, KQ, n_kv_offset + 50);
                struct ggml_tensor* KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);
                struct ggml_tensor* KQV         = ggml_mul_mat(ctx0, V, KQ_soft_max);
                struct ggml_tensor* KQV_merged  = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

                cur = ggml_cont_2d(ctx0, KQV_merged, n_state, n_tokens);
            }
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0,
                               layer.attn_ln_1_w,
                               cur);

            cur = ggml_add(ctx0, cur, layer.attn_ln_1_b);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor* inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_add(ctx0,
                               ggml_mul(ctx0, cur, layer.mlp_ln_w),
                               layer.mlp_ln_b);
            }

            // fully connected
            cur = ggml_mul_mat(ctx0,
                               layer.mlp_0_w,
                               cur);

            cur = ggml_add(ctx0, cur, layer.mlp_0_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0,
                               layer.mlp_1_w,
                               cur);

            cur = ggml_add(ctx0, cur, layer.mlp_1_b);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    // norm
    {
        cur = ggml_norm(ctx0, cur, hparams.eps);

        // cur = ln_f_g*cur + ln_f_b
        cur = ggml_add(ctx0,
                       ggml_mul(ctx0, cur, model_->e_ln_w),
                       model_->e_ln_b);
    }

    // audio projector and average pooling
    {
        cur = ggml_mul_mat(ctx0, model_->proj_1_w, cur);
        cur = ggml_add(ctx0, cur, model_->proj_1_b);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_mul_mat(ctx0, model_->proj_2_w, cur);
        cur = ggml_add(ctx0, cur, model_->proj_2_b);

        // average pooling
        // HACK: without ggml_cpy it will cause segmentation fault in ggml_backend_sched_graph_compute
        cur = ggml_cpy(ctx0,
                       ggml_permute(ctx0, cur, 1, 0, 2, 3),
                       ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[1], cur->ne[0]));
        cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 2, 2, 0);
        cur = ggml_cpy(ctx0,
                       ggml_permute(ctx0, cur, 1, 0, 2, 3),
                       ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[1], cur->ne[0]));
    }

    ggml_build_forward_expand(gf, cur);

    embd_enc_ = cur;

    // ggml_graph_print(gf);

    ////////////////////////////////////////////////////////////////////////////

    // printf("%s: used_mem = %f MB, %f MB, %f MB %f MB %f MB\n", __func__,
    //         ggml_used_mem(ctx0)/1e6,
    //         wstate.get_buf_max_mem(0)/1e6,
    //         wstate.get_buf_max_mem(1)/1e6,
    //         wstate.get_buf_max_mem(2)/1e6,
    //         wstate.get_buf_max_mem(3)/1e6);

    ggml_free(ctx0);

    return gf;
}

void WhisperEncoder::build_graph() {
    backends_ = whisper_backend_init(cparams_);
    // conv allocator
    {
        bool ok = whisper_sched_graph_init(sched_conv_, backends_, [&]() { return _whisper_build_graph_conv(); });
        if (!ok) {
            LOG_ERR("%s: failed to init conv allocator\n", __func__);
            exit(-1);
        }
        LOG_INF("%s: compute buffer (conv)   = %7.2f MB\n", __func__, whisper_sched_size(sched_conv_) / 1e6);
    }

    // encoder allocator
    {
        bool ok = whisper_sched_graph_init(sched_encode_, backends_, [&]() { return _whisper_build_graph_encoder(); });
        if (!ok) {
            LOG_ERR("%s: failed to init encoder allocator\n", __func__);
            exit(-1);
        }
        LOG_INF("%s: compute buffer (encode) = %7.2f MB\n", __func__, whisper_sched_size(sched_encode_) / 1e6);
    }
}

// evaluate the encoder with the given state
bool WhisperEncoder::whisper_encode_internal(
    const int mel_offset,
    const int n_threads) {
    // conv
    {
        auto& sched = sched_conv_.sched;

        ggml_cgraph* gf = this->_whisper_build_graph_conv();

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        struct ggml_tensor* mel = ggml_graph_get_tensor(gf, "mel");

        // set the input
        {
            const auto& mel_inp = mel_;
            const int n_ctx     = exp_n_audio_ctx_ > 0 ? exp_n_audio_ctx_ : model_->hparams.n_audio_ctx;

            assert(mel->type == GGML_TYPE_F32);
            assert(mel_inp.n_mel == model_->hparams.n_mels);

            inp_mel_.resize(ggml_nelements(mel));

            float* dst = inp_mel_.data();
            memset(dst, 0, ggml_nbytes(mel));

            const int i0 = std::min(mel_offset, mel_inp.n_len);
            const int i1 = std::min(mel_offset + 2 * n_ctx, mel_inp.n_len);

            for (int j = 0; j < mel_inp.n_mel; ++j) {
                for (int i = i0; i < i1; ++i) {
                    dst[j * 2 * n_ctx + (i - i0)] = mel_inp.data[j * mel_inp.n_len + i];
                }
            }

            ggml_backend_tensor_set(mel, inp_mel_.data(), 0, ggml_nelements(mel) * sizeof(float));
        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
    }

    // encoder + audio mmprojector
    auto& sched = sched_encode_.sched;

    ggml_cgraph* gf = streaming_ ? this->_stream_whisper_build_graph_encoder()
                                 : this->_whisper_build_graph_encoder();

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        // should never happen as we pre-allocate the memory
        return false;
    }

    if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
        return false;
    }
    kv_self_.n += exp_n_audio_ctx_;
    return true;
}

void WhisperEncoder::forward(std::vector<float>& samples, std::vector<float>& embed_enc_data_out) {
    const size_t n_threads = cparams_.n_threads;
    const size_t n_samples = samples.size();

    // pcm to mel spectrogram
    if (n_samples > 0) {
        // compute log mel spectrogram
        if (!log_mel_spectrogram(samples.data(), n_samples, WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, model_->filters.n_mel, n_threads, model_->filters, false, mel_)) {
            throw std::runtime_error(format("%s: failed to compute mel spectrogram\n", __func__));
        }
    }

    const int seek = 0;
    if (!whisper_encode_internal(seek, n_threads)) {
        LOG_ERR("%s: whisper_encode_internal error", __func__);
        return;
    }

    // sync embd_enc tensor data with embd_enc_data
    ggml_tensor* tensor = embd_enc_;
    embed_enc_data_out.resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, embed_enc_data_out.data(), 0, ggml_nbytes(tensor));

    // for debugging
    // dump_key_tensor_ = kv_self_.k_l[0];
    // // dump tensor
    // std::vector<float> dump_tensor;
    // dump_tensor.resize(ggml_nelements(dump_key_tensor_));
    // ggml_backend_tensor_get(dump_key_tensor_, dump_tensor.data(), 0, ggml_nbytes(dump_key_tensor_));
    // auto dump_bin = [&](const std::vector<float>& dump_tensor, std::string origin = "") -> void {
    //     std::ofstream out("dump_key_" + std::to_string(kv_self_.n) + origin + ".bin", std::ios::binary);
    //     out.write(reinterpret_cast<const char*>(dump_tensor.data()), dump_tensor.size() * sizeof(float));
    //     out.close();
    // };
    // dump_bin(dump_tensor);
    // exit(-1);
}

size_t WhisperEncoder::get_audio_ctx_length() {
    return model_->hparams.n_audio_ctx;
}

void WhisperEncoder::set_exp_n_audio_ctx(int32_t exp_n_ctx) {
    exp_n_audio_ctx_ = exp_n_ctx;
}

void WhisperEncoder::kv_cache_init(int64_t n_state, int64_t n_layer, int n_kv_ctx) {
    kv_self_.n    = 0;
    kv_self_.size = n_kv_ctx;

    struct ggml_init_params params = {
        size_t(2u * n_layer * ggml_tensor_overhead()),
        NULL,
        true,
    };

    ctx_kv_self_ = ggml_init(params);

    if (!ctx_kv_self_) {
        const std::string err_msg = format("%s: failed to allocate memory for the kv cache context\n", __func__);
        throw std::runtime_error(err_msg);
    }

    auto& k_l = kv_self_.k_l;
    auto& v_l = kv_self_.v_l;
    k_l.reserve(n_layer);
    v_l.reserve(n_layer);

    for (int i = 0; i < n_layer; ++i) {
        ggml_tensor* k = ggml_new_tensor_1d(ctx_kv_self_, itype_, n_state * n_kv_ctx);
        ggml_tensor* v = ggml_new_tensor_1d(ctx_kv_self_, itype_, n_state * n_kv_ctx);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        k_l.push_back(k);
        v_l.push_back(v);
    }

    kv_self_.buffer = ggml_backend_alloc_ctx_tensors(ctx_kv_self_, backend_);
    if (!kv_self_.buffer) {
        const std::string err_msg = format("%s: failed to allocate memory for the kv cache\n", __func__);
        throw std::runtime_error(err_msg);
    }

    ggml_backend_buffer_clear(kv_self_.buffer, 0);
    LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(kv_self_.buffer), ggml_backend_buffer_get_size(kv_self_.buffer) / 1024.0 / 1024.0);
}

void WhisperEncoder::kv_cache_free() {
    ggml_backend_buffer_free(kv_self_.buffer);
}

void WhisperEncoder::kv_cache_clear() {
    iter_      = -1;
    kv_self_.n = 0;
    ggml_backend_buffer_clear(kv_self_.buffer, 0);
}

}  // namespace edge
