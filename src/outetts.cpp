#include "outetts.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <regex>
#include <map>
#include "ggml.h"
#include "llama.h"
#include <common/common.h>  // 修正common.h的引入路径
#include <common/sampling.h>  // 修正sampling.h的引入路径

namespace edge {

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

// 数字到文字转换的映射
static const std::map<int, std::string> ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

static void fill_hann_window(int length, bool periodic, float * output) {
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (length - (periodic ? 1 : 0))));
    }
}

static void twiddle(float * real, float * imag, int k, int N) {
    float phi = -2.0 * M_PI * k / N;
    *real = cos(phi);
    *imag = sin(phi);
}

// 实现简单的傅立叶逆变换
static void irfft(int n, const float * inp_cplx, float * out_real) {
    int N = n / 2 + 1;

    std::vector<float> real_input(N);
    std::vector<float> imag_input(N);

    for (int i = 0; i < N; i++) {
        real_input[i] = inp_cplx[2*i];
        imag_input[i] = inp_cplx[2*i + 1];
    }

    std::vector<float> real_output(n);
    std::vector<float> imag_output(n);

    for (int k = 0; k < n; k++) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;

        for (int i = 0; i < N; i++) {
            float real_twiddle, imag_twiddle;
            twiddle(&real_twiddle, &imag_twiddle, k * i, n);

            real_output[k] += real_input[i] * real_twiddle - imag_input[i] * imag_twiddle;
            imag_output[k] += real_input[i] * imag_twiddle + imag_input[i] * real_twiddle;
        }
    }

    for (int i = 0; i < n; i++) {
        out_real[i] = real_output[i] / n;
    }
}

// fold操作，将重叠加窗的STFT转换为波形
static void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width    = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t) data.size()) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}

// 将嵌入转换为音频
static std::vector<float> embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread) {
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop)/2;
    const int n_out = (n_codes - 1)*n_hop + n_win;

    std::vector<float> hann(n_fft);

    fill_hann_window(hann.size(), true, hann.data());

    int n_spec = n_embd*n_codes;

    std::vector<float> E (n_spec);
    std::vector<float> S (n_spec);
    std::vector<float> ST(n_spec);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k*n_codes + l] = embd[l*n_embd + k];
        }
    }

    for (int k = 0; k < n_embd/2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k           )*n_codes + l];
            float phi = E[(k + n_embd/2)*n_codes + l];

            mag = exp(mag);

            if (mag > 1e2) {
                mag = 1e2;
            }
            S[2*(k*n_codes + l) + 0] = mag*cosf(phi);
            S[2*(k*n_codes + l) + 1] = mag*sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd/2; ++k) {
            ST[l*n_embd + 2*k + 0] = S[2*(k*n_codes + l) + 0];
            ST[l*n_embd + 2*k + 1] = S[2*(k*n_codes + l) + 1];
        }
    }

    std::vector<float> res  (n_codes*n_fft);
    std::vector<float> hann2(n_codes*n_fft);

    std::vector<std::thread> workers(n_thread);
    for (int i = 0; i < n_thread; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += n_thread) {
                irfft(n_fft, ST.data() + l*n_embd, res.data() + l*n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res  [l*n_fft + j] *= hann[j];
                    hann2[l*n_fft + j]  = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < n_thread; ++i) {
        workers[i].join();
    }

    std::vector<float> audio;
    std::vector<float> env;

    fold(res,   n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env); // TODO: can be done once

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
}

// 将小于1000的数字转换为英文
static std::string convert_less_than_thousand(int num) {
    std::string result;

    if (num >= 100) {
        result += ones.at(num / 100) + " hundred ";
        num %= 100;
    }

    if (num >= 20) {
        result += tens.at(num / 10);
        if (num % 10 > 0) {
            result += "-" + ones.at(num % 10);
        }
    } else if (num > 0) {
        result += ones.at(num);
    }

    return result;
}

// 将数字转换为英文单词
std::string Outetts::replace_numbers_with_words(const std::string & input_text) {
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string result;
    auto it = std::sregex_iterator(input_text.begin(), input_text.end(), number_pattern);
    auto end = std::sregex_iterator();

    size_t last_pos = 0;
    for (std::sregex_iterator i = it; i != end; ++i) {
        const std::smatch& match = *i;
        result.append(input_text, last_pos, match.position() - last_pos);
        
        try {
            std::string number_str = match.str();
            size_t decimal_pos = number_str.find('.');
            std::string integer_part = number_str.substr(0, decimal_pos);

            int int_number = std::stoi(integer_part);
            std::string word_result;

            if (int_number == 0) {
                word_result = "zero";
            } else {
                if (int_number >= 1000000000) {
                    int billions = int_number / 1000000000;
                    word_result += convert_less_than_thousand(billions) + " billion ";
                    int_number %= 1000000000;
                }

                if (int_number >= 1000000) {
                    int millions = int_number / 1000000;
                    word_result += convert_less_than_thousand(millions) + " million ";
                    int_number %= 1000000;
                }

                if (int_number >= 1000) {
                    int thousands = int_number / 1000;
                    word_result += convert_less_than_thousand(thousands) + " thousand ";
                    int_number %= 1000;
                }

                if (int_number > 0) {
                    word_result += convert_less_than_thousand(int_number);
                }
            }

            // 处理小数部分
            if (decimal_pos != std::string::npos) {
                word_result += " point";
                std::string decimal_part = number_str.substr(decimal_pos + 1);
                for (char digit : decimal_part) {
                    word_result += " " + ones.at(digit - '0');
                }
            }
            
            result.append(word_result);
        } catch (const std::exception& e) {
            // 转换失败则保留原数字
            result.append(match.str());
        }
        
        last_pos = match.position() + match.length();
    }
    result.append(input_text, last_pos);

    return result;
}

// 处理文本，准备TTS输入
std::string Outetts::process_text(const std::string & text) {
    // 移除<|tts_eos|>和</s>
    std::string processed_text = std::regex_replace(text, std::regex(R"(<\|tts_eos\|>|<\/s>)"), "");
    // 将数字转换为单词
    processed_text = replace_numbers_with_words(processed_text);

    // 转换为小写
    std::transform(processed_text.begin(), processed_text.end(),
                   processed_text.begin(), ::tolower);

    // 替换特殊字符为空格
    std::regex special_chars(R"([-_/,\.\\])");
    processed_text = std::regex_replace(processed_text, special_chars, " ");

    // 移除非字母字符
    std::regex non_alpha(R"([^a-z\s])");
    processed_text = std::regex_replace(processed_text, non_alpha, "");

    // 将多个空格替换为单个空格
    std::regex multiple_spaces(R"(\s+)");
    processed_text = std::regex_replace(processed_text, multiple_spaces, " ");

    // 去除首尾空格
    processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");

    // 替换空格为分隔符
    std::string separator = (tts_version_ == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), separator);

    return processed_text;
}

static void prompt_add(llama_tokens & prompt, llama_token token) {
    prompt.push_back(token);
}

static void prompt_add(llama_tokens & prompt, const llama_tokens & tokens) {
    prompt.insert(prompt.end(), tokens.begin(), tokens.end());
}

static void prompt_add(llama_tokens & prompt, const llama_vocab * vocab, const std::string & txt, bool add_special, bool parse_special) {
    auto tmp = common_tokenize(vocab, txt, add_special, parse_special);
    prompt_add(prompt, tmp);
}

static void prompt_init(llama_tokens & prompt, const llama_vocab * vocab) {
    prompt.clear();

    prompt_add(prompt, vocab, "<|im_start|>\n", true, true);
}

void Outetts::init_params_ttc(const std::string& ttc_model_path) {
    params_ttc_.out_file = "output.wav";
    params_ttc_.prompt = "";

    params_ttc_.n_predict = 4096;
    params_ttc_.n_batch   = 8192;
    params_ttc_.n_ctx     = 8192;
    params_ttc_.sampling.top_k = 4;
    params_ttc_.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K, };        
    params_ttc_.model.path = ttc_model_path;
    params_ttc_.n_gpu_layers = 30;
}

void Outetts::init_params_cts(const std::string& cts_model_path) {
    params_cts_.out_file = "output.wav";
    params_cts_.prompt = "";

    params_cts_.n_predict = 4096;
    params_cts_.n_batch   = 8192;
    params_cts_.n_ctx     = 8192;
    params_cts_.model.path = cts_model_path;
    params_cts_.embedding = true;
    params_cts_.n_gpu_layers = 30;
}


void Outetts::init_params(const std::string& ttc_model_path, const std::string& cts_model_path) {
    init_params_ttc(ttc_model_path);
    init_params_cts(cts_model_path);
    n_parallel_ = params_ttc_.n_parallel;
    n_predict_ = params_ttc_.n_predict;
}

// Outetts类实现
Outetts::Outetts(const std::string& ttc_model_path, const std::string& cts_model_path) {
    // 初始化参数
    init_params(ttc_model_path, cts_model_path);

    // common init
    // common_init();

    // // init LLM

    llama_backend_init();
    llama_numa_init(params_ttc_.numa);
    llama_init_ttc =  common_init_from_params(params_ttc_);
    model_ttc_ = llama_init_ttc.model.get();
    ctx_ttc_   = llama_init_ttc.context.get();
    // vocab_ttc_.init_tokenizer();


    llama_init_cts = common_init_from_params(params_cts_);
    model_cts_ = llama_init_cts.model.get();
    ctx_cts_   = llama_init_cts.context.get();
    if (model_cts_ == nullptr || ctx_cts_ == nullptr || model_ttc_ == nullptr || ctx_ttc_ == nullptr) {
        LOG_ERR("Failed to initialize TTC or CTS model\n");
        return;
    }

    // test vocab 

    {
        std::vector<llama_token> prompt_inp;

        const llama_vocab * vocab_ttc_test = llama_model_get_vocab(model_ttc_);

        prompt_init(prompt_inp, vocab_ttc_test);
        prompt_add(prompt_inp, vocab_ttc_test, "hello", false, true);
    }



    // // init sampler
    // std::vector<common_sampler *> smpl(n_parallel_);
    // for (int i = 0; i < n_parallel_; ++i) {
    //     params_ttc.sampling.no_perf = (i != 0);
    //     params_ttc.sampling.seed = params_ttc.sampling.seed + 1;

    //     smpl[i] = common_sampler_init(model_ttc_, params_ttc.sampling);
    // }

    // LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl[0]));
    // LOG_INF("sampler params: \n%s\n", params_ttc.sampling.print().c_str());
    // LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl[0]).c_str());

    // LOG_INF("%s: loading done\n", __func__);


}

// 文本到语音转换的完整流程
bool Outetts::text_to_speech(const std::string& text, std::vector<float>& audio_out) {

   // init LLM

    // llama_backend_init();
    // llama_numa_init(params_ttc_.numa);
    // common_init_result llama_init_ttc = common_init_from_params(params_ttc_);
    // model_ttc_ = llama_init_ttc.model.get();
    // ctx_ttc_   = llama_init_ttc.context.get();
    // // vocab_ttc_.init_tokenizer();


    // common_init_result llama_init_cts = common_init_from_params(params_cts_);
    // model_cts_ = llama_init_cts.model.get();
    // ctx_cts_   = llama_init_cts.context.get();
    // if (model_cts_ == nullptr || ctx_cts_ == nullptr || model_ttc_ == nullptr || ctx_ttc_ == nullptr) {
    //     LOG_ERR("Failed to initialize TTC or CTS model\n");
    //     return false;
    // }

    // // test vocab 

    // {
    //     std::vector<llama_token> prompt_inp;

    //     const llama_vocab * vocab_ttc_test = llama_model_get_vocab(model_ttc_);

    //     prompt_init(prompt_inp, vocab_ttc_test);
    //     prompt_add(prompt_inp, vocab_ttc_test, "hello", false, true);
    // }


    // 1. 处理输入文本
    // the default speaker profile is from: https://github.com/edwko/OuteTTS/blob/main/outetts/version/v1/default_speakers/en_male_1.json
    std::string processed_text = process_text(text);
    // std::string processed_text = process_text(text);
    // LOG_INF("Processed text: %s\n", processed_text.c_str());
    
    // 2. 获取llama模型的词汇表
    std::vector<llama_token> prompt_inp;

    const llama_vocab * vocab_ttc_ = llama_model_get_vocab(model_ttc_);

    prompt_init(prompt_inp, vocab_ttc_);
    prompt_add(prompt_inp, vocab_ttc_, audio_text_, false, true);
    std::string prompt_clean = process_text(text);
    prompt_add(prompt_inp, vocab_ttc_, prompt_clean, false, true);
    prompt_add(prompt_inp, vocab_ttc_, "<|text_end|>\n", false, true);
    const std::string voice_data = audio_data_;

    auto tmp = common_tokenize(vocab_ttc_, voice_data, false, true);

    std::ostringstream tokens_oss;
    for (size_t i = 0; i < tmp.size(); ++i) {
        tokens_oss << tmp[i] << ", ";
    }
    LOG_INF("\n\n%s: llama tokens: %s\n\n", __func__, tokens_oss.str().c_str());
    prompt_add(prompt_inp, tmp);



    
    // 4. 创建批处理
    llama_batch batch = llama_batch_init(prompt_inp.size(), 0, n_parallel_);

    std::vector<llama_seq_id> seq_ids(n_parallel_, 0);
    for (int32_t i = 0; i < n_parallel_; ++i) {
        seq_ids[i] = i;
    }

    // evaluate the initial prompt
    for (size_t i = 0; i < prompt_inp.size(); ++i) {
        common_batch_add(batch, prompt_inp[i], i, seq_ids, false);
    }
    GGML_ASSERT(batch.n_tokens == (int) prompt_inp.size());

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;
    
    // 5. 模型推理
    if (llama_decode(ctx_ttc_, batch) != 0) {
        LOG_ERR("Failed to decode prompt\n");
        llama_batch_free(batch);
        return false;
    }
    llama_synchronize(ctx_ttc_);

    // 6. 生成音频编码序列
    std::vector<llama_token> codes;

    std::vector<common_sampler *> smpl(n_parallel_);
    for (int i = 0; i < n_parallel_; ++i) {
        params_ttc_.sampling.no_perf = (i != 0);
        params_ttc_.sampling.seed = params_ttc_.sampling.seed + 1;

        smpl[i] = common_sampler_init(model_ttc_, params_ttc_.sampling);
    }
    const auto t_dec_start = ggml_time_us();
    // main loop

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    std::vector<int32_t> i_batch(n_parallel_, batch.n_tokens - 1);

    int n_past   = batch.n_tokens;
    int n_decode = 0;


    while (n_decode <= n_predict_) {
        // prepare the next batch
        common_batch_clear(batch);

        // sample the next token for each parallel sequence / stream
        for (int32_t i = 0; i < n_parallel_; ++i) {
            if (i_batch[i] < 0) {
                // the stream has already finished
                continue;
            }

            llama_token new_token_id = common_sampler_sample(smpl[i], ctx_ttc_, i_batch[i]);


            //this is the token id that always precedes a new word

            common_sampler_accept(smpl[i], new_token_id, true);

            codes.push_back(new_token_id);
            // is it an end of generation? -> mark the stream as finished
            if (llama_vocab_is_eog(vocab_ttc_, new_token_id) || n_decode == n_predict_) {
                std::string reason;
                if (llama_vocab_is_eog(vocab_ttc_, new_token_id)) {
                    reason = "eos";
                } else {
                    reason = "n_predict";
                }

                i_batch[i] = -1;

                LOG("\n");
                if (n_parallel_ > 1) {
                    LOG_CNT("\n");
                    LOG_INF("%s: stream %d finished at n_past = %d, reason = '%s'\n", __func__, i, n_past, reason.c_str());
                }

                continue;
            }

            {
                LOG_CNT("%d", i);
            }

            i_batch[i] = batch.n_tokens;

            // push this new token for next evaluation
            common_batch_add(batch, new_token_id, n_past, { i }, true);
        }

        // all streams are finished
        if (batch.n_tokens == 0) {
            break;
        }

        n_decode += 1;
        n_past += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx_ttc_, batch)) {
            LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    llama_batch_free(batch);

    LOG("\n");
    LOG_INF("%s: time for decoder:       %.3f ms\n", __func__, (ggml_time_us() - t_dec_start) / 1000.0f);


    {
        const std::string inp_txt = common_detokenize(ctx_ttc_, codes, true);

        LOG("\n");
        LOG_INF("codes: '%s'\n", inp_txt.c_str());
        LOG_INF("%s: codes size: %d\n", __func__, (int) codes.size());
    }    
    
    // 7. 过滤出音频编码tokens (范围大约在151672到155772之间)
    codes.erase(std::remove_if(codes.begin(), codes.end(), 
                [](llama_token t) { return t < 151672 || t > 155772; }), 
                codes.end());
    
    // 转换编码
    for (auto& token : codes) {
        token -= 151672; // 调整到正确的范围
    }
    
    // 8. 使用vocoder生成音频
    if (codes.empty()) {
        LOG_ERR("No audio codes generated\n");
        return false;
    }
    
    // 创建用于vocoder的batch
    llama_batch cts_batch = llama_batch_init(codes.size(), 0, 1);
    for (size_t i = 0; i < codes.size(); ++i) {
        common_batch_add(cts_batch, codes[i], i, seq_ids, true);
    }
    
    // 执行vocoder解码
    if (llama_decode(ctx_cts_, cts_batch) != 0) {
        LOG_ERR("Failed to decode with vocoder\n");
        llama_batch_free(cts_batch);
        return false;
    }
    llama_synchronize(ctx_cts_);
    
    // 9. 获取嵌入并转换为音频
    const float* embd = llama_get_embeddings(ctx_cts_);
    int n_embd = llama_model_n_embd(model_cts_);
    
    audio_out = embd_to_audio(embd, codes.size(), n_embd, 8);
    
    // 10. 清理
    llama_batch_free(cts_batch);
    
    // 11. 对音频进行后处理（例如静音开头部分）
    int silence_samples = save_sample_rate_ / 4; // 0.25秒静音
    if (audio_out.size() > static_cast<size_t>(silence_samples)) {
        for (int i = 0; i < silence_samples; ++i) {
            audio_out[i] = 0.0f;
        }
    }
    
    return true;
}

// 保存WAV文件
bool Outetts::save_wav(const std::string& fname, const std::vector<float>& data) {
    std::ofstream file(fname, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file: %s\n", fname.c_str());
        return false;
    }

    struct wav_header header;
    header.sample_rate = save_sample_rate_;
    header.bits_per_sample = 16;
    header.num_channels = 1;
    header.byte_rate = save_sample_rate_ * header.bits_per_sample / 8 * header.num_channels;
    header.block_align = header.bits_per_sample / 8 * header.num_channels;
    header.data_size = data.size() * header.bits_per_sample / 8;
    header.chunk_size = 36 + header.data_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const auto & sample : data) {
        int16_t pcm_sample = static_cast<int16_t>(std::max(-32768.0f, std::min(sample * 32767.0f, 32767.0f)));
        file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
    }

    return file.good();
}

} // namespace edge 