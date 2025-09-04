#ifndef INCLUDE_UTILS_HPP_
#define INCLUDE_UTILS_HPP_

#include <common/common.h>
#include <cstdint>
#include <string>
#include <vector>
// #include "log.h"
#include "llama.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

#define GGUF_FREE(ctx)        \
    do {                      \
        if (ctx != nullptr) { \
            gguf_free(ctx);   \
            ctx = nullptr;    \
        }                     \
    } while (0);

#define GGML_FREE(ctx)        \
    do {                      \
        if (ctx != nullptr) { \
            ggml_free(ctx);   \
            ctx = nullptr;    \
        }                     \
    } while (0);

namespace edge {

// Memory layout: RGBRGBRGB...
template <class T>
struct image_buf {
    int nx;  // width
    int ny;  // height
    std::vector<T> buf;
};

struct image_u8_batch {
    image_buf<uint8_t>* data;
    size_t size;
};

struct image_f32_batch {
    std::vector<image_buf<float>> data;
    size_t size;
};

struct image_embed {
    std::vector<float> embed;
    int n_image_pos;
};

bool eval_tokens(struct llama_context* ctx_llama, std::vector<llama_token> tokens, int n_batch, int* n_past);
bool eval_id(struct llama_context* ctx_llama, int id, int* n_past);
bool eval_string(struct llama_context* ctx_llama, const char* str, int n_batch, int* n_past, bool add_bos);

//
// utilities to get data from a gguf file
//

int get_key_idx(const gguf_context* ctx, const char* key);
uint32_t get_u32(const gguf_context* ctx, const std::string& key);
float get_f32(const gguf_context* ctx, const std::string& key);
struct ggml_tensor* get_tensor(struct ggml_context* ctx, const std::string& name);
std::string get_ftype(int ftype);
std::string gguf_data_to_str(enum gguf_type type, const void* data, int i);
void print_tensor_info(const ggml_tensor* tensor, const char* prefix);

bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long* sizeOut);
bool load_image_from_bytes(const unsigned char* bytes, size_t bytes_length, image_buf<uint8_t>& image);

void normalize_image_u8_to_f32(const image_buf<uint8_t>& src, image_buf<float>& dst, const float* mean, const float* std);
}  // namespace edge

#endif  // INCLUDE_UTILS_HPP_
