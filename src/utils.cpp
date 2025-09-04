#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "ggml/src/ggml-impl.h"
#include "gguf.h"
#include "llama-impl.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "log.h"
#include "utils.h"

namespace edge {

bool eval_tokens(struct llama_context* ctx_llama, std::vector<llama_token> tokens, int n_batch, int* n_past) {
    int N = (int)tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval))) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

bool eval_id(struct llama_context* ctx_llama, int id, int* n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

bool eval_string(struct llama_context* ctx_llama, const char* str, int n_batch, int* n_past, bool add_bos) {
    std::string str2                  = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_llama, str2, add_bos, true);
    return eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
}

//
// utilities to get data from a gguf file
//

int get_key_idx(const gguf_context* ctx, const char* key) {
    int i = gguf_find_key(ctx, key);
    if (i == -1) {
        LOG_ERR("key %s not found in file\n", key);
        throw std::runtime_error(format("Missing required key: %s", key));
    }

    return i;
}

uint32_t get_u32(const gguf_context* ctx, const std::string& key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_u32(ctx, i);
}

float get_f32(const gguf_context* ctx, const std::string& key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_f32(ctx, i);
}

struct ggml_tensor* get_tensor(struct ggml_context* ctx, const std::string& name) {
    struct ggml_tensor* cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        throw std::runtime_error(format("%s: unable to find tensor %s\n", __func__, name.c_str()));
    }

    return cur;
}

std::string get_ftype(int ftype) {
    return ggml_type_name(static_cast<ggml_type>(ftype));
}

std::string gguf_data_to_str(enum gguf_type type, const void* data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:
            return std::to_string(((const uint8_t*)data)[i]);
        case GGUF_TYPE_INT8:
            return std::to_string(((const int8_t*)data)[i]);
        case GGUF_TYPE_UINT16:
            return std::to_string(((const uint16_t*)data)[i]);
        case GGUF_TYPE_INT16:
            return std::to_string(((const int16_t*)data)[i]);
        case GGUF_TYPE_UINT32:
            return std::to_string(((const uint32_t*)data)[i]);
        case GGUF_TYPE_INT32:
            return std::to_string(((const int32_t*)data)[i]);
        case GGUF_TYPE_UINT64:
            return std::to_string(((const uint64_t*)data)[i]);
        case GGUF_TYPE_INT64:
            return std::to_string(((const int64_t*)data)[i]);
        case GGUF_TYPE_FLOAT32:
            return std::to_string(((const float*)data)[i]);
        case GGUF_TYPE_FLOAT64:
            return std::to_string(((const double*)data)[i]);
        case GGUF_TYPE_BOOL:
            return ((const bool*)data)[i] ? "true" : "false";
        default:
            return format("unknown type %d", type);
    }
}

void print_tensor_info(const ggml_tensor* tensor, const char* prefix = "") {
    size_t tensor_size = ggml_nbytes(tensor);
    LOG_INF("%s: n_dims = %d, name = %s, tensor_size=%zu, shape:[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], type = %s\n",
            prefix, ggml_n_dims(tensor), tensor->name, tensor_size,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], ggml_type_name(tensor->type));
}

bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long* sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        LOG_ERR("%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char*)malloc(fileSize);  // Allocate memory to hold the file data
    if (buffer == NULL) {
        LOG_ERR("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno      = 0;
    size_t ret = fread(buffer, 1, fileSize, file);  // Read the file into the buffer
    if (ferror(file)) {
        LOG_ERR("read error: %s", strerror(errno));
        free(buffer);
        fclose(file);
        return false;
    }
    if (ret != (size_t)fileSize) {
        LOG_ERR("unexpectedly reached end of file");
        free(buffer);
        fclose(file);
        return false;
    }
    fclose(file);  // Close the file

    *bytesOut = buffer;
    *sizeOut  = fileSize;
    return true;
}

bool load_image_from_bytes(const unsigned char* bytes, size_t bytes_length, image_buf<uint8_t>& image) {
    int nx, ny, nc;
    auto* data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    image.nx = nx;
    image.ny = ny;
    image.buf.resize(3 * nx * ny);
    std::memcpy(image.buf.data(), data, image.buf.size());

    stbi_image_free(data);
    return true;
}

// Normalize image to float32 - careful with pytorch .to(model.device, dtype=torch.float16) - this sometimes reduces precision (32>16>32), sometimes not
void normalize_image_u8_to_f32(const image_buf<uint8_t>& src, image_buf<float>& dst, const float* mean, const float* std) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(src.buf.size());

    for (size_t i = 0; i < src.buf.size(); ++i) {
        int c      = i % 3;  // rgb
        dst.buf[i] = (static_cast<float>(src.buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

}  // namespace edge
