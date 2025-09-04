#ifndef INCLUDE_SIGLIP_H_
#define INCLUDE_SIGLIP_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "utils.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

//
// key constants
//

#define KEY_FTYPE "general.file_type"
#define KEY_NAME "general.name"
#define KEY_DESCRIPTION "general.description"
#define KEY_HAS_TEXT_ENC "clip.has_text_encoder"
#define KEY_HAS_VIS_ENC "clip.has_vision_encoder"
#define KEY_HAS_LLAVA_PROJ "clip.has_llava_projector"
#define KEY_HAS_MINICPMV_PROJ "clip.has_minicpmv_projector"
#define KEY_MINICPMV_VERSION "clip.minicpmv_version"
#define KEY_HAS_QWEN2VL_MERGER "clip.has_qwen2vl_merger"
#define KEY_USE_GELU "clip.use_gelu"
#define KEY_USE_SILU "clip.use_silu"
#define KEY_N_EMBD "clip.%s.embedding_length"
#define KEY_N_FF "clip.%s.feed_forward_length"
#define KEY_N_BLOCK "clip.%s.block_count"
#define KEY_N_HEAD "clip.%s.attention.head_count"
#define KEY_LAYER_NORM_EPS "clip.%s.attention.layer_norm_epsilon"
#define KEY_PROJ_DIM "clip.%s.projection_dim"
#define KEY_TOKENS "tokenizer.ggml.tokens"
#define KEY_N_POSITIONS "clip.text.context_length"
#define KEY_IMAGE_SIZE "clip.vision.image_size"
#define KEY_PATCH_SIZE "clip.vision.patch_size"
#define KEY_IMAGE_MEAN "clip.vision.image_mean"
#define KEY_IMAGE_STD "clip.vision.image_std"
#define KEY_PROJ_TYPE "clip.projector_type"

#define KEY_MM_PATCH_MERGE_TYPE "clip.vision.mm_patch_merge_type"
#define KEY_IMAGE_GRID_PINPOINTS "clip.vision.image_grid_pinpoints"
#define KEY_IMAGE_CROP_RESOLUTION "clip.vision.image_crop_resolution"

//
// tensor name constants
//

#define TN_TOKEN_EMBD "%s.token_embd.weight"
#define TN_POS_EMBD "%s.position_embd.weight"
#define TN_CLASS_EMBD "v.class_embd"
#define TN_PATCH_EMBD "v.patch_embd.weight"  // not rename tensor with ".0" postfix for backwrad compat
#define TN_PATCH_EMBD_1 "v.patch_embd.weight.1"
#define TN_PATCH_BIAS "v.patch_embd.bias"
#define TN_ATTN_K "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT "%s.blk.%d.attn_out.%s"
#define TN_FFN_DOWN "%s.blk.%d.ffn_down.%s"
#define TN_FFN_UP "%s.blk.%d.ffn_up.%s"
#define TN_LN_1 "%s.blk.%d.ln1.%s"
#define TN_LN_2 "%s.blk.%d.ln2.%s"
#define TN_LN_PRE "%s.pre_ln.%s"
#define TN_LN_POST "%s.post_ln.%s"
#define TN_TEXT_PROJ "text_projection.weight"
#define TN_VIS_PROJ "visual_projection.weight"
#define TN_LLAVA_PROJ "mm.%d.%s"
#define TN_MVLM_PROJ_MLP "mm.model.mlp.%d.%s"
#define TN_MVLM_PROJ_BLOCK "mm.model.mb_block.%d.block.%d.%s"
#define TN_MVLM_PROJ_PEG "mm.model.peg.%d.%s"
#define TN_IMAGE_NEWLINE "model.image_newline"

#define TN_MINICPMV_POS_EMBD_K "resampler.pos_embed_k"
#define TN_MINICPMV_QUERY "resampler.query"
#define TN_MINICPMV_PROJ "resampler.proj.weight"
#define TN_MINICPMV_KV_PROJ "resampler.kv.weight"
#define TN_MINICPMV_ATTN "resampler.attn.%s.%s"
#define TN_MINICPMV_LN "resampler.ln_%s.%s"

// struct clip_hparams {
//     int32_t image_size;
//     int32_t patch_size;
//     int32_t hidden_size;
//     int32_t n_intermediate;
//     int32_t projection_dim;
//     int32_t n_head;
//     int32_t n_layer;

//     float eps;

//     char mm_patch_merge_type[32] = "flat"; // spatial_unpad or flat (default)

// };
namespace edge {

struct clip_layer {
    // attention
    struct ggml_tensor* k_w;
    struct ggml_tensor* k_b;
    struct ggml_tensor* q_w;
    struct ggml_tensor* q_b;
    struct ggml_tensor* v_w;
    struct ggml_tensor* v_b;

    struct ggml_tensor* o_w;
    struct ggml_tensor* o_b;

    // layernorm 1
    struct ggml_tensor* ln_1_w;
    struct ggml_tensor* ln_1_b;

    // ff
    struct ggml_tensor* ff_i_w;
    struct ggml_tensor* ff_i_b;

    struct ggml_tensor* ff_o_w;
    struct ggml_tensor* ff_o_b;

    // layernorm 2
    struct ggml_tensor* ln_2_w;
    struct ggml_tensor* ln_2_b;
};

struct clip_vision_model {
    // struct clip_hparams hparams;

    // embeddings
    // struct ggml_tensor * class_embedding;
    struct ggml_tensor* patch_embeddings_0;
    struct ggml_tensor* patch_embeddings_1;  // second Conv2D kernel when we decouple Conv3D along temproal dimension (Qwen2VL)
    struct ggml_tensor* patch_bias;
    struct ggml_tensor* position_embeddings;

    // struct ggml_tensor * pre_ln_w;
    // struct ggml_tensor * pre_ln_b;

    std::vector<clip_layer> layers;

    struct ggml_tensor* post_ln_w;
    struct ggml_tensor* post_ln_b;

    struct ggml_tensor* projection;

    // LLaVA projection
    struct ggml_tensor* mm_0_w = NULL;
    struct ggml_tensor* mm_0_b = NULL;
    struct ggml_tensor* mm_2_w = NULL;
    struct ggml_tensor* mm_2_b = NULL;

    struct ggml_tensor* image_newline = NULL;

    // Yi type models with mlp+normalization projection
    struct ggml_tensor* mm_1_w = NULL;  // Yi type models have 0, 1, 3, 4
    struct ggml_tensor* mm_1_b = NULL;
    struct ggml_tensor* mm_3_w = NULL;
    struct ggml_tensor* mm_3_b = NULL;
    struct ggml_tensor* mm_4_w = NULL;
    struct ggml_tensor* mm_4_b = NULL;

    // MobileVLM projection
    struct ggml_tensor* mm_model_mlp_1_w;
    struct ggml_tensor* mm_model_mlp_1_b;
    struct ggml_tensor* mm_model_mlp_3_w;
    struct ggml_tensor* mm_model_mlp_3_b;
    struct ggml_tensor* mm_model_block_1_block_0_0_w;
    struct ggml_tensor* mm_model_block_1_block_0_1_w;
    struct ggml_tensor* mm_model_block_1_block_0_1_b;
    struct ggml_tensor* mm_model_block_1_block_1_fc1_w;
    struct ggml_tensor* mm_model_block_1_block_1_fc1_b;
    struct ggml_tensor* mm_model_block_1_block_1_fc2_w;
    struct ggml_tensor* mm_model_block_1_block_1_fc2_b;
    struct ggml_tensor* mm_model_block_1_block_2_0_w;
    struct ggml_tensor* mm_model_block_1_block_2_1_w;
    struct ggml_tensor* mm_model_block_1_block_2_1_b;
    struct ggml_tensor* mm_model_block_2_block_0_0_w;
    struct ggml_tensor* mm_model_block_2_block_0_1_w;
    struct ggml_tensor* mm_model_block_2_block_0_1_b;
    struct ggml_tensor* mm_model_block_2_block_1_fc1_w;
    struct ggml_tensor* mm_model_block_2_block_1_fc1_b;
    struct ggml_tensor* mm_model_block_2_block_1_fc2_w;
    struct ggml_tensor* mm_model_block_2_block_1_fc2_b;
    struct ggml_tensor* mm_model_block_2_block_2_0_w;
    struct ggml_tensor* mm_model_block_2_block_2_1_w;
    struct ggml_tensor* mm_model_block_2_block_2_1_b;

    // MobileVLM_V2 projection
    struct ggml_tensor* mm_model_mlp_0_w;
    struct ggml_tensor* mm_model_mlp_0_b;
    struct ggml_tensor* mm_model_mlp_2_w;
    struct ggml_tensor* mm_model_mlp_2_b;
    struct ggml_tensor* mm_model_peg_0_w;
    struct ggml_tensor* mm_model_peg_0_b;

    // MINICPMV projection
    struct ggml_tensor* mm_model_pos_embed_k;
    struct ggml_tensor* mm_model_query;
    struct ggml_tensor* mm_model_proj;
    struct ggml_tensor* mm_model_kv_proj;
    struct ggml_tensor* mm_model_attn_q_w;
    struct ggml_tensor* mm_model_attn_q_b;
    struct ggml_tensor* mm_model_attn_k_w;
    struct ggml_tensor* mm_model_attn_k_b;
    struct ggml_tensor* mm_model_attn_v_w;
    struct ggml_tensor* mm_model_attn_v_b;
    struct ggml_tensor* mm_model_attn_o_w;
    struct ggml_tensor* mm_model_attn_o_b;
    struct ggml_tensor* mm_model_ln_q_w;
    struct ggml_tensor* mm_model_ln_q_b;
    struct ggml_tensor* mm_model_ln_kv_w;
    struct ggml_tensor* mm_model_ln_kv_b;
    struct ggml_tensor* mm_model_ln_post_w;
    struct ggml_tensor* mm_model_ln_post_b;
};

class Siglip {
public:
    Siglip() = delete;
    Siglip(const std::string& model_path, bool use_flash_attn = false, const int verbosity = 1);

    void forward(const image_buf<float>& imgs, const int img_w, const int img_h, std::vector<float>& out);
    size_t _embd_nbytes();

    std::vector<float>& get_image_mean() { return image_mean_; }
    std::vector<float>& get_image_std() { return image_std_; }

    ~Siglip() {
        GGML_FREE(ctx_ggml_);
        GGUF_FREE(ctx_gguf_);
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
    }

protected:
    void load_model(const std::string& model_path, const int verbosity);
    ggml_cgraph* build_graph(const int image_width, const int image_height);

public:
    int embed_dim_  = 3584;
    int n_patches_  = 64;  // for minicpm-omni
    int patch_size_ = 14;
    bool use_fa_    = false;

private:
    clip_vision_model model_ = clip_vision_model{};

    bool use_gelu_ = false;
    std::vector<float> image_mean_;
    std::vector<float> image_std_;
    int32_t ftype_ = 1;

    int image_size_  = 448;
    int hidden_size_ = 1152;
    int n_head_      = 16;
    int n_layer_     = 27;
    // float eps_ = 0.000001;
    float eps_ = 1e-6;

    struct gguf_context* ctx_gguf_ = nullptr;
    struct ggml_context* ctx_ggml_ = nullptr;

    std::vector<uint8_t> buf_compute_meta_;

    // memory buffers to evaluate the model
    ggml_backend_buffer_t params_buffer_ = nullptr;

    ggml_backend_t backend_       = nullptr;
    ggml_gallocr_t compute_alloc_ = nullptr;
};

int ensure_divide(int length, int patch_size);
bool bicubic_resize(const image_buf<uint8_t>& img, image_buf<uint8_t>& dst, int target_width, int target_height);
std::pair<int, int> uhd_best_grid(const int max_slice_nums, const int multiple, const float log_ratio);
std::pair<int, int> uhd_find_best_resize(std::pair<int, int> original_size, int scale_resolution, int patch_size, bool allow_upscale);
std::pair<int, int> uhd_get_refine_size(std::pair<int, int> original_size, std::pair<int, int> grid, int scale_resolution, int patch_size, bool allow_upscale);
// std::vector<std::vector<image_buf<uint8_t>>> uhd_slice_image(const image_buf<uint8_t>& img, int max_slice_nums, const int scale_resolution, const int patch_size);
std::vector<std::vector<image_buf<uint8_t>>> uhd_slice_image(const image_buf<uint8_t>& img, int max_slice_nums = 9, const int scale_resolution = 448, const int patch_size = 14);
image_buf<float> reshape_by_patch(image_buf<float>& image, int patch_size);

std::vector<std::vector<float>> get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int> image_size);
std::vector<std::vector<std::vector<float>>> get_2d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<std::vector<std::vector<float>>>& grid);
std::vector<std::vector<std::vector<float>>> get_1d_sincos_pos_embed_from_grid_new(int embed_dim, const std::vector<std::vector<float>>& pos);

}  // namespace edge
#endif  // INCLUDE_SIGLIP_H_
