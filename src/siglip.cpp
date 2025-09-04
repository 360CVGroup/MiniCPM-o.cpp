/*
 * file based on:
 * https://github.com/tc-mb/llama.cpp/blob/27f5e8a10a3c8916f18b786e8d24db4c672b9891/examples/llava/clip.cpp
 */
#include <thread>
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include "log.h"
#include "siglip.h"
#include "utils.h"

namespace edge {

inline static int clip(int x, int lower, int upper) {
    return std::max(lower, std::min(x, upper));
}

bool bicubic_resize(const image_buf<uint8_t>& img, image_buf<uint8_t>& dst, int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    const float tx = static_cast<float>(nx) / target_width;
    const float ty = static_cast<float>(ny) / target_height;

#ifdef USE_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < target_height; ++i) {
        for (int j = 0; j < target_width; ++j) {
            const float x_f = tx * j;
            const float y_f = ty * i;
            const int x     = static_cast<int>(x_f);
            const int y     = static_cast<int>(y_f);
            const float dx  = x_f - x;
            const float dy  = y_f - y;

            // Precompute clipped coordinates
            const int x_coords[4] = {
                clip(x - 1, 0, nx - 1),
                clip(x, 0, nx - 1),
                clip(x + 1, 0, nx - 1),
                clip(x + 2, 0, nx - 1)};

            alignas(32) int y_coords[4];
#ifdef USE_OPENMP
#pragma omp simd
#endif
            for (int jj = 0; jj < 4; ++jj) {
                y_coords[jj] = clip(y - 1 + jj, 0, ny - 1);
            }

            // Precompute cubic terms
            const float dx_sq = dx * dx;
            const float dx_cu = dx_sq * dx;
            const float dy_sq = dy * dy;
            const float dy_cu = dy_sq * dy;

#ifdef USE_OPENMP
#pragma omp simd
#endif
            for (int k = 0; k < 3; ++k) {
                alignas(32) float C[4];
                for (int jj = 0; jj < 4; ++jj) {
                    // Horizontal interpolation
                    const uint8_t* row = &img.buf[y_coords[jj] * nx * 3];

                    const float p0 = row[x_coords[0] * 3 + k];
                    const float p1 = row[x_coords[1] * 3 + k];
                    const float p2 = row[x_coords[2] * 3 + k];
                    const float p3 = row[x_coords[3] * 3 + k];

                    const float d0 = p0 - p1;
                    const float d2 = p2 - p1;
                    const float d3 = p3 - p1;

                    const float a1 = -1.0f / 3 * d0 + d2 - 1.0f / 6 * d3;
                    const float a2 = 0.5f * d0 + 0.5f * d2;
                    const float a3 = -1.0f / 6 * d0 - 0.5f * d2 + 1.0f / 6 * d3;

                    C[jj] = p1 + a1 * dx + a2 * dx_sq + a3 * dx_cu;
                }

                // Vertical interpolation
                const float d0 = C[0] - C[1];
                const float d2 = C[2] - C[1];
                const float d3 = C[3] - C[1];

                const float a1 = -1.0f / 3 * d0 + d2 - 1.0f / 6 * d3;
                const float a2 = 0.5f * d0 + 0.5f * d2;
                const float a3 = -1.0f / 6 * d0 - 0.5f * d2 + 1.0f / 6 * d3;

                const float Cc = C[1] + a1 * dy + a2 * dy_sq + a3 * dy_cu;
                dst.buf[(i * target_width + j) * 3 + k] =
                    static_cast<uint8_t>(clip(std::round(Cc), 0.0f, 255.0f));
            }
        }
    }

    return true;
}

int ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

std::pair<int, int> uhd_find_best_resize(std::pair<int, int> original_size, int scale_resolution, int patch_size, bool allow_upscale = false) {
    int width  = original_size.first;
    int height = original_size.second;
    if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
        float r = static_cast<float>(width) / height;
        height  = static_cast<int>(scale_resolution / std::sqrt(r));
        width   = static_cast<int>(height * r);
    }
    int best_width  = ensure_divide(width, patch_size);
    int best_height = ensure_divide(height, patch_size);
    return std::make_pair(best_width, best_height);
}

std::pair<int, int> uhd_get_refine_size(std::pair<int, int> original_size, std::pair<int, int> grid, int scale_resolution, int patch_size, bool allow_upscale = false) {
    int width, height;
    std::tie(width, height) = original_size;
    int grid_x, grid_y;
    std::tie(grid_x, grid_y) = grid;

    int refine_width  = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);

    int grid_width  = refine_width / grid_x;
    int grid_height = refine_height / grid_y;

    // auto best_grid_size = find_best_resize(std::make_tuple(grid_width, grid_height), scale_resolution, patch_size, allow_upscale); (old line)
    auto best_grid_size = uhd_find_best_resize(std::make_pair(grid_width, grid_height), scale_resolution, patch_size, allow_upscale);  // (new line) => fixes conversion for make_tuple to make_pair
    int best_grid_width, best_grid_height;
    std::tie(best_grid_width, best_grid_height) = best_grid_size;

    //  std::pair<int, int> refine_size = std::make_tuple(best_grid_width * grid_x, best_grid_height * grid_y); (old line)
    std::pair<int, int> refine_size = std::make_pair(best_grid_width * grid_x, best_grid_height * grid_y);  // (new line)
    return refine_size;
}

std::pair<int, int> uhd_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
    std::vector<int> candidate_split_grids_nums;
    for (int i : {multiple - 1, multiple, multiple + 1}) {
        if (i == 1 || i > max_slice_nums) {
            continue;
        }
        candidate_split_grids_nums.push_back(i);
    }

    std::vector<std::pair<int, int>> candidate_grids;
    for (int split_grids_nums : candidate_split_grids_nums) {
        int m = 1;
        while (m <= split_grids_nums) {
            if (split_grids_nums % m == 0) {
                candidate_grids.emplace_back(m, split_grids_nums / m);
            }
            ++m;
        }
    }

    std::pair<int, int> best_grid{1, 1};
    float min_error = std::numeric_limits<float>::infinity();
    for (const auto& grid : candidate_grids) {
        float error = std::abs(log_ratio - std::log(1.0 * grid.first / grid.second));
        if (error < min_error) {
            best_grid = grid;
            min_error = error;
        }
    }
    return best_grid;
}

// inspired from LLaVA-UHD:
//    -> https://arxiv.org/pdf/2403.11703
//    -> https://github.com/thunlp/LLaVA-UHD
//    -> https://github.com/thunlp/LLaVA-UHD/blob/302301bc2175f7e717fb8548516188e89f649753/llava_uhd/train/llava-uhd/slice_logic.py#L118
std::vector<std::vector<image_buf<uint8_t>>> uhd_slice_image(const image_buf<uint8_t>& img, int max_slice_nums, const int scale_resolution, const int patch_size) {
    const std::pair<int, int> original_size = {img.nx, img.ny};
    const int original_width                = img.nx;
    const int original_height               = img.ny;
    const float log_ratio                   = log(1.0 * original_width / original_height);
    const float ratio                       = 1.0 * original_width * original_height / (scale_resolution * scale_resolution);
    const int multiple                      = fmin(ceil(ratio), max_slice_nums);

    std::vector<std::vector<image_buf<uint8_t>>> images;
    images.push_back(std::vector<image_buf<uint8_t>>());

    if (multiple <= 1) {
        auto best_size = uhd_find_best_resize(original_size, scale_resolution, patch_size, true);
        image_buf<uint8_t> source_image;
        bicubic_resize(img, source_image, best_size.first, best_size.second);
        images[images.size() - 1].push_back(source_image);
    } else if (multiple > 1) {
        auto best_size = uhd_find_best_resize(original_size, scale_resolution, patch_size);
        image_buf<uint8_t> source_image;
        bicubic_resize(img, source_image, best_size.first, best_size.second);
        LOG_INF("%s: image_size: %d %d; source_image size: %d %d\n", __func__, img.nx, img.ny, best_size.first, best_size.second);
        images[images.size() - 1].push_back(source_image);

        std::pair<int, int> best_grid = uhd_best_grid(max_slice_nums, multiple, log_ratio);
        LOG_INF("%s: image_size: %d %d; best_grid: %d %d\n", __func__, img.nx, img.ny, best_grid.first, best_grid.second);

        auto refine_size = uhd_get_refine_size(original_size, best_grid, scale_resolution, patch_size, true);
        image_buf<uint8_t> refine_image;
        bicubic_resize(img, refine_image, refine_size.first, refine_size.second);

        LOG_INF("%s: refine_image_size: %d %d; refine_size: %d %d\n", __func__, refine_image.nx, refine_image.ny, refine_size.first, refine_size.second);

        // split_to_patches
        int width  = refine_image.nx;
        int height = refine_image.ny;
        int grid_x = int(width / best_grid.first);
        int grid_y = int(height / best_grid.second);
        for (int patches_i = 0, ic = 0; patches_i < height && ic < best_grid.second; patches_i += grid_y, ic += 1) {
            images.push_back(std::vector<image_buf<uint8_t>>());
            for (int patches_j = 0, jc = 0; patches_j < width && jc < best_grid.first; patches_j += grid_x, jc += 1) {
                image_buf<uint8_t> patch;
                patch.nx = grid_x;
                patch.ny = grid_y;
                patch.buf.resize(3 * patch.nx * patch.ny);
                for (int y = patches_i; y < patches_i + grid_y; ++y) {
                    for (int x = patches_j; x < patches_j + grid_x; ++x) {
                        const int i      = 3 * (y * refine_image.nx + x);
                        const int j      = 3 * ((y - patches_i) * patch.nx + (x - patches_j));
                        patch.buf[j]     = refine_image.buf[i];
                        patch.buf[j + 1] = refine_image.buf[i + 1];
                        patch.buf[j + 2] = refine_image.buf[i + 2];
                    }
                }
                images[images.size() - 1].push_back(patch);
            }
        }
    }
    return images;
}

image_buf<float> reshape_by_patch(image_buf<float>& image, int patch_size) {
    int width       = image.nx;
    int height      = image.ny;
    int num_patches = (height / patch_size) * (width / patch_size);
    image_buf<float> patch;
    patch.nx = patch_size * num_patches;
    patch.ny = patch_size;
    patch.buf.resize(3 * patch.nx * patch.ny);

    int patch_index = 0;

    for (int i = 0; i < height; i += patch_size) {
        for (int j = 0; j < width; j += patch_size) {
            for (int pi = 0; pi < patch_size; ++pi) {
                for (int pj = 0; pj < patch_size; ++pj) {
                    int input_index             = ((i + pi) * width + (j + pj)) * 3;
                    int output_index            = (pi * patch_size * num_patches + patch_index * patch_size + pj) * 3;
                    patch.buf[output_index]     = image.buf[input_index];
                    patch.buf[output_index + 1] = image.buf[input_index + 1];
                    patch.buf[output_index + 2] = image.buf[input_index + 2];
                }
            }
            patch_index++;
        }
    }
    return patch;
}

std::vector<std::vector<std::vector<float>>> get_1d_sincos_pos_embed_from_grid_new(int embed_dim, const std::vector<std::vector<float>>& pos) {
    assert(embed_dim % 2 == 0);
    const int H        = pos.size();
    const int W        = pos.empty() ? 0 : pos[0].size();
    const int half_dim = embed_dim / 2;

    std::vector<float> omega(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        omega[i] = 1.0f / std::pow(10000.0f, static_cast<float>(i) / static_cast<float>(half_dim));
    }

    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const float pos_val = pos[h][w];
            for (int d = 0; d < half_dim; ++d) {
                const float out_value   = pos_val * omega[d];
                emb[h][w][d]            = std::sin(out_value);
                emb[h][w][d + half_dim] = std::cos(out_value);
            }
        }
    }

    return emb;
}

std::vector<std::vector<std::vector<float>>> get_2d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<std::vector<std::vector<float>>>& grid) {
    assert(embed_dim % 2 == 0);
    const int half_dim = embed_dim / 2;

    auto emb_h = get_1d_sincos_pos_embed_from_grid_new(half_dim, grid[0]);
    auto emb_w = get_1d_sincos_pos_embed_from_grid_new(half_dim, grid[1]);

    const int H = emb_h.size();
    const int W = H > 0 ? emb_h[0].size() : 0;

    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            // cache references to avoid repeated indexing
            auto& emb_h_vec = emb_h[h][w];
            auto& emb_w_vec = emb_w[h][w];
            auto& emb_vec   = emb[h][w];

            // merge the two vectors
            for (int d = 0; d < half_dim; ++d) {
                emb_vec[d]            = emb_h_vec[d];
                emb_vec[d + half_dim] = emb_w_vec[d];
            }
        }
    }
    return emb;
}

std::vector<std::vector<float>> get_2d_sincos_pos_embed(int embed_dim,
                                                        const std::pair<int, int> image_size) {
    const int grid_h_size = image_size.first;
    const int grid_w_size = image_size.second;

    std::vector<float> grid_h(grid_h_size);
    for (int i = 0; i < grid_h_size; ++i) {
        grid_h[i] = static_cast<float>(i);
    }
    std::vector<float> grid_w(grid_w_size);
    for (int i = 0; i < grid_w_size; ++i) {
        grid_w[i] = static_cast<float>(i);
    }

    std::vector<std::vector<float>> grid(grid_h_size, std::vector<float>(grid_w_size));
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid[h][w] = grid_w[w];
        }
    }
    std::vector<std::vector<std::vector<float>>> grid_2d(2, std::vector<std::vector<float>>(grid_h_size, std::vector<float>(grid_w_size)));
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid_2d[0][h][w] = grid_h[h];
            grid_2d[1][h][w] = grid_w[w];
        }
    }

    auto pos_embed_3d = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_2d);

    const int H = image_size.first;
    const int W = image_size.second;
    std::vector<std::vector<float>> pos_embed_2d(H * W, std::vector<float>(embed_dim));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const int flat_idx     = h * W + w;
            pos_embed_2d[flat_idx] = pos_embed_3d[h][w];
        }
    }

    return pos_embed_2d;
}

Siglip::Siglip(const std::string& model_path, bool use_flash_attn, const int verbosity) {
    this->use_fa_ = use_flash_attn;
    this->load_model(model_path, verbosity);
    // measure mem requirement and allocate
    buf_compute_meta_.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
    if (compute_alloc_ == nullptr) {
        compute_alloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    }
}

void Siglip::load_model(const std::string& model_path, const int verbosity = 1) {
    const char* fname         = model_path.c_str();
    struct ggml_context* meta = nullptr;
    struct gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = &meta;

    // struct gguf_context* ctx_gguf_ = gguf_init_from_file(fname, params);
    ctx_gguf_ = gguf_init_from_file(fname, params);
    if (!ctx_gguf_) {
        throw std::runtime_error(format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname));
    }
    if (verbosity >= 1) {
        const int n_tensors           = gguf_get_n_tensors(ctx_gguf_);
        const int n_kv                = gguf_get_n_kv(ctx_gguf_);
        const int ftype               = get_u32(ctx_gguf_, KEY_FTYPE);
        const std::string ftype_str   = get_ftype(ftype);
        const int idx_desc            = get_key_idx(ctx_gguf_, KEY_DESCRIPTION);
        const std::string description = gguf_get_val_str(ctx_gguf_, idx_desc);
        const int idx_name            = gguf_find_key(ctx_gguf_, KEY_NAME);
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

#ifdef GGML_USE_METAL
    backend_ = ggml_backend_metal_init();
#endif

    if (backend_ == nullptr) {
        backend_ = ggml_backend_cpu_init();
        LOG_INF("%s: Siglip using CPU backend\n", __func__);
    }

    int idx   = get_key_idx(ctx_gguf_, KEY_USE_GELU);
    use_gelu_ = gguf_get_val_bool(ctx_gguf_, idx);

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

    // load vision model
    // auto& model_ = model_;

    // auto& hparams = model_.hparams;
    // hparams.hidden_size    = get_u32(ctx_gguf_, format(KEY_N_EMBD, "vision"));
    // hparams.n_head         = get_u32(ctx_gguf_, format(KEY_N_HEAD, "vision"));
    // hparams.n_intermediate = get_u32(ctx_gguf_, format(KEY_N_FF, "vision"));
    // hparams.n_layer        = get_u32(ctx_gguf_, format(KEY_N_BLOCK, "vision"));
    // hparams.image_size     = get_u32(ctx_gguf_, KEY_IMAGE_SIZE);
    // hparams.patch_size     = get_u32(ctx_gguf_, KEY_PATCH_SIZE);
    // hparams.projection_dim = get_u32(ctx_gguf_, format(KEY_PROJ_DIM, "vision"));
    // hparams.eps            = get_f32(ctx_gguf_, format(KEY_LAYER_NORM_EPS, "vision"));

    // auto& hparams          = model_.hparams;
    hidden_size_ = get_u32(ctx_gguf_, format(KEY_N_EMBD, "vision"));
    n_head_      = get_u32(ctx_gguf_, format(KEY_N_HEAD, "vision"));
    // hparams.n_intermediate = get_u32(ctx_gguf_, format(KEY_N_FF, "vision"));
    n_layer_    = get_u32(ctx_gguf_, format(KEY_N_BLOCK, "vision"));
    image_size_ = get_u32(ctx_gguf_, KEY_IMAGE_SIZE);
    patch_size_ = get_u32(ctx_gguf_, KEY_PATCH_SIZE);
    // hparams.projection_dim = get_u32(ctx_gguf_, format(KEY_PROJ_DIM, "vision"));
    eps_ = get_f32(ctx_gguf_, format(KEY_LAYER_NORM_EPS, "vision"));

    int idx_mean = get_key_idx(ctx_gguf_, KEY_IMAGE_MEAN);
    int idx_std  = get_key_idx(ctx_gguf_, KEY_IMAGE_STD);

    const float* mean_data = (const float*)gguf_get_arr_data(ctx_gguf_, idx_mean);
    const float* std_data  = (const float*)gguf_get_arr_data(ctx_gguf_, idx_std);

    image_mean_.resize(3);
    image_std_.resize(3);
    for (int i = 0; i < 3; ++i) {
        image_mean_[i] = mean_data[i];
        image_std_[i]  = std_data[i];
    }

    model_.post_ln_w = get_tensor(ctx_ggml_, format(TN_LN_POST, "v", "weight"));
    model_.post_ln_b = get_tensor(ctx_ggml_, format(TN_LN_POST, "v", "bias"));

    model_.patch_bias          = get_tensor(ctx_ggml_, TN_PATCH_BIAS);
    model_.patch_embeddings_0  = get_tensor(ctx_ggml_, TN_PATCH_EMBD);
    model_.position_embeddings = get_tensor(ctx_ggml_, format(TN_POS_EMBD, "v"));

    // resampler
    model_.mm_model_pos_embed_k = get_tensor(ctx_ggml_, TN_MINICPMV_POS_EMBD_K);
    model_.mm_model_query       = get_tensor(ctx_ggml_, TN_MINICPMV_QUERY);
    model_.mm_model_proj        = get_tensor(ctx_ggml_, TN_MINICPMV_PROJ);
    model_.mm_model_kv_proj     = get_tensor(ctx_ggml_, TN_MINICPMV_KV_PROJ);
    model_.mm_model_attn_q_w    = get_tensor(ctx_ggml_, format(TN_MINICPMV_ATTN, "q", "weight"));
    model_.mm_model_attn_k_w    = get_tensor(ctx_ggml_, format(TN_MINICPMV_ATTN, "k", "weight"));
    model_.mm_model_attn_v_w    = get_tensor(ctx_ggml_, format(TN_MINICPMV_ATTN, "v", "weight"));
    model_.mm_model_attn_q_b    = get_tensor(ctx_ggml_, format(TN_MINICPMV_ATTN, "q", "bias"));
    model_.mm_model_attn_k_b    = get_tensor(ctx_ggml_, format(TN_MINICPMV_ATTN, "k", "bias"));
    model_.mm_model_attn_v_b    = get_tensor(ctx_ggml_, format(TN_MINICPMV_ATTN, "v", "bias"));
    model_.mm_model_attn_o_w    = get_tensor(ctx_ggml_, format(TN_MINICPMV_ATTN, "out", "weight"));
    model_.mm_model_attn_o_b    = get_tensor(ctx_ggml_, format(TN_MINICPMV_ATTN, "out", "bias"));
    model_.mm_model_ln_q_w      = get_tensor(ctx_ggml_, format(TN_MINICPMV_LN, "q", "weight"));
    model_.mm_model_ln_q_b      = get_tensor(ctx_ggml_, format(TN_MINICPMV_LN, "q", "bias"));
    model_.mm_model_ln_kv_w     = get_tensor(ctx_ggml_, format(TN_MINICPMV_LN, "kv", "weight"));
    model_.mm_model_ln_kv_b     = get_tensor(ctx_ggml_, format(TN_MINICPMV_LN, "kv", "bias"));
    model_.mm_model_ln_post_w   = get_tensor(ctx_ggml_, format(TN_MINICPMV_LN, "post", "weight"));
    model_.mm_model_ln_post_b   = get_tensor(ctx_ggml_, format(TN_MINICPMV_LN, "post", "bias"));

    model_.layers.resize(n_layer_);

    for (int il = 0; il < n_layer_; ++il) {
        auto& layer  = model_.layers[il];
        layer.k_w    = get_tensor(ctx_ggml_, format(TN_ATTN_K, "v", il, "weight"));
        layer.q_w    = get_tensor(ctx_ggml_, format(TN_ATTN_Q, "v", il, "weight"));
        layer.v_w    = get_tensor(ctx_ggml_, format(TN_ATTN_V, "v", il, "weight"));
        layer.o_w    = get_tensor(ctx_ggml_, format(TN_ATTN_OUTPUT, "v", il, "weight"));
        layer.ln_1_w = get_tensor(ctx_ggml_, format(TN_LN_1, "v", il, "weight"));
        layer.ln_2_w = get_tensor(ctx_ggml_, format(TN_LN_2, "v", il, "weight"));
        layer.ff_i_w = get_tensor(ctx_ggml_, format(TN_FFN_DOWN, "v", il, "weight"));
        layer.ff_o_w = get_tensor(ctx_ggml_, format(TN_FFN_UP, "v", il, "weight"));
        layer.k_b    = get_tensor(ctx_ggml_, format(TN_ATTN_K, "v", il, "bias"));
        layer.q_b    = get_tensor(ctx_ggml_, format(TN_ATTN_Q, "v", il, "bias"));
        layer.v_b    = get_tensor(ctx_ggml_, format(TN_ATTN_V, "v", il, "bias"));
        layer.o_b    = get_tensor(ctx_ggml_, format(TN_ATTN_OUTPUT, "v", il, "bias"));
        layer.ln_1_b = get_tensor(ctx_ggml_, format(TN_LN_1, "v", il, "bias"));
        layer.ln_2_b = get_tensor(ctx_ggml_, format(TN_LN_2, "v", il, "bias"));
        layer.ff_i_b = get_tensor(ctx_ggml_, format(TN_FFN_DOWN, "v", il, "bias"));
        layer.ff_o_b = get_tensor(ctx_ggml_, format(TN_FFN_UP, "v", il, "bias"));
    }

    ggml_free(meta);
}

static ggml_tensor* build_flash_attn(
    struct ggml_context* ctx0,
    ggml_cgraph* gf,
    ggml_tensor* q,
    ggml_tensor* k,
    ggml_tensor* v,
    ggml_tensor* kq_mask,
    bool v_trans,
    float kq_scale) {
    const auto n_tokens = q->ne[1];
    const auto n_head   = q->ne[2];
    const auto n_kv     = k->ne[1];

    ggml_tensor* cur;

    // TODO: replace hardcoded padding with ggml-provided padding
    if (n_kv % 256 == 0) {
        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * n_head, n_tokens);
    } else {
        ggml_tensor* kq = ggml_mul_mat(ctx0, k, q);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
        }

        ggml_tensor* kqv = ggml_mul_mat(ctx0, v, kq);
        cur              = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur              = ggml_cont_2d(ctx0, cur, cur->ne[0] * n_head, n_tokens);
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_cgraph* Siglip::build_graph(const int image_width, const int image_height) {
    const int batch_size       = 1;
    const int num_patches      = (image_width / patch_size_) * (image_height / patch_size_);
    const int num_positions    = num_patches;
    const int num_position_ids = num_patches;
    const int d_head           = hidden_size_ / n_head_;

    struct ggml_init_params params;
    params.mem_size   = buf_compute_meta_.size();
    params.mem_buffer = buf_compute_meta_.data();
    params.no_alloc   = true;

    struct ggml_context* _ctx = ggml_init(params);

    ggml_cgraph* gf = ggml_new_graph(_ctx);

    struct ggml_tensor* inp_raw = ggml_new_tensor_4d(_ctx, GGML_TYPE_F32, image_width, image_height, 3, batch_size);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    struct ggml_tensor* inp = ggml_conv_2d(_ctx, model_.patch_embeddings_0, inp_raw, patch_size_, patch_size_, 0, 0, 1, 1);

    inp = ggml_reshape_3d(_ctx, inp, num_patches, hidden_size_, batch_size);
    inp = ggml_cont(_ctx, ggml_permute(_ctx, inp, 1, 0, 2, 3));
    inp = ggml_add(_ctx, inp, model_.patch_bias);

    struct ggml_tensor* embeddings = inp;
    struct ggml_tensor* pos_embed  = nullptr;

    struct ggml_tensor* positions = ggml_new_tensor_1d(_ctx, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    embeddings = ggml_add(_ctx, embeddings, ggml_get_rows(_ctx, model_.position_embeddings, positions));

    int pos_w = image_width / patch_size_;
    int pos_h = image_height / patch_size_;

    pos_embed = ggml_new_tensor_3d(_ctx, GGML_TYPE_F32, embed_dim_, pos_w * pos_h, 1);
    ggml_set_name(pos_embed, "pos_embed");
    ggml_set_input(pos_embed);

    for (int il = 0; il < n_layer_; il++) {
        struct ggml_tensor* cur = embeddings;  // embeddings = residual, cur = hidden_states

        // layernorm1
        {
            cur = ggml_norm(_ctx, cur, eps_);

            cur = ggml_add_inplace(_ctx, ggml_mul(_ctx, cur, model_.layers[il].ln_1_w),
                                   model_.layers[il].ln_1_b);
        }

        // self-attention
        {
            struct ggml_tensor* Q =
                ggml_add_inplace(_ctx, ggml_mul_mat(_ctx, model_.layers[il].q_w, cur), model_.layers[il].q_b);

            Q = ggml_reshape_4d(_ctx, Q, d_head, n_head_, num_positions, batch_size);

            Q = ggml_cont(_ctx, ggml_permute(_ctx, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(_ctx, Q, d_head, num_positions, n_head_ * batch_size);

            struct ggml_tensor* K =
                ggml_add_inplace(_ctx, ggml_mul_mat(_ctx, model_.layers[il].k_w, cur), model_.layers[il].k_b);

            K = ggml_reshape_4d(_ctx, K, d_head, n_head_, num_positions, batch_size);

            K = ggml_cont(_ctx, ggml_permute(_ctx, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(_ctx, K, d_head, num_positions, n_head_ * batch_size);

            struct ggml_tensor* V =
                ggml_add_inplace(_ctx, ggml_mul_mat(_ctx, model_.layers[il].v_w, cur), model_.layers[il].v_b);

            V = ggml_reshape_4d(_ctx, V, d_head, n_head_, num_positions, batch_size);
            V = ggml_cont(_ctx, ggml_permute(_ctx, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(_ctx, V, num_positions, d_head, n_head_ * batch_size);

            if (use_fa_) {
                // hardcoded padding to fit flash attention
                Q = ggml_pad(_ctx, Q, 80 - Q->ne[0], GGML_PAD(Q->ne[1], 256) - Q->ne[1], 0, 0);
                K = ggml_pad(_ctx, K, 80 - K->ne[0], GGML_PAD(K->ne[1], 256) - K->ne[1], 0, 0);
                V = ggml_pad(_ctx, V, GGML_PAD(V->ne[0], 256) - V->ne[0], 80 - V->ne[1], 0, 0);

                V = ggml_transpose(_ctx, V);

                // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
                if (K->type == GGML_TYPE_F32) {
                    K = ggml_cast(_ctx, K, GGML_TYPE_F16);
                }

                if (V->type == GGML_TYPE_F32) {
                    V = ggml_cast(_ctx, V, GGML_TYPE_F16);
                }

                cur = ggml_flash_attn_ext(_ctx, Q, K, V, nullptr, 1.0f / sqrt((float)d_head), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

                // attention output
                cur = ggml_view_4d(_ctx, cur, d_head, num_patches, n_head_, 1,
                                   d_head * ggml_type_size(GGML_TYPE_F32),
                                   d_head * num_patches * ggml_type_size(GGML_TYPE_F32),
                                   d_head * num_patches * n_head_ * ggml_type_size(GGML_TYPE_F32),
                                   0);

                cur = ggml_reshape_2d(_ctx, cur, hidden_size_, num_patches);
            } else {
                struct ggml_tensor* KQ  = ggml_mul_mat(_ctx, K, Q);
                KQ                      = ggml_soft_max_ext(_ctx, KQ, nullptr, 1.0f / sqrt((float)d_head), 0.0f);
                struct ggml_tensor* KQV = ggml_mul_mat(_ctx, V, KQ);
                KQV                     = ggml_reshape_4d(_ctx, KQV, d_head, num_positions, n_head_, batch_size);
                KQV                     = ggml_permute(_ctx, KQV, 0, 2, 1, 3);

                cur = ggml_cont_3d(_ctx, KQV, hidden_size_, num_positions, batch_size);
            }
        }
        cur = ggml_add_inplace(_ctx, ggml_mul_mat(_ctx, model_.layers[il].o_w, cur), model_.layers[il].o_b);

        // re-add the layer input, e.g., residual
        cur = ggml_add_inplace(_ctx, cur, embeddings);

        embeddings = cur;  // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(_ctx, cur, eps_);
            cur = ggml_add_inplace(_ctx, ggml_mul(_ctx, cur, model_.layers[il].ln_2_w), model_.layers[il].ln_2_b);
        }

        cur = ggml_mul_mat(_ctx, model_.layers[il].ff_i_w, cur);
        cur = ggml_add_inplace(_ctx, cur, model_.layers[il].ff_i_b);

        if (use_gelu_) {
            cur = ggml_gelu_inplace(_ctx, cur);
        }
        cur = ggml_mul_mat(_ctx, model_.layers[il].ff_o_w, cur);
        cur = ggml_add_inplace(_ctx, cur, model_.layers[il].ff_o_b);

        // residual 2
        cur = ggml_add_inplace(_ctx, embeddings, cur);

        embeddings = cur;
    }

    embeddings = ggml_norm(_ctx, embeddings, eps_);
    ggml_set_name(embeddings, "post_ln");

    embeddings = ggml_add_inplace(_ctx, ggml_mul(_ctx, embeddings, model_.post_ln_w), model_.post_ln_b);

    // projector, resampler
    struct ggml_tensor* q = model_.mm_model_query;
    {  // layernorm
        q = ggml_norm(_ctx, q, eps_);
        q = ggml_add_inplace(_ctx, ggml_mul(_ctx, q, model_.mm_model_ln_q_w), model_.mm_model_ln_q_b);
    }
    struct ggml_tensor* v = ggml_mul_mat(_ctx, model_.mm_model_kv_proj, embeddings);
    {  // layernorm
        v = ggml_norm(_ctx, v, eps_);
        v = ggml_add_inplace(_ctx, ggml_mul(_ctx, v, model_.mm_model_ln_kv_w), model_.mm_model_ln_kv_b);
    }
    struct ggml_tensor* k;
    {  // position
        // q = ggml_add(_ctx, q, model_.mm_model_pos_embed);
        k = ggml_add_inplace(_ctx, v, pos_embed);
    }

    {  // attention
        // int hidden_size = 4096;
        const int d_head    = 128;
        const int n_head    = embed_dim_ / d_head;
        const int num_query = 64;

        struct ggml_tensor* Q = ggml_add(_ctx, ggml_mul_mat(_ctx, model_.mm_model_attn_q_w, q), model_.mm_model_attn_q_b);
        struct ggml_tensor* K = ggml_add(_ctx, ggml_mul_mat(_ctx, model_.mm_model_attn_k_w, k), model_.mm_model_attn_k_b);
        struct ggml_tensor* V = ggml_add(_ctx, ggml_mul_mat(_ctx, model_.mm_model_attn_v_w, v), model_.mm_model_attn_v_b);
        // permute
        Q = ggml_reshape_4d(_ctx, Q, d_head, n_head, num_query, batch_size);
        Q = ggml_cont(_ctx, ggml_permute(_ctx, Q, 0, 2, 1, 3));
        Q = ggml_reshape_3d(_ctx, Q, d_head, num_query, n_head * batch_size);
        K = ggml_reshape_4d(_ctx, K, d_head, n_head, num_positions, batch_size);
        K = ggml_cont(_ctx, ggml_permute(_ctx, K, 0, 2, 1, 3));
        K = ggml_reshape_3d(_ctx, K, d_head, num_positions, n_head * batch_size);
        V = ggml_reshape_4d(_ctx, V, d_head, n_head, num_positions, batch_size);
        V = ggml_cont(_ctx, ggml_permute(_ctx, V, 1, 2, 0, 3));
        V = ggml_reshape_3d(_ctx, V, num_positions, d_head, n_head * batch_size);

        ggml_tensor* KQV = nullptr;
        ggml_tensor* KQ  = nullptr;
        if (use_fa_) {
            K   = ggml_pad(_ctx, K, 0, GGML_PAD(K->ne[1], 256) - K->ne[1], 0, 0);
            V   = ggml_pad(_ctx, V, GGML_PAD(V->ne[0], 256) - V->ne[0], 0, 0, 0);
            KQV = build_flash_attn(_ctx, gf, Q, K, V, nullptr, true, 1.0f / sqrt((float)d_head));
        } else {
            KQ  = ggml_mul_mat(_ctx, K, Q);
            KQ  = ggml_soft_max_ext(_ctx, KQ, nullptr, 1.0f / sqrt((float)d_head), 0.0f);
            KQV = ggml_mul_mat(_ctx, V, KQ);
            KQV = ggml_reshape_4d(_ctx, KQV, d_head, num_query, n_head, batch_size);
            KQV = ggml_permute(_ctx, KQV, 0, 2, 1, 3);
            KQV = ggml_cont_3d(_ctx, KQV, embed_dim_, num_query, batch_size);
        }
        embeddings = ggml_add_inplace(_ctx, ggml_mul_mat(_ctx, model_.mm_model_attn_o_w, KQV), model_.mm_model_attn_o_b);
    }

    {  // layernorm
        embeddings = ggml_norm(_ctx, embeddings, eps_);
        embeddings = ggml_add_inplace(_ctx, ggml_mul(_ctx, embeddings, model_.mm_model_ln_post_w), model_.mm_model_ln_post_b);
    }
    embeddings = ggml_mul_mat(_ctx, model_.mm_model_proj, embeddings);

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    ggml_free(_ctx);

    // ggml_gallocr_alloc_graph(compute_alloc_, gf);

    // size_t compute_memory_buffer_size = ggml_gallocr_get_buffer_size(compute_alloc_, 0);
    // LOG_INF("%s: compute allocated memory: %.2f MB\n", __func__, compute_memory_buffer_size / 1024.0 / 1024.0);

    return gf;
}

size_t Siglip::_embd_nbytes() {
    size_t bytes  = 0;
    int n_patches = 64;  // for minicpm-omni
    bytes         = n_patches * embed_dim_ * sizeof(float);
    return bytes;
}

// TODO:(lvsen) support multiple images
void Siglip::forward(const image_buf<float>& imgs, const int img_w, const int img_h, std::vector<float>& out) {
    auto gf = this->build_graph(imgs.nx, imgs.ny);
    ggml_gallocr_alloc_graph(compute_alloc_, gf);

    // set inputs

    const int pos_w = img_w / patch_size_;
    const int pos_h = img_h / patch_size_;

    {
        struct ggml_tensor* inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        std::vector<float> data(ggml_nelements(inp_raw), 0);

        const int batch_size  = 1;
        const int image_batch = 1;
        const int nx          = imgs.nx;
        const int ny          = imgs.ny;
        const int n           = nx * ny;
        for (size_t i = 0; i < image_batch; i++) {
            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < 3; k++) {
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            const size_t dst_idx = (b * 3 * n) + k * n + y * nx + x;
                            const size_t src_idx = 3 * (y * nx + x) + k;
                            data[dst_idx]        = imgs.buf[src_idx];
                        }
                    }
                }
            }
        }
        ggml_backend_tensor_set(inp_raw, data.data(), 0, ggml_nbytes(inp_raw));
    }

    {
        // inspired from siglip:
        //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
        //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit/blob/d66538faeba44480d0bfaa42145eef26f9423199/modeling_siglip.py#L316
        struct ggml_tensor* positions = ggml_graph_get_tensor(gf, "positions");
        std::vector<int> positions_data(ggml_nelements(positions), 0);
        int bucket_coords_h[1024];
        int bucket_coords_w[1024];
        for (int i = 0; i < pos_h; i++) {
            bucket_coords_h[i] = std::floor(70.0 * i / pos_h);
        }
        for (int i = 0; i < pos_w; i++) {
            bucket_coords_w[i] = std::floor(70.0 * i / pos_w);
        }
        for (int i = 0; i < pos_h; i++) {
            for (int j = 0; j < pos_w; j++) {
                const int id       = i * pos_w + j;
                positions_data[id] = bucket_coords_h[i] * 70 + bucket_coords_w[j];
            }
        }
        ggml_backend_tensor_set(positions, positions_data.data(), 0, ggml_nbytes(positions));
    }

    {
        // inspired from resampler of Qwen-VL:
        //    -> https://huggingface.co/Qwen/Qwen-VL/tree/main
        //    -> https://huggingface.co/Qwen/Qwen-VL/blob/0547ed36a86561e2e42fecec8fd0c4f6953e33c4/visual.py#L23
        struct ggml_tensor* pos_embed = ggml_graph_get_tensor(gf, "pos_embed");
        auto pos_embed_t              = get_2d_sincos_pos_embed(embed_dim_, std::make_pair(pos_w, pos_h));

        std::vector<float> pos_embed_data(ggml_nelements(pos_embed), 0);
        for (int i = 0; i < pos_w * pos_h; ++i) {
            for (int j = 0; j < embed_dim_; ++j) {
                const int dst_idx       = i * embed_dim_ + j;
                pos_embed_data[dst_idx] = pos_embed_t[i][j];
            }
        }

        ggml_backend_tensor_set(pos_embed, pos_embed_data.data(), 0, ggml_nbytes(pos_embed));
    }

    if (ggml_backend_is_cpu(backend_)) {
        auto n_threads = std::thread::hardware_concurrency() / 2;
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }

    ggml_backend_graph_compute(backend_, gf);

    // the last node is the embedding tensor
    struct ggml_tensor* embeddings = ggml_graph_node(gf, -1);

    // copy the embeddings to the location passed by the user
    ggml_backend_tensor_get(embeddings, out.data(), 0, ggml_nbytes(embeddings));
}
}  // namespace edge
