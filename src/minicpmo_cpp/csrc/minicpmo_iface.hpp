#include <pybind11/numpy.h>
#include "minicpmo.h"

namespace py = pybind11;
using namespace edge;

struct minicpmo {
public:
    minicpmo(minicpmo_params params) {
        minicpmo_ = std::make_shared<edge::MiniCPMO>(params);
    }
    // ~minicpmo() {}

    void streaming_prefill(py::array_t<uint8_t> image_npy, py::array_t<float> pcmf32_npy) {
        const int sample_rate = 16000;
        image_buf<uint8_t> image;
        std::vector<float> pcmf32(sample_rate, 0);

        {
            py::buffer_info info = image_npy.request();
            const int ndims      = info.ndim;
            assert(ndims == 3 && format("only support RGB format image, but got dims: %d", ndims).c_str());

            uint8_t* data = static_cast<uint8_t*>(info.ptr);
            std::vector<uint8_t> vec(data, data + info.size);
            // image.buf = std::move(vec);
            image.buf = vec;
            image.nx  = info.shape[1];  // width
            image.ny  = info.shape[0];  // height
        }
        {
            py::buffer_info info = pcmf32_npy.request();
            const int ndims      = info.ndim;
            assert(ndims == 1 && format("only support 1 dim format pcm data, but got dims: %d", ndims).c_str());
            const int size = info.size;
            assert(size == sample_rate && format("only support 16k sample rate, but got size: %d", size).c_str());
            float* data = static_cast<float*>(info.ptr);
            std::memcpy(pcmf32.data(), data, sample_rate * sizeof(float));
        }
        minicpmo_->streaming_prefill(image, pcmf32);
    }

    void eval_system_prompt(std::string lang) { minicpmo_->eval_system_prompt(lang); }
    std::string streaming_generate(std::string prompt) { return minicpmo_->streaming_generate(prompt); }
    std::string chat_generate(std::string prompt, bool stream) { return minicpmo_->_chat(prompt, stream); }
    void apm_streaming_mode(bool stream) { minicpmo_->apm_streaming_mode(stream); }
    void apm_kv_clear() { minicpmo_->apm_kv_clear(); }
    void reset() { minicpmo_->reset(); }

private:
    std::shared_ptr<MiniCPMO> minicpmo_;
};
