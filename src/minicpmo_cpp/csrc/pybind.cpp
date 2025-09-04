#include <pybind11/pybind11.h>
#include "minicpmo_iface.hpp"

PYBIND11_MODULE(_C, m) {
    m.doc() = "pybind11 for minicpmo-cpp";

    m.def("get_minicpmo_default_llm_params", &get_minicpmo_default_llm_params, "default llm params for MiniCPM-omni");

    py::class_<minicpmo>(m, "minicpmo")
        .def(py::init<minicpmo_params>())
        .def("streaming_prefill", &minicpmo::streaming_prefill)
        .def("eval_system_prompt", &minicpmo::eval_system_prompt, py::arg("lang") = "en")
        .def("streaming_generate", &minicpmo::streaming_generate, py::arg("prompt"))
        .def("chat_generate", &minicpmo::chat_generate, py::arg("prompt"), py::arg("stream") = true)
        .def("apm_kv_clear", &minicpmo::apm_kv_clear)
        .def("apm_streaming_mode", &minicpmo::apm_streaming_mode, py::arg("stream") = false)
        .def("reset", &minicpmo::reset);

    py::class_<minicpmo_params>(m, "minicpmo_params")
        .def(py::init<>())
        .def_readwrite("vpm_path", &minicpmo_params::vpm_path)
        .def_readwrite("apm_path", &minicpmo_params::apm_path)
        .def_readwrite("llm_path", &minicpmo_params::llm_path)
        .def_readwrite("ttc_model_path", &minicpmo_params::ttc_model_path)
        .def_readwrite("cts_model_path", &minicpmo_params::cts_model_path)
        .def_readwrite("common_params", &minicpmo_params::llm_params);

    py::class_<common_params>(m, "common_params")
        .def(py::init<>())
        .def_readwrite("n_batch", &common_params::n_batch)
        .def_readwrite("n_predict", &common_params::n_predict)
        .def_readwrite("n_ctx", &common_params::n_ctx)
        .def_readwrite("flash_attn", &common_params::flash_attn)
        .def_readwrite("n_keep", &common_params::n_keep);
}
