import numpy as np
from typing import Generator

from . import _C as m


class CommonParams(object):
    def __init__(self, n_ctx: int = 8192, n_keep: int = 0, flash_attn: bool = False):
        self._cparams = m.get_minicpmo_default_llm_params()
        self._cparams.n_ctx = n_ctx
        self._cparams.n_keep = n_keep
        self._cparams.flash_attn = flash_attn

    def get(self):
        return self._cparams


class MinicpmoParams:
    def __init__(
        self,
        vpm_path: str = "",
        apm_path: str = "",
        llm_path: str = "",
        ttc_model_path: str = "",
        cts_model_path: str = "",
        common_params: CommonParams = None,
    ):
        self.params = m.minicpmo_params()
        self.params.vpm_path = vpm_path
        self.params.apm_path = apm_path
        self.params.llm_path = llm_path
        self.params.ttc_model_path = ttc_model_path
        self.params.cts_model_path = cts_model_path
        self.params.common_params = common_params.get()

    def get(self):
        return self.params


class MiniCPMO:
    def __init__(self, params: MinicpmoParams):
        self._lib = m
        self._minicpmo = self._lib.minicpmo(params.get())

    def streaming_prefill(self, image: np.ndarray, pcmf32: np.ndarray) -> None:
        self._minicpmo.streaming_prefill(image, pcmf32)

    def eval_system_prompt(self, lang: str = "en") -> None:
        self._minicpmo.eval_system_prompt(lang)

    def streaming_generate(self, prompt: str = "") -> Generator[str, str, str]:
        ret = self._minicpmo.streaming_generate(prompt)
        while len(ret) != 0:
            yield ret
            ret = self._minicpmo.streaming_generate(prompt)

    def chat_generate(self, prompt: str = "", stream: bool = True) -> str:
        return self._minicpmo.chat_generate(prompt, stream)

    def apm_kv_clear(self) -> None:
        self._minicpmo.apm_kv_clear()

    def apm_streaming_mode(self, stream: bool) -> None:
        self._minicpmo.apm_streaming_mode(stream)

    def reset(self) -> None:
        self._minicpmo.reset()
