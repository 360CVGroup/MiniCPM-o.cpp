import math
import tempfile
import librosa
import argparse
import numpy as np
from PIL import Image
from moviepy import VideoFileClip

from minicpmo_cpp import CommonParams, MinicpmoParams, MiniCPMO  # type: ignore


# fmt: off
parser = argparse.ArgumentParser(description="MiniCPMO video/audio streaming test")
parser.add_argument("--apm-path", type=str, required=True, help="Path to audio encoder model")
parser.add_argument("--vpm-path", type=str, required=True, help="Path to image encoder model")
parser.add_argument("--llm-path", type=str, required=True, help="Path to LLM model")
parser.add_argument("--video-path", type=str, required=True, help="Path to input video")
parser.add_argument("--n-ctx", type=int, default=8192, help="Context size")
parser.add_argument("--n-keep", type=int, default=4, help="Keep tokens")
parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
# fmt: on


def get_video_chunk_content(video_path):
    video = VideoFileClip(video_path)
    print("video_duration:", video.duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

    # 1 frame + 1s audio chunk
    contents = []
    for i in range(num_units):
        frame = video.get_frame(i + 1)  # HWC

        h, w = frame.shape[:2]
        target_long_side = 640
        scale = target_long_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_image = Image.fromarray(frame)
        resized_pil = pil_image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        image = resized_pil

        audio = audio_np[sr * i : sr * (i + 1)]
        # padding to 1s
        if audio.size < sr:
            audio_pad = np.zeros(sr, dtype=np.float32)
            audio_pad[: audio.size] = audio
            audio = audio_pad
        contents.append([image, audio])

    return contents


def streaming_process(minicpmo, contents, clear_kv=True):
    if clear_kv:
        minicpmo.apm_kv_clear()
    minicpmo.apm_streaming_mode(True)

    for image, audio in contents:
        minicpmo.streaming_prefill(image, audio)
    result = minicpmo.chat_generate("", False)
    return result


def main():
    args = parser.parse_args()
    use_flash_attn = not args.no_flash_attn

    c_params = CommonParams(n_ctx=args.n_ctx, n_keep=args.n_keep, flash_attn=use_flash_attn)
    model_params = MinicpmoParams(
        apm_path=args.apm_path,
        vpm_path=args.vpm_path,
        llm_path=args.llm_path,
        common_params=c_params,
    )
    minicpmo = MiniCPMO(model_params)

    video_path = args.video_path
    contents = get_video_chunk_content(video_path)
    minicpmo.eval_system_prompt("en")

    minicpmo.apm_kv_clear()
    minicpmo.apm_streaming_mode(True)

    user_prompt = ""  # can set user prompt here

    for image, audio in contents:
        minicpmo.streaming_prefill(image, audio)
    for s in minicpmo.streaming_generate(prompt=user_prompt):
        print(s, end="")
    print()


if __name__ == "__main__":
    main()
