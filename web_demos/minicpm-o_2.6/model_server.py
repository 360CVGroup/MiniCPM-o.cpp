import base64
import json
import asyncio
import os
import sys
import io
import threading
import time
import librosa
import soundfile
import wave
from typing import Dict, Any, Optional
import argparse
import logging
from PIL import Image
import numpy as np
import uvicorn
from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from minicpmo_cpp import _C as m  # type: ignore
from minicpmo_cpp import CommonParams, MinicpmoParams  # type: ignore


ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=32550)
ap.add_argument(
    "--dump-data",
    action="store_true",
    default=False,
    help="dump image and audio slice fed into model",
)
ap.add_argument("--apm-path", type=str, required=True, help="Path to audio encoder model")
ap.add_argument("--vpm-path", type=str, required=True, help="Path to image encoder model")
ap.add_argument("--llm-path", type=str, required=True, help="Path to LLM model")
ap.add_argument("--n-ctx", type=int, default=8192, help="Context size")
ap.add_argument("--n-keep", type=int, default=4, help="Keep tokens")
ap.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention")
args = ap.parse_args()


def setup_logger():
    logger = logging.getLogger("api_logger")
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d-%(levelname)s-[%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create handlers for stdout and stderr
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)  # INFO and DEBUG go to stdout
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)  # WARNING, ERROR, CRITICAL go to stderr
    stderr_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger


app = FastAPI()
logger = setup_logger()

c_params = CommonParams(n_ctx=args.n_ctx, n_keep=args.n_keep, flash_attn=not args.no_flash_attn)
model_params = MinicpmoParams(
    vpm_path=args.vpm_path,
    apm_path=args.apm_path,
    llm_path=args.llm_path,
    common_params=c_params,
)


class MiniCPMOServer:
    def __init__(self, params: MinicpmoParams):
        self._lib = m
        self._minicpmo = self._lib.minicpmo(params.params)
        self._minicpmo.apm_streaming_mode(True)
        self.session_id = None
        self.warm_up(3)
        self.reset_session()

    def warm_up(self, warmup_steps: int = 2):
        # Dummy image
        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Dummy audio (1 second of silence)
        fake_audio = np.zeros((16000,), dtype=np.float32)
        for _ in range(warmup_steps):
            self.streaming_prefill(-1, fake_image, fake_audio)
        for _ in self.streaming_generate(-1, ""):
            pass

    def reset_session(self):
        self.session_id = None
        self.reset()

    def streaming_prefill(self, session_id: int, image: np.ndarray, pcmf32: np.ndarray) -> None:
        assert session_id is not None
        if self.session_id is None or session_id != self.session_id:  # new session
            self.session_id = session_id
            self.is_first = True
        else:
            self.is_first = False

        if self.is_first:
            self.eval_system_prompt("zh")

        self._minicpmo.streaming_prefill(np.array(image), pcmf32)

    def eval_system_prompt(self, lang: str = "en") -> None:
        self._minicpmo.eval_system_prompt(lang)

    def streaming_generate(self, session_id: int, prompt: str = ""):
        ret = self._minicpmo.streaming_generate(prompt)
        while len(ret) != 0:
            yield ret
            ret = self._minicpmo.streaming_generate(prompt)

    def text_to_speech(self, text: str, output_wav: str) -> bool:
        return self._minicpmo.text_to_speech(text, output_wav)

    def apm_kv_clear(self) -> None:
        self._minicpmo.apm_kv_clear()

    def apm_streaming_mode(self, stream: bool) -> None:
        self._minicpmo.apm_streaming_mode(stream)

    def reset(self) -> None:
        self._minicpmo.reset()
        self._minicpmo.apm_kv_clear()


class StreamManager:
    def __init__(self):
        self.debugging = True if args.dump_data else False
        self.uid = None

        self.is_streaming_complete = threading.Event()
        self.conversation_started = threading.Event()
        self.last_request_time = None
        self.last_stream_time = None
        self.timeout = 900  # seconds timeout
        self.stream_timeout = 3  # seconds no stream
        self.stream_started = False
        self.stop_response = False

        self.audio_prefill = []
        self.audio_input = []
        self.image_prefill = None
        self.audio_chunk = 200

        self.minicpmo_model = MiniCPMOServer(model_params)
        self.minicpmo_model.apm_streaming_mode(True)

        self.input_audio_id = 0
        self.output_audio_id = 0
        self.flag_decode = False
        self.cnts = None

        self.session_id = 233

        self.server_wait = True

        self.past_session_id = 0
        self.dump_data_init()
        self.past_session_id = self.session_id
        self.session_id += 1

    def start_conversation(self):
        logger.info(f"uid {self.uid}: new conversation started.")
        self.conversation_started.set()
        self.stop_response = False

    def update_last_request_time(self):
        self.last_request_time = time.time()
        # logger.info(f"update last_request_time {self.last_request_time}")

    def update_last_stream_time(self):
        self.last_stream_time = time.time()
        # logger.info(f"update last_stream_time {self.last_stream_time}")

    def reset(self):
        logger.info("reset")
        self.is_streaming_complete.clear()
        self.conversation_started.clear()
        self.last_request_time = None
        self.last_stream_time = None
        self.stream_started = False
        self.stop_response = False

        self.minicpmo_model.reset()
        # clear model
        self.clear()

    def merge_wav_files(self, input_bytes_list, output_file):
        with wave.open(io.BytesIO(input_bytes_list[0]), "rb") as wav:
            params = wav.getparams()
            n_channels, sampwidth, framerate, n_frames, comptype, compname = params

        with wave.open(output_file, "wb") as output_wav:
            output_wav.setnchannels(n_channels)
            output_wav.setsampwidth(sampwidth)
            output_wav.setframerate(framerate)
            output_wav.setcomptype(comptype, compname)

            for wav_bytes in input_bytes_list:
                with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                    output_wav.writeframes(wav.readframes(wav.getnframes()))

    def is_timed_out(self):
        if self.last_request_time is not None:
            return time.time() - self.last_request_time > self.timeout
        return False

    def no_active_stream(self):
        if self.last_stream_time is not None and self.stream_started:
            no_stream_duration = time.time() - self.last_stream_time
            if no_stream_duration > self.stream_timeout:
                # logger.info(f"no active stream for {no_stream_duration} secs.")
                return True
        return False

    def dump_data_init(self):
        self.savedir = os.path.join(
            f"{os.path.abspath(os.path.dirname(__file__))}/log_data/{args.port}/", str(time.time())
        )
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        if not os.path.exists(self.savedir + "/input_audio_log"):
            os.makedirs(self.savedir + "/input_audio_log")
        if not os.path.exists(self.savedir + "/input_image_log"):
            os.makedirs(self.savedir + "/input_image_log")
        # debugging
        if self.debugging:
            if not os.path.exists(self.savedir + "/all_input_audio_log"):
                os.makedirs(self.savedir + "/all_input_audio_log")

    def cleanup_dump_data(self):
        import shutil

        full_path = os.path.realpath(self.savedir)
        if not self.debugging and os.path.exists(self.savedir):
            shutil.rmtree(full_path)
            logger.info(f"Removed savedir {full_path}")

    def clear(self):
        try:
            self.flag_decode = False
            self.stream_started = False
            self.cnts = None
            self.audio_prefill = []
            self.audio_input = []
            self.image_prefill = None

            # if self.minicpmo_model.get_kv_cache_size() > 8192:
            #     self.session_id += 1  # to clear all kv cache
            #     self.sys_prompt_flag = False

        except Exception as e:
            raise ValueError(f"Clear error: {str(e)}")

    def process_message(self, message: Dict[str, Any]):
        try:
            # Process content items
            audio_data = None
            image_data = None
            for content_item in message["content"]:
                if content_item["type"] == "stop_response":
                    logger.info("process_message: received request to stop_response")
                    self.stop_response = True
                    return "stop"
                elif content_item["type"] == "input_audio":
                    audio_data = content_item["input_audio"]["data"]
                    # audio_timestamp = content_item["input_audio"].get("timestamp", "")
                elif content_item["type"] == "image_data":
                    image_data = content_item["image_data"]["data"]
            if audio_data is None:
                return "empty audio"

            if self.conversation_started.is_set() and self.is_streaming_complete.is_set():
                self.audio_prefill.clear()
                logger.info("conversation not started or still in generation, skip stream message.")
                return "skip"

            if self.flag_decode:
                return "skip"

            try:
                audio_bytes = base64.b64decode(audio_data)

                image = None
                if image_data is not None:
                    if len(image_data) > 0:
                        image_bytes = base64.b64decode(image_data)
                        image_buffer = io.BytesIO(image_bytes)
                        image_buffer.seek(0)
                        image = Image.open(image_buffer)
                        # logger.info("read image")

                self.prefill(audio_bytes, image, False)

                return "done"

            except Exception as e:
                raise ValueError(f"Audio processing error: {str(e)}")

        except Exception as e:
            raise ValueError(f"Message processing error: {str(e)}")

    def prefill(self, audio, image: Image, is_end: bool):
        if self.flag_decode:
            return False

        if image is not None:
            self.image_prefill = image
        try:
            self.audio_prefill.append(audio)
            if self.debugging:
                self.audio_input.append(audio)
            if (len(self.audio_prefill) == (1000 / self.audio_chunk)) or (is_end and len(self.audio_prefill) > 0):
                # time_prefill = time.time()
                input_audio_path = self.savedir + f"/input_audio_log/input_audio_{self.input_audio_id}.wav"
                self.merge_wav_files(self.audio_prefill, input_audio_path)
                with open(input_audio_path, "rb") as wav_io:
                    signal, sr = soundfile.read(wav_io, dtype="float32")
                    soundfile.write(input_audio_path, signal, 16000)
                    audio_np, sr = librosa.load(input_audio_path, sr=16000, mono=True)
                self.audio_prefill = []

                if len(audio_np) > 16000:
                    audio_np = audio_np[:16000]

                if self.image_prefill is not None:
                    input_image_path = self.savedir + f"/input_image_log/input_image_{self.input_audio_id}.png"
                    if args.dump_data:
                        self.image_prefill.save(input_image_path, "PNG")
                    self.image_prefill = self.image_prefill.convert("RGB")

                cnts = None
                if self.image_prefill is not None:
                    cnts = ["<unit>", self.image_prefill, audio_np]
                else:
                    cnts = [audio_np]

                if cnts is not None:
                    try:
                        self.minicpmo_model.streaming_prefill(
                            session_id=str(self.session_id),
                            image=cnts[1],
                            pcmf32=cnts[2],
                        )
                    except Exception as e:
                        logger.error(f"prefill error: {e}")
                        import traceback

                        traceback.print_exc()
                        raise

                self.input_audio_id += 1
            return True

        except Exception as e:
            logger.error(f"prefill error: {e}")
            import traceback

            traceback.print_exc()
            raise

    def generate_end(self):
        self.input_audio_id += 10
        self.output_audio_id += 10
        self.flag_decode = False
        self.reset()
        return

    async def generate(self):
        """return response text"""
        if self.stop_response:
            self.generate_end()
            return

        self.flag_decode = True
        try:
            logger.info("=== model gen start ===")

            # debugging
            if self.debugging:
                input_audio_path = self.savedir + f"/all_input_audio_log/all_input_audio_{self.input_audio_id}.wav"
                self.merge_wav_files(self.audio_input, input_audio_path)
                logger.info(f"input audio saved to {input_audio_path}")
            yield "assistant:\n"

            if self.stop_response:
                self.generate_end()
                return

            try:
                for r in self.minicpmo_model.streaming_generate(
                    session_id=str(self.session_id),
                    prompt="",
                ):
                    if self.stop_response:
                        self.generate_end()
                        return
                    text = r.replace("<|tts_eos|>", "")  # eof token filter
                    if self.debugging:
                        print("> text: ", text)
                    yield text
            except Exception as e:
                logger.error(f"Error happened during generation: {str(e)}")
            yield "\n<end>"

        except Exception as e:
            logger.error(f"发生异常:{e}")
            import traceback

            traceback.print_exc()
            raise

        finally:
            logger.info(f"uid {self.uid}: generation finished!")
            self.generate_end()

    async def check_activity(self):
        while True:
            # Check for overall inactivity (30 minutes)
            if self.is_timed_out():
                self.reset()
            if self.no_active_stream() and not self.is_streaming_complete.is_set():
                self.is_streaming_complete.set()

            await asyncio.sleep(1)  # Check every second


stream_manager = StreamManager()


@app.on_event("startup")
async def startup_event():
    logger.info("Starting application and activity checker")
    asyncio.create_task(stream_manager.check_activity())


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")
    try:
        stream_manager.cleanup_dump_data()
    except Exception as e:
        logger.error(f"Error during StreamManager cleanup: {str(e)}")


@app.post("/stream")
@app.post("/api/v1/stream")
async def stream(request: Request, uid: Optional[str] = Header(None)):
    global stream_manager

    stream_manager.update_last_request_time()
    stream_manager.update_last_stream_time()

    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid in headers")
    if stream_manager.uid is not None and stream_manager.uid != uid:
        logger.error(f"uid changed during steram: previous uid {stream_manager.uid}, new uid {uid}")
        raise HTTPException(status_code=400, detail="uid changed in stream")

    try:
        # Parse JSON request
        data = await request.json()

        # Validate basic structure
        if not isinstance(data, dict) or "messages" not in data:
            raise HTTPException(status_code=400, detail="Invalid request format")

        # Process messages
        reason = ""
        for message in data["messages"]:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise HTTPException(status_code=400, detail="Invalid message format")
            reason = stream_manager.process_message(message)

        # Return response using uid from header
        response = {
            "id": uid,
            "choices": {
                "role": "assistant",
                "content": "success",
                "finish_reason": reason,
            },
        }
        return JSONResponse(content=response, status_code=200)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_sse_response(request: Request, uid: Optional[str] = Header(None)):
    global stream_manager
    print(f"uid: {uid}")
    try:
        # stream_manager.is_streaming_complete.set()
        # # Wait for streaming to complete or timeout
        while not stream_manager.is_streaming_complete.is_set():
            # if stream_manager.is_timed_out():
            #     yield f"data: {json.dumps({'error': 'Stream timeout'})}\n\n"
            #     return
            # print(f"{uid} whille not stream_manager.is_streaming_complete.is_set(), asyncio.sleep(0.1)")
            await asyncio.sleep(0.1)

        logger.info("streaming complete\n")
        # Generate response
        try:
            yield "event: message\n"
            async for text in stream_manager.generate():
                if text == "stop":
                    break
                res = {
                    "id": stream_manager.uid,
                    "response_id": stream_manager.output_audio_id,
                    "choices": [
                        {
                            "role": "assistant",
                            "text": text,
                            "finish_reason": "processing",
                        }
                    ],
                }
                # logger.info("generate_sse_response yield response")
                yield f"data: {json.dumps(res)}\n\n"
                await asyncio.sleep(0)
            # now this conversation is done, reset session
            stream_manager.session_id += 1

        except Exception as e:
            logger.error(f"Error while generation: {str(e)}")
            yield f'data:{{"error": "{str(e)}"}}\n\n'
    except Exception as e:
        yield f'data:{{"error": "{str(e)}"}}\n\n'


@app.post("/btn_trigger")
@app.post("/api/v1/btn_trigger")
async def btn_trigger(request: Request, uid: Optional[str] = Header(None)):
    global stream_manager
    if not stream_manager.is_streaming_complete.is_set():
        await asyncio.sleep(1.1)  # wait for the streaming to complete
        stream_manager.is_streaming_complete.set()
    return {"status": "OK"}


@app.post("/completions")
@app.post("/api/v1/completions")
async def completions(request: Request, uid: Optional[str] = Header(None)):
    global stream_manager

    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid in headers")

    try:
        if stream_manager.uid != uid:
            stream_manager.session_id += 1
            logger.info(f"uid {uid}: session_id changed to {stream_manager.session_id}")
            stream_manager.reset()

        stream_manager.update_last_request_time()
        stream_manager.uid = uid
        stream_manager.start_conversation()

        return StreamingResponse(
            generate_sse_response(request, uid),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server busy, please try again later")
    except Exception as e:
        logger.error(f"Error processing request for user {uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop")
@app.post("/api/v1/stop")
async def stop_response(request: Request, uid: Optional[str] = Header(None)):
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid in headers")

    global stream_manager
    # stream_manager.session_id += 1
    logger.info(f"uid {uid}: received stop_response")
    stream_manager.stop_response = True
    response = {
        "id": uid,
        "choices": {"role": "assistant", "content": "success", "finish_reason": "stop"},
    }
    return JSONResponse(content=response, status_code=200)


@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    return {"status": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
