# @Enhanced by @klynwuu 2026-02-20
# @OriginalAuthor: Bi Ying
# @Date:   2024-07-10 17:22:55
import asyncio
import shutil
import subprocess
import os
import time
from pathlib import Path
from typing import Union

import numpy as np
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, status
from tqdm import tqdm


app = FastAPI()

TMP_DIR = "./tmp"

# 确保临时目录存在
os.makedirs(TMP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model configuration: switch backend and model via env or edit here.
# - BACKEND: "sensevoice" (FunASR/ModelScope) or "mlx" (Hugging Face / MLX, faster on Mac)
# - LOCAL_MODEL: model identifier. For MLX use a Hugging Face repo id (e.g. mlx-community/Qwen3-ASR-1.7B-8bit).
#   Models are downloaded once to cache (~/.cache/huggingface/hub for MLX, ~/.cache/modelscope for SenseVoice).
# ---------------------------------------------------------------------------
BACKEND = os.getenv("BACKEND", "sensevoice").lower()  # "sensevoice" | "mlx"
LOCAL_MODEL = os.getenv(
    "LOCAL_MODEL",
    "mlx-community/Qwen3-ASR-1.7B-8bit" if BACKEND == "mlx" else "iic/SenseVoiceSmall",
)

model = None
sensevoice_model = None
mlx_model = None


def _load_sensevoice():
    """Load SenseVoice (FunASR). Uses ModelScope; downloads once to ~/.cache/modelscope."""
    import torch
    import torchaudio
    from funasr import AutoModel

    local_model_path = os.path.expanduser(
        os.getenv("SENSEVOICE_LOCAL_PATH", "~/.cache/modelscope/hub/models/iic/SenseVoiceSmall")
    )
    local_vad_path = os.path.expanduser(
        os.getenv("SENSEVOICE_VAD_PATH", "~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
    )
    model_id = LOCAL_MODEL or "iic/SenseVoiceSmall"
    vad_id = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

    if os.path.exists(local_model_path) and os.path.exists(local_vad_path):
        m = AutoModel(
            model=local_model_path,
            vad_model=local_vad_path,
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=True,
            disable_update=True,
        )
        print(f"[SenseVoice] Using local model: {local_model_path}")
    else:
        m = AutoModel(
            model=model_id,
            vad_model=vad_id,
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=True,
            disable_update=True,
        )
        print(f"[SenseVoice] Model not in cache; downloading once: {model_id}")
    return m


def _load_mlx():
    """Load MLX STT model (e.g. Qwen3-ıASR). Uses Hugging Face; downloads once to ~/.cache/huggingface/hub."""
    from mlx_audio.stt.utils import load_model as load_stt_model

    model_id = LOCAL_MODEL or "mlx-community/Qwen3-ASR-1.7B-8bit"
    # load_model downloads once and caches; subsequent runs use cache
    m = load_stt_model(model_id)
    print(f"[MLX] Model loaded (from cache if previously downloaded): {model_id}")
    return m


# Initialize the selected backend (downloads once on first run, then uses local cache)
if BACKEND == "mlx":
    mlx_model = _load_mlx()
    model = "mlx"
else:
    sensevoice_model = _load_sensevoice()
    model = "sensevoice"


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "message": "OpenAI Compatible API Server (SenseVoice / MLX)",
        "version": "1.0.0",
        "backend": BACKEND,
        "model": LOCAL_MODEL,
        "endpoints": {
            "transcriptions": "/v1/audio/transcriptions",
            "models": "/v1/models",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "healthy", "backend": BACKEND, "model": LOCAL_MODEL}

emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {
    "🎼",
    "👏",
    "😀",
    "😭",
    "🤧",
    "😷",
}


def format_str_v2(text: str, show_emo=True, show_event=True):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = text.count(sptk)
        text = text.replace(sptk, "")

    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    if show_emo:
        text = text + emo_dict[emo]

    for e in event_dict:
        if sptk_dict[e] > 0 and show_event:
            text = event_dict[e] + text

    for emoji in emo_set.union(event_set):
        text = text.replace(" " + emoji, emoji)
        text = text.replace(emoji + " ", emoji)

    return text.strip()


def format_str_v3(text: str, show_emo=True, show_event=True):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None

    def get_event(s):
        return s[0] if s[0] in event_set else None

    text = text.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        text = text.replace(lang, "<|lang|>")
    parts = [format_str_v2(part, show_emo, show_event).strip(" ") for part in text.split("<|lang|>")]
    new_s = " " + parts[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(parts)):
        if len(parts[i]) == 0:
            continue
        if get_event(parts[i]) == cur_ent_event and get_event(parts[i]) is not None:
            parts[i] = parts[i][1:]
        cur_ent_event = get_event(parts[i])
        if get_emo(parts[i]) is not None and get_emo(parts[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += parts[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def model_inference_sensevoice(input_wav, language, fs=16000, show_emo=True, show_event=True):
    import torch
    import torchaudio

    language = "auto" if len(language) < 1 else language
    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
    if len(input_wav) == 0:
        raise ValueError("The provided audio is empty.")
    merge_vad = True
    text = sensevoice_model.generate(
        input=input_wav,
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=0,
        merge_vad=merge_vad,
    )
    text = text[0]["text"]
    text = format_str_v3(text, show_emo, show_event)
    return text


def model_inference_mlx(audio_path: str, language: str = "auto", **_kwargs):
    # Model already loaded; HF cache used at load time. generate() reads from file.
    # Never pass None for language: some MLX code paths call .lower() on it and crash.
    lang_str = (language or "auto").strip().lower()
    lang = "auto" if lang_str in ("auto", "") else lang_str
    result = mlx_model.generate(audio_path, language=lang)
    return result.text if hasattr(result, "text") else str(result).strip()


def model_inference(audio_path=None, input_wav=None, language="auto", fs=16000, show_emo=True, show_event=True):
    """Run transcription. Dispatches to SenseVoice or MLX based on BACKEND."""
    if model == "mlx":
        if not audio_path:
            raise ValueError("MLX backend requires audio_path (path to audio file).")
        return model_inference_mlx(audio_path=audio_path, language=language)
    else:
        if input_wav is None:
            raise ValueError("SenseVoice backend requires input_wav.")
        return model_inference_sensevoice(input_wav=input_wav, language=language, fs=fs, show_emo=show_emo, show_event=show_event)


@app.get("/v1/models")
async def models():
    """返回可用模型列表，兼容OpenAI API格式"""
    return {
        "object": "list",
        "data": [
            {
                "id": LOCAL_MODEL,
                "object": "model",
                "created": 1677610602,
                "owned_by": BACKEND,
                "root": LOCAL_MODEL,
                "parent": None,
                "permission": [
                    {
                        "id": "modelperm-123",
                        "object": "model_permission",
                        "created": 1677610602,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ]
            }
        ]
    }


def _preprocess_audio(input_path: Path, output_path: Path) -> Path:
    """Downsample audio to 16kHz mono WAV using ffmpeg.

    ASR models only need 16kHz mono. Converting upfront reduces memory usage
    and speeds up inference — especially for high-quality inputs (44.1/48kHz stereo).
    The API contract is unchanged; callers can still send any format ffmpeg supports.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-ar", "16000",      # 16kHz sample rate
                "-ac", "1",          # mono
                "-sample_fmt", "s16", # 16-bit (sufficient for speech)
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg not available or conversion failed — fall back to original file
        return input_path


def _get_audio_duration_no_torch(path: Path) -> float:
    """Get duration in seconds without torch. Uses stdlib wave for .wav; else default."""
    try:
        if path.suffix.lower() == ".wav":
            import wave
            with wave.open(str(path), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                return frames / float(rate) if rate else 5.0
    except Exception:
        pass
    return 5.0  # fallback so progress bar still moves


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: Union[UploadFile, None] = File(default=None),
    language: Union[str, None] = Form(default="auto"),
):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bad Request, no file provided")

    # Ensure language is never None (e.g. when client omits form field) to avoid .lower() errors in backends
    language = (language or "auto").strip() or "auto"

    os.makedirs(TMP_DIR, exist_ok=True)
    filename = file.filename
    fileobj = file.file
    tmp_file = Path(TMP_DIR) / filename

    preprocessed_file = None
    try:
        with open(tmp_file, "wb+") as upload_file:
            shutil.copyfileobj(fileobj, upload_file)

        # Downsample to 16kHz mono WAV before inference (smaller, faster, less memory)
        preprocessed_path = tmp_file.with_suffix(".16k.wav")
        actual_file = _preprocess_audio(tmp_file, preprocessed_path)
        if actual_file != tmp_file:
            preprocessed_file = actual_file  # track for cleanup

        # Duration for progress bar: never use torch here so MLX-only envs work (no torch installed).
        duration_seconds = _get_audio_duration_no_torch(actual_file)
        estimated_seconds = max(1.0, duration_seconds)

        backend = model  # capture for closure

        def run_inference():
            if backend == "mlx":
                return model_inference(audio_path=str(actual_file), language=language)
            # SenseVoice: load and convert inside this thread so torch is only imported here.
            import torch
            import torchaudio
            waveform, sample_rate = torchaudio.load(actual_file)
            waveform_int = (waveform * np.iinfo(np.int32).max).to(dtype=torch.int32).squeeze()
            if len(waveform_int.shape) > 1:
                waveform_int = waveform_int.float().mean(axis=0)
            input_wav = (sample_rate, waveform_int.numpy())
            return model_inference(input_wav=input_wav, language=language, show_emo=False)

        # MLX/Metal is NOT thread-safe — running inference in a background thread
        # causes "A command encoder is already encoding to this command buffer" crashes.
        # Run MLX inference directly on the main thread; use executor only for SenseVoice.
        if backend == "mlx":
            pbar = tqdm(total=100, desc="Transcribing", unit="%", ncols=80, leave=True)
            try:
                pbar.n = 10
                pbar.refresh()
                result = run_inference()
                pbar.n = 100
                pbar.refresh()
                return {"text": result}
            finally:
                pbar.close()
        else:
            pbar = tqdm(total=100, desc="Transcribing", unit="%", ncols=80, leave=True)
            try:
                async def update_progress(fut: asyncio.Future):
                    start = time.perf_counter()
                    while not fut.done():
                        await asyncio.sleep(0.1)
                        elapsed = time.perf_counter() - start
                        n = min(99, int(elapsed / estimated_seconds * 100))
                        pbar.n = n
                        pbar.refresh()

                loop = asyncio.get_running_loop()
                inference_fut = loop.run_in_executor(None, run_inference)
                progress_task = asyncio.create_task(update_progress(inference_fut))
                try:
                    result = await inference_fut
                finally:
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                pbar.n = 100
                return {"text": result}
            finally:
                pbar.close()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing audio file: {str(e)}")
    finally:
        if tmp_file.exists():
            tmp_file.unlink()
        if preprocessed_file and preprocessed_file.exists():
            preprocessed_file.unlink()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
