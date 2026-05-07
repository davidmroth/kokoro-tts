from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import os
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response
from kokoro_onnx import Kokoro
import soundfile as sf


DEFAULT_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
DEFAULT_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _model_dir() -> Path:
    return Path(os.getenv("KOKORO_MODEL_DIR", "/models"))


def _model_path() -> Path:
    return _model_dir() / os.getenv("KOKORO_MODEL_FILENAME", "kokoro-v1.0.onnx")


def _voices_path() -> Path:
    return _model_dir() / os.getenv("KOKORO_VOICES_FILENAME", "voices-v1.0.bin")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, NamedTemporaryFile(dir=destination.parent, delete=False) as tmp_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            tmp_file.write(chunk)
        temp_path = Path(tmp_file.name)
    temp_path.replace(destination)


def _ensure_assets() -> None:
    model_path = _model_path()
    voices_path = _voices_path()
    if not model_path.exists():
        _download(os.getenv("KOKORO_MODEL_URL", DEFAULT_MODEL_URL), model_path)
    if not voices_path.exists():
        _download(os.getenv("KOKORO_VOICES_URL", DEFAULT_VOICES_URL), voices_path)


@lru_cache(maxsize=1)
def _engine() -> Kokoro:
    _ensure_assets()
    return Kokoro(str(_model_path()), str(_voices_path()))


def _render_wav_bytes(text: str, *, voice: str, lang: str, speed: float) -> bytes:
    samples, sample_rate = _engine().create(text, voice=voice, speed=speed, lang=lang)
    buffer = BytesIO()
    buffer.name = "audio.wav"
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


app = FastAPI(title="Kokoro TTS Adapter", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    _ensure_assets()
    return {"status": "ok"}


@app.get("/voices")
def voices() -> dict[str, list[str]]:
    return {"voices": sorted(_engine().get_voices())}


@app.post("/tts")
def tts(
    text: str = Form(...),
    voice: str | None = Form(None),
    lang: str | None = Form(None),
    speed: float | None = Form(None),
) -> Response:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="text is required")

    resolved_voice = (voice or os.getenv("KOKORO_VOICE", "af_sky")).strip() or "af_sky"
    resolved_lang = (lang or os.getenv("KOKORO_LANG", "en-us")).strip() or "en-us"
    resolved_speed = speed if speed is not None else float(os.getenv("KOKORO_SPEED", "1.0"))

    try:
        audio_bytes = _render_wav_bytes(
            cleaned_text,
            voice=resolved_voice,
            lang=resolved_lang,
            speed=resolved_speed,
        )
    except AssertionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"kokoro synthesis failed: {exc}") from exc

    return Response(content=audio_bytes, media_type="audio/wav")