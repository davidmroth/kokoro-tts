from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import json
import os
import re
from tempfile import NamedTemporaryFile
from threading import RLock
from urllib.request import urlopen

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response
from kokoro_onnx import Kokoro
from pydantic import BaseModel
import soundfile as sf


DEFAULT_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
DEFAULT_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
_DECIMAL_PATTERN = re.compile(r"(?<!\d)(\d+)\.(\d+)(?!\d)")
_PRONUNCIATION_LOCK = RLock()
_NEMO_INIT_LOCK = RLock()


class PronunciationEntry(BaseModel):
    phrase: str
    pronunciation: str


class PronunciationBulkUpdate(BaseModel):
    entries: list[PronunciationEntry]


class PronunciationEdit(BaseModel):
    phrase: str | None = None
    pronunciation: str | None = None


def _model_dir() -> Path:
    return Path(os.getenv("KOKORO_MODEL_DIR", "/models"))


def _model_path() -> Path:
    return _model_dir() / os.getenv("KOKORO_MODEL_FILENAME", "kokoro-v1.0.onnx")


def _voices_path() -> Path:
    return _model_dir() / os.getenv("KOKORO_VOICES_FILENAME", "voices-v1.0.bin")


def _pronunciations_path() -> Path:
    configured_path = os.getenv("KOKORO_PRONUNCIATIONS_PATH", "").strip()
    if configured_path:
        return Path(configured_path)

    return _model_dir() / "pronunciations.json"


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


def _load_pronunciations() -> dict[str, str]:
    pronunciation_path = _pronunciations_path()
    if not pronunciation_path.exists():
        return {}

    with pronunciation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("pronunciations file must contain a JSON object")

    pronunciations: dict[str, str] = {}
    for phrase, pronunciation in data.items():
        if not isinstance(phrase, str) or not isinstance(pronunciation, str):
            raise ValueError("pronunciations file must map strings to strings")
        cleaned_phrase = phrase.strip()
        cleaned_pronunciation = pronunciation.strip()
        if cleaned_phrase and cleaned_pronunciation:
            pronunciations[cleaned_phrase] = cleaned_pronunciation

    return pronunciations


def _save_pronunciations(pronunciations: dict[str, str]) -> dict[str, str]:
    cleaned = dict(sorted(pronunciations.items(), key=lambda item: item[0].lower()))
    pronunciation_path = _pronunciations_path()
    pronunciation_path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile("w", dir=pronunciation_path.parent, delete=False, encoding="utf-8") as tmp_file:
        json.dump(cleaned, tmp_file, indent=2, ensure_ascii=True, sort_keys=True)
        tmp_file.write("\n")
        temp_path = Path(tmp_file.name)

    temp_path.replace(pronunciation_path)
    return cleaned


def _normalize_phrase(value: str, *, field_name: str) -> str:
    cleaned_value = value.strip()
    if not cleaned_value:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    return cleaned_value


def _normalize_entry(entry: PronunciationEntry) -> tuple[str, str]:
    return (
        _normalize_phrase(entry.phrase, field_name="phrase"),
        _normalize_phrase(entry.pronunciation, field_name="pronunciation"),
    )


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    prefix = r"(?<!\w)" if phrase[:1].isalnum() else ""
    suffix = r"(?!\w)" if phrase[-1:].isalnum() else ""
    return re.compile(f"{prefix}{re.escape(phrase)}{suffix}")


def _apply_pronunciations(text: str) -> str:
    updated_text = text
    for phrase, pronunciation in sorted(_load_pronunciations().items(), key=lambda item: len(item[0]), reverse=True):
        updated_text = _phrase_pattern(phrase).sub(pronunciation, updated_text)
    return updated_text


def _list_pronunciations() -> list[dict[str, str]]:
    return [
        {"phrase": phrase, "pronunciation": pronunciation}
        for phrase, pronunciation in _load_pronunciations().items()
    ]


@lru_cache(maxsize=1)
def _nemo_normalizer_factory():
    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer
    except Exception:
        return None

    try:
        return Normalizer(input_case="cased", lang="en")
    except Exception:
        return None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_with_nemo(text: str) -> str | None:
    # Normalizer initialization and grammar loading can be expensive.
    with _NEMO_INIT_LOCK:
        normalizer = _nemo_normalizer_factory()
    if normalizer is None:
        return None

    punct_post_process = _bool_env("KOKORO_NEMO_PUNCT_POST_PROCESS", True)
    try:
        sentences = normalizer.split_text_into_sentences(text)
        if sentences:
            normalized_sentences = normalizer.normalize_list(
                sentences,
                punct_post_process=punct_post_process,
            )
            if normalized_sentences:
                return " ".join(segment.strip() for segment in normalized_sentences if segment.strip())

        normalized_text = normalizer.normalize(
            text,
            punct_post_process=punct_post_process,
        )
        if isinstance(normalized_text, str) and normalized_text.strip():
            return normalized_text
    except Exception:
        return None

    return None


def _normalize_text(text: str, *, lang: str) -> str:
    rewritten_text = _apply_pronunciations(text)
    if not lang.lower().startswith("en"):
        return rewritten_text

    backend = os.getenv("KOKORO_TEXT_NORMALIZATION_BACKEND", "auto").strip().lower() or "auto"
    if backend not in {"auto", "nemo", "regex", "off"}:
        backend = "auto"

    if backend in {"auto", "nemo"}:
        nemo_output = _normalize_with_nemo(rewritten_text)
        if nemo_output is not None:
            return nemo_output
        if backend == "nemo":
            raise HTTPException(
                status_code=500,
                detail="NeMo normalization requested but unavailable",
            )

    if backend == "off":
        return rewritten_text

    return _DECIMAL_PATTERN.sub(r"\1 point \2", rewritten_text)


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


@app.post("/pronunciations")
def upsert_pronunciation(entry: PronunciationEntry) -> dict[str, object]:
    phrase, pronunciation = _normalize_entry(entry)

    with _PRONUNCIATION_LOCK:
        pronunciations = _load_pronunciations()
        pronunciations[phrase] = pronunciation
        saved = _save_pronunciations(pronunciations)

    return {
        "status": "ok",
        "entry": {"phrase": phrase, "pronunciation": pronunciation},
        "count": len(saved),
    }


@app.post("/pronunciations/bulk")
def bulk_upsert_pronunciations(update: PronunciationBulkUpdate) -> dict[str, object]:
    if not update.entries:
        raise HTTPException(status_code=400, detail="entries are required")

    normalized_entries = [_normalize_entry(entry) for entry in update.entries]

    with _PRONUNCIATION_LOCK:
        pronunciations = _load_pronunciations()
        for phrase, pronunciation in normalized_entries:
            pronunciations[phrase] = pronunciation
        saved = _save_pronunciations(pronunciations)

    return {
        "status": "ok",
        "updated": len(normalized_entries),
        "count": len(saved),
        "entries": [
            {"phrase": phrase, "pronunciation": pronunciation}
            for phrase, pronunciation in normalized_entries
        ],
    }


@app.patch("/pronunciations/{phrase}")
def edit_pronunciation(phrase: str, update: PronunciationEdit) -> dict[str, object]:
    current_phrase = _normalize_phrase(phrase, field_name="phrase")
    if update.phrase is None and update.pronunciation is None:
        raise HTTPException(status_code=400, detail="phrase or pronunciation is required")

    with _PRONUNCIATION_LOCK:
        pronunciations = _load_pronunciations()
        current_pronunciation = pronunciations.get(current_phrase)
        if current_pronunciation is None:
            raise HTTPException(status_code=404, detail="pronunciation not found")

        next_phrase = _normalize_phrase(update.phrase, field_name="phrase") if update.phrase is not None else current_phrase
        next_pronunciation = (
            _normalize_phrase(update.pronunciation, field_name="pronunciation")
            if update.pronunciation is not None
            else current_pronunciation
        )

        if next_phrase != current_phrase:
            pronunciations.pop(current_phrase, None)
        pronunciations[next_phrase] = next_pronunciation
        saved = _save_pronunciations(pronunciations)

    return {
        "status": "ok",
        "entry": {"phrase": next_phrase, "pronunciation": next_pronunciation},
        "count": len(saved),
    }


@app.delete("/pronunciations/{phrase}")
def delete_pronunciation(phrase: str) -> dict[str, object]:
    current_phrase = _normalize_phrase(phrase, field_name="phrase")

    with _PRONUNCIATION_LOCK:
        pronunciations = _load_pronunciations()
        deleted_pronunciation = pronunciations.pop(current_phrase, None)
        if deleted_pronunciation is None:
            raise HTTPException(status_code=404, detail="pronunciation not found")
        saved = _save_pronunciations(pronunciations)

    return {
        "status": "ok",
        "deleted": {"phrase": current_phrase, "pronunciation": deleted_pronunciation},
        "count": len(saved),
    }


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
    normalized_text = _normalize_text(cleaned_text, lang=resolved_lang)

    try:
        audio_bytes = _render_wav_bytes(
            normalized_text,
            voice=resolved_voice,
            lang=resolved_lang,
            speed=resolved_speed,
        )
    except AssertionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"kokoro synthesis failed: {exc}") from exc

    return Response(content=audio_bytes, media_type="audio/wav")