FROM python:3.12-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng gcc g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY .services/kokoro-tts/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY .services/kokoro-tts/app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# test layer — builds a test image for Kokoro TTS text-normalisation tests.
#
# Extends the pre-built production image (hermes-agent-kokoro-tts) so that the
# ~3 GB PyTorch/NeMo install is reused from cache.
#
# Build context: hermes-agent root directory.
#
# Usage (from hermes-agent root):
#   docker compose -f docker-compose.kokoro-test.yml run --rm kokoro-tts-test
#
# No TTS model files are downloaded or required; only NeMo text-processing is
# exercised, which is already bundled in the production image.

FROM production AS test

ENV KOKORO_TEXT_NORMALIZATION_BACKEND=auto \
    KOKORO_MODEL_DIR=/tmp/kokoro-test-models \
    KOKORO_PRONUNCIATIONS_PATH=/tmp/kokoro-test-pronunciations.json

# Add pytest (not present in the production image)
RUN pip install --no-cache-dir pytest==8.3.5

# Copy the test suite into the image
COPY .services/kokoro-tts/tests /app/tests

CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-s"]
