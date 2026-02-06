FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies (ffmpeg for mp3 conversion)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install RunPod SDK and HTTP client
RUN pip install --no-cache-dir runpod requests pydub

# Install Chatterbox TTS and all its dependencies from our fork
RUN pip install --no-cache-dir git+https://github.com/pietbez99/chatterbox_runpods.git

# Pre-download model weights during build (avoids cold-start download)
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu'); print('Model weights cached successfully')"

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
