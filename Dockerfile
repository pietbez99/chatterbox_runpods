FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2204

WORKDIR /app

# Install system dependencies (ffmpeg for mp3 conversion)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install RunPod SDK and HTTP client
RUN pip install --no-cache-dir runpod requests pydub

# Install Chatterbox TTS and all its dependencies
RUN pip install --no-cache-dir git+https://github.com/pietbez99/chatterbox_runpods.git

# Pre-download model weights during build (avoids cold-start download)
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu'); print('Model weights cached successfully')"

# Bake voice reference into the image (downloaded once at build time)
RUN mkdir -p /app/voice && \
    curl -o /app/voice/reference.mp3 "https://mybigtooth.com/assets/voices/new-voice.mp3"

# Download handler.py directly from the repo (avoids COPY build context issues)
RUN curl -o /app/handler.py https://raw.githubusercontent.com/pietbez99/chatterbox_runpods/master/handler.py

CMD ["python", "-u", "/app/handler.py"]
