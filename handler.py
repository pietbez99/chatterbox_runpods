"""
RunPod Serverless Handler for Chatterbox TTS

Accepts:
  - text (str): Text to synthesize
  - audio_url (str, optional): URL to voice reference audio for cloning
  - exaggeration (float, optional): Voice exaggeration factor (default 0.5)
  - temperature (float, optional): Generation temperature (default 0.8)
  - cfg (float, optional): CFG weight (default 0.5)
  - output_format (str, optional): "mp3" or "wav" (default "wav")

Performance: Voice conditionals are cached after first download so subsequent
requests skip the expensive audio processing step.
"""

import os
import io
import base64
import tempfile
import requests
import runpod
import torch
import torchaudio as ta
from pydub import AudioSegment

# Global model instance (loaded once on cold start)
MODEL = None

# Cache: voice URL -> local file path (persists across requests on same worker)
VOICE_CACHE = {}


def load_model():
    """Load ChatterboxTTS model once on cold start."""
    global MODEL
    if MODEL is not None:
        return MODEL

    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Chatterbox] Loading model on device: {device}")
    MODEL = ChatterboxTTS.from_pretrained(device=device)
    print("[Chatterbox] Model loaded successfully")
    return MODEL


def get_voice_ref(url: str, model, exaggeration: float) -> None:
    """
    Download voice reference and prepare conditionals, with caching.
    If the same URL was already processed, the model's self.conds are
    already set and we skip the expensive prepare_conditionals() call.
    """
    if url in VOICE_CACHE:
        # Voice already downloaded and conditionals prepared
        # The model retains self.conds from the previous call
        # Just check if exaggeration changed (generate() handles this)
        print(f"[Chatterbox] Using cached voice for: {url}")
        return

    print(f"[Chatterbox] Downloading voice reference from: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Determine file extension
    content_type = response.headers.get("content-type", "")
    if "mp3" in content_type or url.lower().split("?")[0].endswith(".mp3"):
        ext = ".mp3"
    elif "wav" in content_type or url.lower().split("?")[0].endswith(".wav"):
        ext = ".wav"
    else:
        ext = ".mp3"

    # Save to a persistent file (not auto-deleted)
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(response.content)
    tmp.close()
    voice_path = tmp.name
    print(f"[Chatterbox] Voice reference saved to: {voice_path} ({len(response.content)} bytes)")

    # Prepare conditionals (the expensive step - embeds the voice)
    print(f"[Chatterbox] Preparing voice conditionals (one-time)...")
    model.prepare_conditionals(voice_path, exaggeration=exaggeration)
    print(f"[Chatterbox] Voice conditionals ready")

    # Cache for future requests
    VOICE_CACHE[url] = voice_path


def handler(job):
    """RunPod serverless handler."""
    job_input = job["input"]

    text = job_input.get("text", "")
    if not text:
        return {"error": "No text provided"}

    audio_url = job_input.get("audio_url")
    exaggeration = float(job_input.get("exaggeration", 0.5))
    temperature = float(job_input.get("temperature", 0.8))
    cfg_weight = float(job_input.get("cfg", 0.5))
    output_format = job_input.get("output_format", "wav").lower()

    print(f"[Chatterbox] Generating TTS:")
    print(f"  text: {text[:100]}...")
    print(f"  audio_url: {audio_url or 'NONE (using default voice)'}")
    print(f"  exaggeration: {exaggeration}")
    print(f"  temperature: {temperature}")
    print(f"  cfg: {cfg_weight}")
    print(f"  output_format: {output_format}")
    print(f"  voice_cached: {audio_url in VOICE_CACHE if audio_url else 'N/A'}")

    model = load_model()

    try:
        # Prepare voice conditionals (cached after first call)
        if audio_url:
            get_voice_ref(audio_url, model, exaggeration)
            # Generate WITHOUT audio_prompt_path - reuses cached self.conds
            wav = model.generate(
                text,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
        else:
            wav = model.generate(
                text,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )

        # Convert to output format
        wav_buffer = io.BytesIO()
        ta.save(wav_buffer, wav, model.sr, format="wav")
        wav_buffer.seek(0)

        if output_format == "mp3":
            audio_segment = AudioSegment.from_wav(wav_buffer)
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3", bitrate="192k")
            mp3_buffer.seek(0)
            audio_base64 = base64.b64encode(mp3_buffer.read()).decode("utf-8")
        else:
            audio_base64 = base64.b64encode(wav_buffer.read()).decode("utf-8")

        print(f"[Chatterbox] Audio generated successfully ({len(audio_base64)} bytes base64)")

        return {
            "audio_base64": audio_base64,
            "format": output_format,
            "sample_rate": model.sr,
        }

    except Exception as e:
        print(f"[Chatterbox] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
