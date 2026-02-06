"""
RunPod Serverless Handler for Chatterbox TTS

Accepts:
  - text (str): Text to synthesize
  - audio_url (str, optional): URL to voice reference audio for cloning
  - exaggeration (float, optional): Voice exaggeration factor (default 0.5)
  - temperature (float, optional): Generation temperature (default 0.8)
  - cfg (float, optional): CFG weight (default 0.5)
  - output_format (str, optional): "mp3" or "wav" (default "wav")
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


def download_audio_ref(url: str) -> str:
    """Download voice reference audio from URL to a temp file."""
    print(f"[Chatterbox] Downloading voice reference from: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Determine file extension from content-type or URL
    content_type = response.headers.get("content-type", "")
    if "mp3" in content_type or url.lower().split("?")[0].endswith(".mp3"):
        ext = ".mp3"
    elif "wav" in content_type or url.lower().split("?")[0].endswith(".wav"):
        ext = ".wav"
    else:
        ext = ".mp3"  # Default to mp3

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(response.content)
    tmp.close()
    print(f"[Chatterbox] Voice reference saved to: {tmp.name} ({len(response.content)} bytes)")
    return tmp.name


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

    model = load_model()
    audio_prompt_path = None

    try:
        # Download voice reference if provided
        if audio_url:
            audio_prompt_path = download_audio_ref(audio_url)

        # Generate audio
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )

        # Convert to output format
        # Save as WAV first
        wav_buffer = io.BytesIO()
        ta.save(wav_buffer, wav, model.sr, format="wav")
        wav_buffer.seek(0)

        if output_format == "mp3":
            # Convert WAV to MP3
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

    finally:
        # Clean up temp file
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.unlink(audio_prompt_path)
            print(f"[Chatterbox] Cleaned up temp file: {audio_prompt_path}")


runpod.serverless.start({"handler": handler})
