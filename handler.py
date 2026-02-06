"""
RunPod Serverless Handler for Chatterbox TTS

Voice reference is baked into the Docker image at build time.
The model loads with pre-prepared voice conditionals so every
request is just: text in → audio out.

Accepts:
  - text (str): Text to synthesize
  - exaggeration (float, optional): Voice exaggeration factor (default 0.3)
  - temperature (float, optional): Generation temperature (default 0.7)
  - cfg (float, optional): CFG weight (default 0.5)
  - output_format (str, optional): "mp3" or "wav" (default "wav")
"""

import io
import base64
import runpod
import torch
import torchaudio as ta
from pydub import AudioSegment

# Global model instance (loaded once on cold start, with voice pre-baked)
MODEL = None
VOICE_REF_PATH = "/app/voice/reference.mp3"


def load_model():
    """Load ChatterboxTTS model and prepare voice conditionals once on cold start."""
    global MODEL
    if MODEL is not None:
        return MODEL

    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Chatterbox] Loading model on device: {device}")
    MODEL = ChatterboxTTS.from_pretrained(device=device)
    print("[Chatterbox] Model loaded successfully")

    # Prepare voice conditionals from baked-in reference audio
    print(f"[Chatterbox] Preparing voice conditionals from: {VOICE_REF_PATH}")
    MODEL.prepare_conditionals(VOICE_REF_PATH, exaggeration=0.3)
    print("[Chatterbox] Voice conditionals ready - all requests will use this voice")

    return MODEL


def handler(job):
    """RunPod serverless handler."""
    job_input = job["input"]

    text = job_input.get("text", "")
    if not text:
        return {"error": "No text provided"}

    exaggeration = float(job_input.get("exaggeration", 0.3))
    temperature = float(job_input.get("temperature", 0.7))
    cfg_weight = float(job_input.get("cfg", 0.5))
    output_format = job_input.get("output_format", "wav").lower()

    print(f"[Chatterbox] Generating TTS:")
    print(f"  text: {text[:100]}...")
    print(f"  exaggeration: {exaggeration}")
    print(f"  temperature: {temperature}")
    print(f"  cfg: {cfg_weight}")
    print(f"  output_format: {output_format}")

    model = load_model()

    try:
        # Generate using pre-baked voice conditionals
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
