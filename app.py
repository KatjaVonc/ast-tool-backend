"""
AST Tool Backend
Pipeline: Audio → Whisper (ASR) → Claude (MT) → Deepgram (TTS) → Audio
"""

import os
import io
import json
import tempfile
import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import anthropic
import whisper

app = Flask(__name__)
CORS(app)

# ── API Keys (set as environment variables on Render) ──────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEEPGRAM_API_KEY  = os.environ.get("DEEPGRAM_API_KEY")

# ── Whisper model (loaded once at startup) ─────────────────────────────────────
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")   # "small" for better accuracy, costs ~1GB RAM
print("Whisper ready.")

# ── Language config ────────────────────────────────────────────────────────────
LANGUAGE_NAMES = {
    "de": "German",
    "it": "Italian",
}

DEEPGRAM_VOICE = {
    "it": "aura-2-andromeda-it",   # Italian female — change if you prefer another
    "de": "aura-2-thalia-de",      # German female
}

# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ── Main pipeline endpoint ─────────────────────────────────────────────────────
@app.route("/translate-speech", methods=["POST"])
def translate_speech():
    """
    Expects:
      - audio file in multipart form field "audio"
      - form field "source_lang" (e.g. "de")
      - form field "target_lang" (e.g. "it")
    Returns:
      JSON with keys: transcript, translation, audio_url
      (audio is returned as base64 in the same response for simplicity)
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file  = request.files["audio"]
    source_lang = request.form.get("source_lang", "de")
    target_lang = request.form.get("target_lang", "it")

    # ── Step 1: Save audio to temp file ───────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── Step 2: Whisper ASR ───────────────────────────────────────────────
        print(f"[ASR] Transcribing {source_lang} audio...")
        result = whisper_model.transcribe(tmp_path, language=source_lang)
        transcript = result["text"].strip()
        print(f"[ASR] Transcript: {transcript}")

        if not transcript:
            return jsonify({"error": "Could not transcribe audio"}), 422

        # ── Step 3: Claude MT ─────────────────────────────────────────────────
        print(f"[MT] Translating {source_lang} → {target_lang}...")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Translate the following {src_name} speech excerpt into {tgt_name}. "
                        f"Preserve the register, tone, and any rhetorical features of the original. "
                        f"Return ONLY the translation, no explanations or notes.\n\n"
                        f"{transcript}"
                    )
                }
            ]
        )
        translation = message.content[0].text.strip()
        print(f"[MT] Translation: {translation}")

        # ── Step 4: Deepgram TTS ──────────────────────────────────────────────
        print(f"[TTS] Synthesising {target_lang} speech...")
        voice = DEEPGRAM_VOICE.get(target_lang, "aura-2-andromeda-it")

        dg_response = requests.post(
            f"https://api.deepgram.com/v1/speak?model={voice}",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"text": translation},
            timeout=30,
        )

        if dg_response.status_code != 200:
            print(f"[TTS] Deepgram error: {dg_response.status_code} {dg_response.text}")
            return jsonify({
                "transcript":   transcript,
                "translation":  translation,
                "error_tts":    f"TTS failed: {dg_response.status_code}"
            }), 200   # still return text even if TTS fails

        # ── Step 5: Return everything ─────────────────────────────────────────
        import base64
        audio_b64 = base64.b64encode(dg_response.content).decode("utf-8")

        return jsonify({
            "transcript":  transcript,
            "translation": translation,
            "audio_b64":   audio_b64,
            "audio_type":  "audio/mpeg",
        })

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
