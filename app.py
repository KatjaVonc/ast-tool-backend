"""
AST Tool Backend
Pipeline: Audio → Deepgram (ASR) → Claude API direct (MT) → Deepgram (TTS) → Audio
"""

import os
import base64
import tempfile
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_APY_KEY") or os.environ.get("CLAUDE_API_KEY")
DEEPGRAM_API_KEY  = os.environ.get("DEEPGRAM_API_KEY")

LANGUAGE_NAMES = {"de": "German", "it": "Italian"}
DEEPGRAM_LANG  = {"de": "de", "it": "it"}
DEEPGRAM_VOICE = {"it": "aura-2-andromeda-it", "de": "aura-2-thalia-de"}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/translate-speech", methods=["POST"])
def translate_speech():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file  = request.files["audio"]
    source_lang = request.form.get("source_lang", "de")
    target_lang = request.form.get("target_lang", "it")

    suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── Step 1: Deepgram ASR ──────────────────────────────────────────────
        print(f"[ASR] Transcribing {source_lang}...")
        dg_lang = DEEPGRAM_LANG.get(source_lang, source_lang)

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        asr_resp = requests.post(
            f"https://api.deepgram.com/v1/listen?language={dg_lang}&model=nova-2&punctuate=true",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type":  "audio/mpeg",
            },
            data=audio_bytes,
            timeout=60,
        )

        if asr_resp.status_code != 200:
            return jsonify({"error": f"ASR failed: {asr_resp.status_code}"}), 500

        transcript = (
            asr_resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        )
        print(f"[ASR] Done: {transcript[:80]}...")

        if not transcript:
            return jsonify({"error": "Empty transcript"}), 422

        # ── Step 2: Claude MT (raw HTTP, no SDK) ──────────────────────────────
        print(f"[MT] Translating {source_lang} → {target_lang}...")
        src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        mt_resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role":    "user",
                        "content": (
                            f"Translate the following {src_name} speech excerpt into {tgt_name}. "
                            f"Preserve the register, tone, and rhetorical features. "
                            f"Return ONLY the translation, no notes.\n\n{transcript}"
                        )
                    }
                ]
            },
            timeout=30,
        )

        if mt_resp.status_code != 200:
            print(f"[MT] Claude error: {mt_resp.status_code} {mt_resp.text}")
            return jsonify({"error": f"MT failed: {mt_resp.status_code} — {mt_resp.text}"}), 500

        translation = mt_resp.json()["content"][0]["text"].strip()
        print(f"[MT] Done: {translation[:80]}...")

        # ── Step 3: Deepgram TTS ──────────────────────────────────────────────
        print(f"[TTS] Synthesising {target_lang}...")
        voice = DEEPGRAM_VOICE.get(target_lang, "aura-2-andromeda-it")

        tts_resp = requests.post(
            f"https://api.deepgram.com/v1/speak?model={voice}",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={"text": translation},
            timeout=30,
        )

        if tts_resp.status_code != 200:
            return jsonify({
                "transcript":  transcript,
                "translation": translation,
                "error_tts":   f"TTS failed: {tts_resp.status_code}",
            }), 200

        audio_b64 = base64.b64encode(tts_resp.content).decode("utf-8")
        print("[TTS] Done.")

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
