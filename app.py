"""
AST Tool Backend - Live Streaming
Single-port: uses gevent-websocket to serve both HTTP and WS on $PORT
Pipeline: WebSocket audio → Deepgram streaming ASR → Claude MT → Deepgram TTS → audio back
"""

import os
import base64
import json
import re
import requests
import asyncio
import threading
from flask import Flask, jsonify, request
from flask_cors import CORS
from geventwebsocket import WebSocketError
from geventwebsocket.handler import WebSocketHandler
from gevent import pywsgi
import gevent

app = Flask(__name__)
CORS(app)

ANTHROPIC_API_KEY = (
    os.environ.get("ANTHROPIC_API_KEY") or
    os.environ.get("CLAUDE_APY_KEY") or
    os.environ.get("CLAUDE_API_KEY")
)
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

LANGUAGE_NAMES = {"de": "German", "it": "Italian"}
DEEPGRAM_VOICE  = {"it": "aura-2-andromeda-it", "de": "aura-2-thalia-de"}

# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ── MT via Claude ──────────────────────────────────────────────────────────────
def translate_with_claude(text, source_lang, target_lang):
    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":      "claude-haiku-4-5-20251001",
            "max_tokens": 512,
            "messages": [{
                "role":    "user",
                "content": (
                    f"Translate this {src_name} speech segment into {tgt_name}. "
                    f"Preserve register and tone. Return ONLY the translation.\n\n{text}"
                )
            }]
        },
        timeout=10,
    )
    if resp.status_code == 200:
        return resp.json()["content"][0]["text"].strip()
    print(f"[MT] Claude error {resp.status_code}: {resp.text}")
    return None

# ── TTS via Deepgram ───────────────────────────────────────────────────────────
def synthesise_speech(text, target_lang):
    voice = DEEPGRAM_VOICE.get(target_lang, "aura-2-andromeda-it")
    resp = requests.post(
        f"https://api.deepgram.com/v1/speak?model={voice}",
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={"text": text},
        timeout=15,
    )
    if resp.status_code == 200:
        return base64.b64encode(resp.content).decode("utf-8")
    print(f"[TTS] Deepgram error {resp.status_code}: {resp.text}")
    return None

# ── WebSocket route ────────────────────────────────────────────────────────────
@app.route("/ws")
def ws_handler():
    ws = request.environ.get("wsgi.websocket")
    if not ws:
        return "WebSocket required", 400

    print("[WS] Client connected")

    # 1. Receive config
    try:
        config     = json.loads(ws.receive())
        source_lang = config.get("source_lang", "de")
        target_lang = config.get("target_lang", "it")
        print(f"[WS] {source_lang} → {target_lang}")
    except Exception as e:
        print(f"[WS] Config error: {e}")
        return ""

    ws.send(json.dumps({"status": "ready"}))

    # 2. Open Deepgram streaming connection (in a thread since it's async)
    import websocket as ws_sync  # websocket-client for sync use

    dg_url = (
        f"wss://api.deepgram.com/v1/listen"
        f"?language={source_lang}&model=nova-2&punctuate=true"
        f"&interim_results=true&endpointing=500"
    )

    audio_queue_local = []
    dg_ws_holder = [None]
    done = [False]

    def on_dg_message(dg, message):
        try:
            data = json.loads(message)
            if data.get("type") != "Results":
                return
            alt        = data["channel"]["alternatives"][0]
            transcript = alt.get("transcript", "").strip()
            is_final   = data.get("is_final", False)
            if not transcript:
                return
            if not is_final:
                ws.send(json.dumps({"transcript": transcript, "is_final": False}))
            else:
                print(f"[ASR] {transcript}")
                translation = translate_with_claude(transcript, source_lang, target_lang)
                if not translation:
                    ws.send(json.dumps({"transcript": transcript, "is_final": True}))
                    return
                print(f"[MT] {translation}")
                audio_b64 = synthesise_speech(translation, target_lang)
                payload = {"transcript": transcript, "translation": translation, "is_final": True}
                if audio_b64:
                    payload["audio_b64"]  = audio_b64
                    payload["audio_type"] = "audio/mpeg"
                ws.send(json.dumps(payload))
        except Exception as e:
            print(f"[DG msg error] {e}")

    def on_dg_error(dg, error):
        print(f"[DG error] {error}")

    def on_dg_close(dg, code, msg):
        print("[DG] Connection closed")

    def on_dg_open(dg):
        print("[DG] Connected")

    dg_ws = ws_sync.WebSocketApp(
        dg_url,
        header={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
        on_message=on_dg_message,
        on_error=on_dg_error,
        on_close=on_dg_close,
        on_open=on_dg_open,
    )
    dg_ws_holder[0] = dg_ws

    dg_thread = threading.Thread(target=dg_ws.run_forever, daemon=True)
    dg_thread.start()

    # 3. Forward audio from client → Deepgram
    try:
        while True:
            message = ws.receive()
            if message is None:
                break
            if isinstance(message, bytes):
                if dg_ws_holder[0]:
                    dg_ws_holder[0].send(message, opcode=0x2)  # binary
            else:
                try:
                    data = json.loads(message)
                    if data.get("type") == "close":
                        break
                except:
                    pass
    except WebSocketError as e:
        print(f"[WS] Client disconnected: {e}")
    finally:
        if dg_ws_holder[0]:
            dg_ws_holder[0].close()
        print("[WS] Done")

    return ""

# ── Start server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting on port {port}")
    server = pywsgi.WSGIServer(("0.0.0.0", port), app, handler_class=WebSocketHandler)
    server.serve_forever()
