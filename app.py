"""
AST Tool - Pure asyncio WebSocket server
"""

import asyncio
import base64
import json
import os
import threading
import websockets
import websocket as ws_sync
import requests

ANTHROPIC_API_KEY = (
    os.environ.get("ANTHROPIC_API_KEY") or
    os.environ.get("CLAUDE_APY_KEY") or
    os.environ.get("CLAUDE_API_KEY")
)
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

LANGUAGE_NAMES = {"de": "German", "it": "Italian"}
DEEPGRAM_VOICE  = {"it": "aura-2-andromeda-it", "de": "aura-2-thalia-de"}


def translate_with_claude(text, source_lang, target_lang):
    try:
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
                        f"Translate this {LANGUAGE_NAMES.get(source_lang, source_lang)} "
                        f"speech segment into {LANGUAGE_NAMES.get(target_lang, target_lang)}. "
                        f"Preserve register and tone. Return ONLY the translation.\n\n{text}"
                    )
                }]
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"].strip()
        print(f"[MT] Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[MT] Exception: {e}")
    return None


def synthesise_speech(text, target_lang):
    try:
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
        print(f"[TTS] Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[TTS] Exception: {e}")
    return None


async def handle_client(client_ws):
    print("[WS] Client connected")

    # 1. Config
    try:
        config      = json.loads(await client_ws.recv())
        source_lang = config.get("source_lang", "de")
        target_lang = config.get("target_lang", "it")
        print(f"[WS] {source_lang} → {target_lang}")
    except Exception as e:
        print(f"[WS] Config error: {e}")
        return

    loop      = asyncio.get_event_loop()
    dg_ready  = threading.Event()   # fires when Deepgram confirms open
    dg_holder = [None]
    closed    = [False]

    dg_url = (
        f"wss://api.deepgram.com/v1/listen"
        f"?language={source_lang}&model=nova-2&punctuate=true"
        f"&interim_results=true&endpointing=500"
    )

    def on_dg_open(dg):
        print("[DG] Connected")
        dg_ready.set()   # signal that Deepgram is ready

    def on_dg_message(dg, message):
        if closed[0]:
            return
        try:
            data       = json.loads(message)
            if data.get("type") != "Results":
                return
            alt        = data["channel"]["alternatives"][0]
            transcript = alt.get("transcript", "").strip()
            is_final   = data.get("is_final", False)
            if not transcript:
                return

            if not is_final:
                asyncio.run_coroutine_threadsafe(
                    client_ws.send(json.dumps({"transcript": transcript, "is_final": False})),
                    loop
                )
            else:
                print(f"[ASR] {transcript}")
                translation = translate_with_claude(transcript, source_lang, target_lang)
                if not translation:
                    asyncio.run_coroutine_threadsafe(
                        client_ws.send(json.dumps({"transcript": transcript, "is_final": True})),
                        loop
                    )
                    return
                print(f"[MT] {translation}")
                audio_b64 = synthesise_speech(translation, target_lang)
                payload = {"transcript": transcript, "translation": translation, "is_final": True}
                if audio_b64:
                    payload["audio_b64"]  = audio_b64
                    payload["audio_type"] = "audio/mpeg"
                asyncio.run_coroutine_threadsafe(
                    client_ws.send(json.dumps(payload)),
                    loop
                )
        except Exception as e:
            print(f"[DG msg] {e}")

    def on_dg_error(dg, e):
        print(f"[DG] Error: {e}")
        dg_ready.set()   # unblock even on error

    def on_dg_close(dg, c, m):
        print("[DG] Closed")
        dg_ready.set()   # unblock if closed before open

    dg_ws = ws_sync.WebSocketApp(
        dg_url,
        header={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
        on_open=on_dg_open,
        on_message=on_dg_message,
        on_error=on_dg_error,
        on_close=on_dg_close,
    )
    dg_holder[0] = dg_ws

    dg_thread = threading.Thread(
        target=dg_ws.run_forever,
        kwargs={"ping_interval": 20},
        daemon=True
    )
    dg_thread.start()

    # Wait for Deepgram to confirm open (max 5s)
    await loop.run_in_executor(None, lambda: dg_ready.wait(timeout=5))

    if not dg_ws.sock or not dg_ws.sock.connected:
        print("[WS] Deepgram failed to connect")
        await client_ws.send(json.dumps({"error": "Could not connect to ASR service"}))
        return

    # Now tell the client we're ready
    await client_ws.send(json.dumps({"status": "ready"}))
    print("[WS] Sent ready to client")

    # 3. Forward audio client → Deepgram
    try:
        async for message in client_ws:
            if isinstance(message, bytes):
                try:
                    if dg_ws.sock and dg_ws.sock.connected:
                        dg_ws.sock.send_binary(message)
                except Exception as e:
                    print(f"[FWD] Error: {e}")
            else:
                try:
                    data = json.loads(message)
                    if data.get("type") == "close":
                        break
                except:
                    pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        closed[0] = True
        dg_ws.close()
        print("[WS] Client disconnected")


async def main():
    port = int(os.environ.get("PORT", 5000))
    print(f"[WS] Server starting on port {port}")
    async with websockets.serve(
        handle_client, "0.0.0.0", port,
        ping_interval=20, ping_timeout=60
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
