"""
AST Tool - Pure asyncio, single library for all WebSocket connections
"""

import asyncio
import base64
import json
import os
import requests
import websockets

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

    dg_url = (
        f"wss://api.deepgram.com/v1/listen"
        f"?language={source_lang}&model=nova-2&punctuate=true"
        f"&interim_results=true&endpointing=500"
    )

    print("[WS] Connecting to Deepgram...")
    try:
        async with websockets.connect(
            dg_url,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            ping_interval=20,
            ping_timeout=30,
        ) as dg_ws:
            print("[DG] Connected to Deepgram")
            await client_ws.send(json.dumps({"status": "ready"}))
            print("[WS] Sent ready to client")

            async def receive_from_deepgram():
                async for message in dg_ws:
                    try:
                        data = json.loads(message)
                        if data.get("type") != "Results":
                            continue
                        alt        = data["channel"]["alternatives"][0]
                        transcript = alt.get("transcript", "").strip()
                        is_final   = data.get("is_final", False)
                        if not transcript:
                            continue

                        if not is_final:
                            await client_ws.send(json.dumps({
                                "transcript": transcript,
                                "is_final":   False,
                            }))
                        else:
                            print(f"[ASR] {transcript}")
                            # Run blocking calls in executor
                            loop = asyncio.get_event_loop()
                            translation = await loop.run_in_executor(
                                None, translate_with_claude, transcript, source_lang, target_lang
                            )
                            if not translation:
                                await client_ws.send(json.dumps({
                                    "transcript": transcript,
                                    "is_final":   True,
                                }))
                                continue
                            print(f"[MT] {translation}")
                            audio_b64 = await loop.run_in_executor(
                                None, synthesise_speech, translation, target_lang
                            )
                            payload = {
                                "transcript":  transcript,
                                "translation": translation,
                                "is_final":    True,
                            }
                            if audio_b64:
                                payload["audio_b64"]  = audio_b64
                                payload["audio_type"] = "audio/mpeg"
                            await client_ws.send(json.dumps(payload))
                    except Exception as e:
                        print(f"[DG recv] {e}")

            async def receive_from_client():
                async for message in client_ws:
                    if isinstance(message, bytes):
                        await dg_ws.send(message)
                    else:
                        try:
                            data = json.loads(message)
                            if data.get("type") == "close":
                                return
                        except:
                            pass

            # Run both directions concurrently
            await asyncio.gather(
                receive_from_deepgram(),
                receive_from_client(),
                return_exceptions=True,
            )

    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        print("[WS] Client disconnected")


async def main():
    port = int(os.environ.get("PORT", 5000))
    print(f"[WS] Starting on port {port}")
    async with websockets.serve(
        handle_client, "0.0.0.0", port,
        ping_interval=20,
        ping_timeout=60,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
