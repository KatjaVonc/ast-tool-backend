from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
import os
import json
import threading
import queue
import requests
import base64

app = Flask(__name__)
CORS(app)
sock = Sock(app)

DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY', '')
ANTHROPIC_API_KEY = (
    os.environ.get('ANTHROPIC_API_KEY') or
    os.environ.get('CLAUDE_APY_KEY') or
    os.environ.get('CLAUDE_API_KEY', '')
)

LANGUAGE_NAMES = {"de": "German", "it": "Italian"}
DEEPGRAM_VOICE  = {"it": "aura-arcas-en", "de": "aura-arcas-en"}

@app.route('/')
def home():
    return {
        'status': 'ok',
        'name': 'AST Tool Backend',
        'deepgram_configured': bool(DEEPGRAM_API_KEY),
        'anthropic_configured': bool(ANTHROPIC_API_KEY)
    }

@app.route('/health')
def health():
    return {'status': 'healthy'}


def translate_with_claude(text, source_lang, target_lang):
    try:
        resp = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key':         ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01',
                'content-type':      'application/json',
            },
            json={
                'model':      'claude-haiku-4-5-20251001',
                'max_tokens': 512,
                'messages': [{
                    'role':    'user',
                    'content': (
                        f"Translate this {LANGUAGE_NAMES.get(source_lang, source_lang)} "
                        f"speech segment into {LANGUAGE_NAMES.get(target_lang, target_lang)}. "
                        f"Preserve register and tone. Return ONLY the translation.\n\n{text}"
                    )
                }]
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()['content'][0]['text'].strip()
        print(f"[MT] Error {resp.status_code}: {resp.text}", flush=True)
    except Exception as e:
        print(f"[MT] Exception: {e}", flush=True)
    return None


def synthesise_speech(text, target_lang):
    try:
        voice = DEEPGRAM_VOICE.get(target_lang, 'aura-2-andromeda-it')
        print(f"[TTS] Calling Deepgram voice={voice} text_len={len(text)}", flush=True)
        resp = requests.post(
            f'https://api.deepgram.com/v1/speak?model={voice}',
            headers={
                'Authorization': f'Token {DEEPGRAM_API_KEY}',
                'Content-Type':  'application/json',
            },
            json={'text': text},
            timeout=15,
        )
        print(f"[TTS] Response status: {resp.status_code}", flush=True)
        if resp.status_code == 200:
            print(f"[TTS] Success, audio bytes: {len(resp.content)}", flush=True)
            return base64.b64encode(resp.content).decode('utf-8')
        print(f"[TTS] Error {resp.status_code}: {resp.text}", flush=True)
    except Exception as e:
        print(f"[TTS] Exception: {e}", flush=True)
    return None


@sock.route('/ws')
def websocket_endpoint(ws):
    print("Client connected", flush=True)

    try:
        config_msg  = ws.receive()
        config      = json.loads(config_msg)
        source_lang = config.get('source_lang', 'de')
        target_lang = config.get('target_lang', 'it')
        print(f"[WS] {source_lang} → {target_lang}", flush=True)
        ws.send(json.dumps({'status': 'ready'}))

        audio_queue = queue.Queue(maxsize=100)
        stop_flag   = threading.Event()

        def receive_audio():
            while not stop_flag.is_set():
                try:
                    msg = ws.receive(timeout=0.1)
                    if msg:
                        if isinstance(msg, bytes):
                            try:
                                audio_queue.put(msg, timeout=0.1)
                            except queue.Full:
                                pass
                        elif isinstance(msg, str):
                            data = json.loads(msg)
                            if data.get('type') == 'close':
                                stop_flag.set()
                                break
                except:
                    continue

        def process_deepgram():
            import websockets
            import asyncio

            async def stream():
                dg_url = (
                    f"wss://api.deepgram.com/v1/listen"
                    f"?model=nova-2"
                    f"&language={source_lang}"
                    f"&punctuate=true"
                    f"&interim_results=true"
                    f"&endpointing=500"
                    f"&encoding=linear16"
                    f"&sample_rate=16000"
                )
                headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
                print("Connecting to Deepgram...", flush=True)

                try:
                    async with websockets.connect(
                        dg_url,
                        extra_headers=headers,
                        ping_interval=5,
                        ping_timeout=10
                    ) as dg_ws:
                        print("Connected to Deepgram", flush=True)

                        async def send_audio():
                            try:
                                while not stop_flag.is_set():
                                    try:
                                        audio_data = audio_queue.get(timeout=0.1)
                                        await dg_ws.send(audio_data)
                                    except queue.Empty:
                                        await asyncio.sleep(0.01)
                            except Exception as e:
                                print(f"Send error: {e}", flush=True)

                        async def receive_transcription():
                            try:
                                async for msg in dg_ws:
                                    data = json.loads(msg)
                                    if 'channel' not in data:
                                        continue
                                    alts = data['channel'].get('alternatives', [])
                                    if not alts:
                                        continue
                                    transcript = alts[0].get('transcript', '').strip()
                                    is_final   = data.get('is_final', False)
                                    if not transcript:
                                        continue

                                    if not is_final:
                                        ws.send(json.dumps({
                                            'transcript': transcript,
                                            'is_final':   False,
                                        }))
                                    else:
                                        print(f"[ASR] {transcript}", flush=True)
                                        translation = translate_with_claude(transcript, source_lang, target_lang)
                                        if not translation:
                                            ws.send(json.dumps({'transcript': transcript, 'is_final': True}))
                                            continue
                                        print(f"[MT] {translation}", flush=True)
                                        audio_b64 = synthesise_speech(translation, target_lang)
                                        payload = {
                                            'transcript':  transcript,
                                            'translation': translation,
                                            'is_final':    True,
                                        }
                                        if audio_b64:
                                            payload['audio_b64']  = audio_b64
                                            payload['audio_type'] = 'audio/mpeg'
                                        ws.send(json.dumps(payload))

                            except Exception as e:
                                print(f"Receive error: {e}", flush=True)

                        await asyncio.gather(
                            send_audio(),
                            receive_transcription(),
                            return_exceptions=True
                        )

                except Exception as e:
                    print(f"Deepgram connection error: {e}", flush=True)

            asyncio.run(stream())

        audio_thread    = threading.Thread(target=receive_audio,    daemon=True)
        deepgram_thread = threading.Thread(target=process_deepgram, daemon=True)

        audio_thread.start()
        deepgram_thread.start()

        deepgram_thread.join()
        stop_flag.set()
        audio_thread.join(timeout=2)

    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        print("Client disconnected", flush=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting AST Tool backend on port {port}", flush=True)
    app.run(host='0.0.0.0', port=port)
