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
DEEPL_API_KEY    = os.environ.get('DEEPL_API_KEY', '')

DEEPGRAM_VOICE   = {"it": "aura-2-livia-it", "de": "aura-2-viktoria-de"}
ANTHROPIC_API_KEY = (
    os.environ.get("ANTHROPIC_API_KEY") or
    os.environ.get("CLAUDE_APY_KEY") or
    os.environ.get("CLAUDE_API_KEY", "")
)
AZURE_KEY         = os.environ.get("AZURE_TRANSLATOR_KEY", "")
AZURE_REGION      = os.environ.get("AZURE_REGION", "westeurope")
GOOGLE_KEY        = os.environ.get("GOOGLE_TRANSLATE_KEY", "")

@app.route('/')
def home():
    return {'status': 'ok', 'name': 'AST Tool Backend'}

@app.route('/health')
def health():
    return {'status': 'healthy'}


def translate(text, source_lang, target_lang, engine="deepl"):
    try:
        if engine == "deepl":
            resp = requests.post(
                'https://api-free.deepl.com/v2/translate',
                headers={'Authorization': f'DeepL-Auth-Key {DEEPL_API_KEY}', 'Content-Type': 'application/json'},
                json={'text': [text], 'source_lang': source_lang.upper(), 'target_lang': target_lang.upper()},
                timeout=5,
            )
            if resp.status_code == 200:
                result = resp.json()['translations'][0]['text']
                print(f"[MT/DeepL] {result[:80]}", flush=True)
                return result
            print(f"[MT/DeepL] Error {resp.status_code}: {resp.text}", flush=True)

        elif engine == "claude":
            from_name = {"de": "German", "it": "Italian"}.get(source_lang, source_lang)
            to_name   = {"de": "German", "it": "Italian"}.get(target_lang, target_lang)
            resp = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={'x-api-key': ANTHROPIC_API_KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'},
                json={
                    'model': 'claude-haiku-4-5-20251001',
                    'max_tokens': 512,
                    'messages': [{'role': 'user', 'content': f'Translate this {from_name} speech into {to_name}. Preserve register and tone. Return ONLY the translation.

{text}'}]
                },
                timeout=10,
            )
            if resp.status_code == 200:
                result = resp.json()['content'][0]['text'].strip()
                print(f"[MT/Claude] {result[:80]}", flush=True)
                return result
            print(f"[MT/Claude] Error {resp.status_code}", flush=True)

        elif engine == "google":
            resp = requests.post(
                f'https://translation.googleapis.com/language/translate/v2?key={GOOGLE_KEY}',
                json={'q': text, 'source': source_lang, 'target': target_lang, 'format': 'text'},
                timeout=5,
            )
            if resp.status_code == 200:
                result = resp.json()['data']['translations'][0]['translatedText']
                print(f"[MT/Google] {result[:80]}", flush=True)
                return result
            print(f"[MT/Google] Error {resp.status_code}: {resp.text}", flush=True)

        elif engine == "azure":
            resp = requests.post(
                f'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&from={source_lang}&to={target_lang}',
                headers={
                    'Ocp-Apim-Subscription-Key': AZURE_KEY,
                    'Ocp-Apim-Subscription-Region': AZURE_REGION,
                    'Content-Type': 'application/json',
                },
                json=[{'text': text}],
                timeout=5,
            )
            if resp.status_code == 200:
                result = resp.json()[0]['translations'][0]['text']
                print(f"[MT/Azure] {result[:80]}", flush=True)
                return result
            print(f"[MT/Azure] Error {resp.status_code}: {resp.text}", flush=True)

    except Exception as e:
        print(f"[MT/{engine}] Exception: {e}", flush=True)
    return None


def synthesise_streaming(text, target_lang, ws):
    """Full REST TTS — clean single MP3, no chunk artifacts."""
    try:
        voice = DEEPGRAM_VOICE.get(target_lang, 'aura-2-livia-it')
        print(f"[TTS] Calling {voice}...", flush=True)
        resp = requests.post(
            f'https://api.deepgram.com/v1/speak?model={voice}',
            headers={
                'Authorization': f'Token {DEEPGRAM_API_KEY}',
                'Content-Type':  'application/json',
            },
            json={'text': text},
            timeout=15,
        )
        if resp.status_code == 200:
            audio_b64 = base64.b64encode(resp.content).decode('utf-8')
            ws.send(json.dumps({
                'type':       'tts_chunk',
                'audio_b64':  audio_b64,
                'audio_type': 'audio/mpeg',
                'chunk_index': 0,
            }))
            ws.send(json.dumps({'type': 'tts_done'}))
            print(f"[TTS] Done {len(resp.content)} bytes", flush=True)
        else:
            print(f"[TTS] Error {resp.status_code}: {resp.text}", flush=True)
    except Exception as e:
        print(f"[TTS] Exception: {e}", flush=True)


@sock.route('/ws')
def websocket_endpoint(ws):
    print("Client connected", flush=True)

    try:
        config_msg  = ws.receive()
        config      = json.loads(config_msg)
        source_lang = config.get('source_lang', 'de')
        target_lang = config.get('target_lang', 'it')
        mt_engine   = config.get('mt_engine', 'deepl')
        print(f"[WS] {source_lang} → {target_lang} via {mt_engine}", flush=True)
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
                            try:
                                data = json.loads(msg)
                                if data.get('type') == 'close':
                                    stop_flag.set()
                                    break
                            except:
                                pass
                except:
                    continue

        def process_deepgram():
            import websockets
            import asyncio

            async def stream():
                dg_url = (
                    f"wss://api.deepgram.com/v1/listen"
                    f"?model=nova-3"
                    f"&language={source_lang}"
                    f"&smart_format=true"
                    f"&interim_results=true"
                    f"&endpointing=1000"
                    f"&encoding=linear16"
                    f"&sample_rate=16000"
                )
                headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
                print("Connecting to Deepgram ASR...", flush=True)

                try:
                    async with websockets.connect(
                        dg_url,
                        extra_headers=headers,
                        ping_interval=10,
                        ping_timeout=30
                    ) as dg_ws:
                        print("Connected to Deepgram ASR", flush=True)

                        async def send_audio():
                            try:
                                last_keepalive = asyncio.get_event_loop().time()
                                while not stop_flag.is_set():
                                    try:
                                        audio_data = audio_queue.get(timeout=0.1)
                                        await dg_ws.send(audio_data)
                                        last_keepalive = asyncio.get_event_loop().time()
                                    except queue.Empty:
                                        now = asyncio.get_event_loop().time()
                                        if now - last_keepalive > 5:
                                            await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                                            last_keepalive = now
                                        await asyncio.sleep(0.01)
                            except Exception as e:
                                print(f"Send error: {e}", flush=True)

                        async def receive_transcription():
                            try:
                                async for msg in dg_ws:
                                    data = json.loads(msg)
                                    # Skip non-dict messages (VAD events come as lists)
                                    if not isinstance(data, dict):
                                        continue
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
                                        translation = translate(transcript, source_lang, target_lang, mt_engine)
                                        if not translation:
                                            ws.send(json.dumps({'transcript': transcript, 'is_final': True}))
                                            continue

                                        # Send text immediately — don't wait for TTS
                                        ws.send(json.dumps({
                                            'transcript':  transcript,
                                            'translation': translation,
                                            'is_final':    True,
                                        }))

                                        # Stream TTS in a separate thread so ASR keeps running
                                        tts_thread = threading.Thread(
                                            target=synthesise_streaming,
                                            args=(translation, target_lang, ws),
                                            daemon=True
                                        )
                                        tts_thread.start()

                            except Exception as e:
                                print(f"Receive error: {e}", flush=True)

                        await asyncio.gather(
                            send_audio(),
                            receive_transcription(),
                            return_exceptions=True
                        )

                except Exception as e:
                    print(f"Deepgram ASR connection error: {e}", flush=True)

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
