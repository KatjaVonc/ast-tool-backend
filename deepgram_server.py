from flask import Flask
from flask_cors import CORS
from flask_sock import Sock
import os
import json
import threading
import queue
import base64
import re
import asyncio
import httpx
from num2words import num2words

app = Flask(__name__)
CORS(app)
sock = Sock(app)

DEEPGRAM_API_KEY  = os.environ.get('DEEPGRAM_API_KEY', '')
DEEPL_API_KEY     = os.environ.get('DEEPL_API_KEY', '')
ANTHROPIC_API_KEY = (
    os.environ.get("ANTHROPIC_API_KEY") or
    os.environ.get("CLAUDE_APY_KEY") or
    os.environ.get("CLAUDE_API_KEY", "")
)
AZURE_KEY         = os.environ.get("AZURE_TRANSLATOR_KEY", "")
AZURE_SPEECH_KEY  = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_REGION      = os.environ.get("AZURE_REGION", "westeurope")
GOOGLE_KEY        = os.environ.get("GOOGLE_TRANSLATE_KEY", "")
GOOGLE_AI_KEY     = os.environ.get("GOOGLE_AI_KEY", "")

DEEPGRAM_VOICE   = {"it": "aura-2-livia-it",    "de": "aura-2-viktoria-de"}
GOOGLE_TTS_VOICE = {"it": "it-IT-Neural2-A",     "de": "de-DE-Neural2-F"}
AZURE_TTS_VOICE  = {"it": "it-IT-ElsaNeural",    "de": "de-DE-KatjaNeural"}


@app.route('/')
def home():
    return {'status': 'ok', 'name': 'AST Tool Backend'}

@app.route('/health')
def health():
    return {'status': 'healthy'}


# ── Helpers ───────────────────────────────────────────────────────────────────

def convert_numbers_to_words(text, lang):
    lang_code = {"it": "it", "de": "de"}.get(lang, "en")
    def replace_number(match):
        try:
            return num2words(int(match.group(0)), lang=lang_code)
        except Exception:
            return match.group(0)
    return re.sub(r'\d+', replace_number, text)


# ── Async MT ──────────────────────────────────────────────────────────────────

async def translate_async(client, text, source_lang, target_lang,
                          engine="deepl", context_brief="", formality="default"):
    """All MT engines via async httpx -- no thread blocking."""
    try:
        if engine == "deepl":
            resp = await client.post(
                'https://api-free.deepl.com/v2/translate',
                headers={'Authorization': f'DeepL-Auth-Key {DEEPL_API_KEY}',
                         'Content-Type': 'application/json'},
                json={'text': [text],
                      'source_lang': source_lang.upper(),
                      'target_lang': target_lang.upper(),
                      **(({'formality': formality}) if formality != 'default' else {})},
                timeout=8,
            )
            if resp.status_code == 200:
                result = resp.json()['translations'][0]['text']
                print(f"[MT/DeepL] {result[:80]}", flush=True)
                return result
            print(f"[MT/DeepL] Error {resp.status_code}", flush=True)

        elif engine == "claude":
            from_name = {"de": "German", "it": "Italian"}.get(source_lang, source_lang)
            to_name   = {"de": "German", "it": "Italian"}.get(target_lang, target_lang)
            system = (
                'You are a simultaneous interpreter output channel. '
                'Rules: Output ONLY the ' + to_name + ' translation of the '
                + from_name + ' input. '
                'No notes, no bold text, no commentary, no asterisks, no explanations, '
                'no apologies. Never mention that text is incomplete or has errors. '
                'Never add headers like "German translation:". '
                'Just translate and output the translation, nothing else, '
                'even if the segment is a fragment.'
                + (('\n\nSession context for translation decisions:\n' + context_brief)
                   if context_brief else '')
            )
            # Retry up to 3 times on 500/529
            for attempt in range(3):
                try:
                    resp = await client.post(
                        'https://api.anthropic.com/v1/messages',
                        headers={'x-api-key': ANTHROPIC_API_KEY,
                                 'anthropic-version': '2023-06-01',
                                 'content-type': 'application/json'},
                        json={'model': 'claude-haiku-4-5-20251001',
                              'max_tokens': 512,
                              'temperature': 0,
                              'system': system,
                              'messages': [{'role': 'user', 'content': text}]},
                        timeout=15,
                    )
                    if resp.status_code == 200:
                        result = resp.json()['content'][0]['text'].strip()
                        print(f"[MT/Claude] {result[:80]}", flush=True)
                        return result
                    elif resp.status_code in (500, 529) and attempt < 2:
                        wait = 2 ** attempt
                        print(f"[MT/Claude] Error {resp.status_code}, retry {attempt+1}/3 in {wait}s",
                              flush=True)
                        await asyncio.sleep(wait)
                    else:
                        print(f"[MT/Claude] Error {resp.status_code} (gave up after {attempt+1})",
                              flush=True)
                        break
                except httpx.TimeoutException:
                    if attempt < 2:
                        print(f"[MT/Claude] Timeout, retry {attempt+1}/3", flush=True)
                    else:
                        print("[MT/Claude] Timeout (gave up)", flush=True)
                        break

        elif engine == "google":
            resp = await client.post(
                f'https://translation.googleapis.com/language/translate/v2?key={GOOGLE_KEY}',
                json={'q': text, 'source': source_lang,
                      'target': target_lang, 'format': 'text'},
                timeout=8,
            )
            if resp.status_code == 200:
                result = resp.json()['data']['translations'][0]['translatedText']
                print(f"[MT/Google] {result[:80]}", flush=True)
                return result
            print(f"[MT/Google] Error {resp.status_code}", flush=True)

        elif engine == "azure":
            resp = await client.post(
                f'https://api.cognitive.microsofttranslator.com/translate'
                f'?api-version=3.0&from={source_lang}&to={target_lang}',
                headers={'Ocp-Apim-Subscription-Key': AZURE_KEY,
                         'Ocp-Apim-Subscription-Region': AZURE_REGION,
                         'Content-Type': 'application/json'},
                json=[{'text': text}],
                timeout=8,
            )
            if resp.status_code == 200:
                result = resp.json()[0]['translations'][0]['text']
                print(f"[MT/Azure] {result[:80]}", flush=True)
                return result
            print(f"[MT/Azure] Error {resp.status_code}", flush=True)

    except Exception as e:
        print(f"[MT/{engine}] Exception: {e}", flush=True)
    return None


# ── Async TTS ─────────────────────────────────────────────────────────────────

async def synthesise_async(client, text, target_lang, ws, tts_engine="deepgram"):
    """TTS via async httpx -- runs concurrently with next MT call."""
    try:
        text = convert_numbers_to_words(text, target_lang)
        audio_bytes = None

        if tts_engine == "deepgram":
            voice = DEEPGRAM_VOICE.get(target_lang, 'aura-2-livia-it')
            print(f"[TTS/Deepgram] {voice}...", flush=True)
            resp = await client.post(
                f'https://api.deepgram.com/v1/speak?model={voice}',
                headers={'Authorization': f'Token {DEEPGRAM_API_KEY}',
                         'Content-Type': 'application/json'},
                json={'text': text},
                timeout=15,
            )
            if resp.status_code == 200:
                audio_bytes = resp.content
            else:
                print(f"[TTS/Deepgram] Error {resp.status_code}", flush=True)

        elif tts_engine == "google":
            voice     = GOOGLE_TTS_VOICE.get(target_lang, 'it-IT-Neural2-A')
            lang_code = 'it-IT' if target_lang == 'it' else 'de-DE'
            print(f"[TTS/Google] {voice}...", flush=True)
            resp = await client.post(
                f'https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_KEY}',
                json={'input': {'text': text},
                      'voice': {'languageCode': lang_code, 'name': voice},
                      'audioConfig': {'audioEncoding': 'MP3', 'speakingRate': 1.0}},
                timeout=15,
            )
            if resp.status_code == 200:
                audio_bytes = base64.b64decode(resp.json()['audioContent'])
            else:
                print(f"[TTS/Google] Error {resp.status_code}", flush=True)

        elif tts_engine == "azure":
            voice     = AZURE_TTS_VOICE.get(target_lang, 'it-IT-ElsaNeural')
            lang_code = 'it-IT' if target_lang == 'it' else 'de-DE'
            print(f"[TTS/Azure] {voice}...", flush=True)
            token_resp = await client.post(
                f'https://{AZURE_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken',
                headers={'Ocp-Apim-Subscription-Key': AZURE_SPEECH_KEY},
                timeout=10,
            )
            if token_resp.status_code == 200:
                ssml = (f"<speak version='1.0' xml:lang='{lang_code}'>"
                        f"<voice name='{voice}'>{text}</voice></speak>")
                tts_resp = await client.post(
                    f'https://{AZURE_REGION}.tts.speech.microsoft.com/cognitiveservices/v1',
                    headers={'Authorization': f'Bearer {token_resp.text}',
                             'Content-Type': 'application/ssml+xml',
                             'X-Microsoft-OutputFormat': 'audio-24khz-48kbitrate-mono-mp3'},
                    content=ssml.encode('utf-8'),
                    timeout=15,
                )
                if tts_resp.status_code == 200:
                    audio_bytes = tts_resp.content
                else:
                    print(f"[TTS/Azure] Error {tts_resp.status_code}", flush=True)
            else:
                print(f"[TTS/Azure] Token error {token_resp.status_code}", flush=True)

        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            ws.send(json.dumps({'type': 'tts_chunk', 'audio_b64': audio_b64,
                                'audio_type': 'audio/mpeg', 'chunk_index': 0}))
            ws.send(json.dumps({'type': 'tts_done'}))
            print(f"[TTS] Done {len(audio_bytes)} bytes", flush=True)
        else:
            print("[TTS] No audio produced", flush=True)

    except Exception as e:
        print(f"[TTS] Exception: {e}", flush=True)


# ── Gemini Live path (unchanged) ──────────────────────────────────────────────

def handle_gemini_live(ws, source_lang, target_lang, api_key=""):
    import websockets

    print(f"[Gemini] Starting Live Translate {source_lang} → {target_lang}", flush=True)

    audio_queue_g = queue.Queue(maxsize=200)
    stop_flag_g   = threading.Event()

    def receive_audio_g():
        while not stop_flag_g.is_set():
            try:
                msg = ws.receive(timeout=0.1)
                if msg:
                    if isinstance(msg, bytes):
                        try:
                            audio_queue_g.put(msg, timeout=0.1)
                        except queue.Full:
                            pass
                    elif isinstance(msg, str):
                        try:
                            data = json.loads(msg)
                            if data.get('type') == 'close':
                                stop_flag_g.set()
                                break
                        except Exception:
                            pass
            except Exception:
                continue

    async def _gemini_stream():
        url = (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
            f"?key={api_key}"
        )
        setup_msg = {
            "setup": {
                "model": "models/gemini-3.5-live-translate-preview",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "translationConfig": {
                        "targetLanguageCode": target_lang,
                        "echoTargetLanguage": False
                    }
                },
                "inputAudioTranscription": {},
                "outputAudioTranscription": {}
            }
        }
        try:
            async with websockets.connect(url, ping_interval=20) as gemini_ws:
                print("[Gemini] Connected", flush=True)
                await gemini_ws.send(json.dumps(setup_msg))
                print("[Gemini] Setup sent", flush=True)

                async def send_audio():
                    try:
                        while not stop_flag_g.is_set():
                            try:
                                audio_data = audio_queue_g.get(timeout=0.1)
                                await gemini_ws.send(json.dumps({
                                    "realtimeInput": {
                                        "audio": {
                                            "data": base64.b64encode(audio_data).decode('utf-8'),
                                            "mimeType": "audio/pcm;rate=16000"
                                        }
                                    }
                                }))
                            except queue.Empty:
                                await asyncio.sleep(0.01)
                    except Exception as e:
                        print(f"[Gemini] Send error: {e}", flush=True)

                async def receive_output():
                    input_buf = ""
                    output_buf = ""
                    try:
                        async for message in gemini_ws:
                            if stop_flag_g.is_set():
                                break
                            try:
                                data = json.loads(message)
                                sc = data.get("serverContent", {})

                                it = sc.get("inputTranscription", {})
                                if it.get("text"):
                                    input_buf += it["text"]
                                    ws.send(json.dumps({"transcript": input_buf,
                                                        "is_final": False, "mode": "gemini"}))

                                ot = sc.get("outputTranscription", {})
                                if ot.get("text"):
                                    output_buf += ot["text"]
                                    ws.send(json.dumps({"translation": output_buf,
                                                        "is_final": False, "mode": "gemini"}))

                                for part in sc.get("modelTurn", {}).get("parts", []):
                                    if "inlineData" in part:
                                        ws.send(json.dumps({
                                            "type": "tts_chunk",
                                            "audio_b64": part["inlineData"]["data"],
                                            "audio_type": "audio/pcm",
                                            "mode": "gemini"
                                        }))

                                if sc.get("turnComplete"):
                                    if input_buf or output_buf:
                                        ws.send(json.dumps({"transcript": input_buf,
                                                            "translation": output_buf,
                                                            "is_final": True, "mode": "gemini"}))
                                        print(f"[Gemini] Turn: {input_buf[:60]}", flush=True)
                                    input_buf = ""
                                    output_buf = ""
                                    ws.send(json.dumps({"type": "tts_done", "mode": "gemini"}))
                            except Exception as e:
                                print(f"[Gemini] Parse error: {e}", flush=True)
                    except Exception as e:
                        print(f"[Gemini] Receive error: {e}", flush=True)

                await asyncio.gather(send_audio(), receive_output(), return_exceptions=True)

        except Exception as e:
            print(f"[Gemini] Connection error: {e}", flush=True)

    def process_gemini():
        asyncio.run(_gemini_stream())

    audio_thread_g = threading.Thread(target=receive_audio_g, daemon=True)
    gemini_thread  = threading.Thread(target=process_gemini,  daemon=True)
    audio_thread_g.start()
    gemini_thread.start()
    gemini_thread.join()
    stop_flag_g.set()
    audio_thread_g.join(timeout=2)


# ── Main WebSocket endpoint ───────────────────────────────────────────────────

@sock.route('/ws')
def websocket_endpoint(ws):
    print("Client connected", flush=True)
    try:
        config       = json.loads(ws.receive())
        source_lang  = config.get('source_lang', 'de')
        target_lang  = config.get('target_lang', 'it')
        mt_engine    = config.get('mt_engine', 'deepl')
        context_brief = config.get('context_brief', '').strip()
        formality    = config.get('formality', 'default')
        tts_engine   = config.get('tts_engine', 'deepgram')
        print(f"[WS] {source_lang} → {target_lang} via {mt_engine}", flush=True)

        if mt_engine == 'gemini':
            ws.send(json.dumps({'status': 'ready'}))
            handle_gemini_live(ws, source_lang, target_lang, api_key=GOOGLE_AI_KEY)
            return

        if context_brief:
            print(f"[WS] Context brief: {context_brief[:80]}...", flush=True)
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
                                if json.loads(msg).get('type') == 'close':
                                    stop_flag.set()
                                    break
                            except Exception:
                                pass
                except Exception:
                    continue

        def process_deepgram():
            import websockets

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

                # Single shared httpx client for all MT + TTS calls this session
                async with httpx.AsyncClient() as http:
                    try:
                        async with websockets.connect(
                            dg_url,
                            additional_headers=headers,
                            ping_interval=5,
                            ping_timeout=20,
                            close_timeout=10,
                        ) as dg_ws:
                            print("Connected to Deepgram ASR", flush=True)

                            async def send_audio():
                                last_ka = asyncio.get_event_loop().time()
                                try:
                                    while not stop_flag.is_set():
                                        try:
                                            await dg_ws.send(audio_queue.get(timeout=0.1))
                                            last_ka = asyncio.get_event_loop().time()
                                        except queue.Empty:
                                            now = asyncio.get_event_loop().time()
                                            if now - last_ka > 5:
                                                try:
                                                    await dg_ws.send(
                                                        json.dumps({"type": "KeepAlive"}))
                                                except Exception:
                                                    pass
                                                last_ka = now
                                            await asyncio.sleep(0.01)
                                except Exception as e:
                                    print(f"Send error: {e}", flush=True)

                            async def receive_transcription():
                                try:
                                    async for msg in dg_ws:
                                        data = json.loads(msg)
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
                                            ws.send(json.dumps(
                                                {'transcript': transcript, 'is_final': False}))
                                            continue

                                        print(f"[ASR] {transcript}", flush=True)
                                        if len(transcript.split()) < 3:
                                            print(f"[ASR] Skipping short segment", flush=True)
                                            ws.send(json.dumps(
                                                {'transcript': transcript, 'is_final': True}))
                                            continue

                                        # MT and send text -- await (non-blocking)
                                        translation = await translate_async(
                                            http, transcript, source_lang, target_lang,
                                            mt_engine, context_brief, formality)

                                        if not translation:
                                            ws.send(json.dumps({'transcript': transcript,
                                                                'translation': '[MT error]',
                                                                'is_final': True}))
                                            continue

                                        # Send text to frontend immediately
                                        ws.send(json.dumps({'transcript': transcript,
                                                            'translation': translation,
                                                            'is_final': True}))

                                        # TTS -- also awaited, but doesn't block ASR
                                        # because Deepgram keeps streaming in the
                                        # send_audio coroutine concurrently
                                        asyncio.ensure_future(
                                            synthesise_async(
                                                http, translation, target_lang,
                                                ws, tts_engine))

                                except Exception as e:
                                    print(f"Receive error: {e}", flush=True)

                            await asyncio.gather(
                                send_audio(),
                                receive_transcription(),
                                return_exceptions=True,
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
