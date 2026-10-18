"""
VoiceRAG Telephony Server
=========================

FastAPI application that bridges Twilio phone calls with a Pipecat AI pipeline.

Architecture:
  Twilio PSTN ──▶ /twilio-hook (TwiML) ──▶ WebSocket /media-stream
       ▼
  Pipecat Pipeline:
    FastAPIWebsocketTransport.input (TwilioFrameSerializer)
        ──▶ Deepgram STT
        ──▶ RAG Context Injector
        ──▶ LLM User Context Aggregator (with Silero VAD)
        ──▶ Groq LLM
        ──▶ ElevenLabs TTS
        ──▶ LLM Assistant Context Aggregator
        ──▶ FastAPIWebsocketTransport.output

Environment variables (loaded from .env):
  DEEPGRAM_API_KEY, GROQ_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
  TWILIO_ACCOUNT_SID (optional), TWILIO_AUTH_TOKEN (optional)
"""

from __future__ import annotations

import logging
import os
import sys
import asyncio
from typing import Any

import uvicorn
from dotenv import load_dotenv
import json
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, Response

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMMessagesAppendFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from src.rag_tool import get_embedding_model, rag_lookup

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("voicerag.server")

# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY") or os.getenv("Gemini_API_KEY", "")
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")

# Validate that critical keys are present
if not (GEMINI_API_KEY or GROQ_API_KEY):
    logger.warning("Neither GEMINI_API_KEY nor GROQ_API_KEY is set – LLM will fail.")
if not DEEPGRAM_API_KEY:
    logger.warning("DEEPGRAM_API_KEY is not set – STT will fail.")
if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY is not set – TTS will fail.")

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a warm, courteous, and professional female telecom customer support \
representative. Your goal is to provide immediate, clear, and helpful answers to callers about \
plans, 5G data, billing, eSIM, SIM replacement, roaming, and telecom services.

CRITICAL VOICE & LATENCY RULES:
1. **Language Matching**: Always reply in the EXACT language the customer speaks.
   - If the customer speaks Hindi (or Hinglish), reply in natural, polite Hindi.
   - If they speak English, Spanish, or French, reply fluently in that language.
2. **Extreme Conciseness (Low Latency)**: Keep every answer to 1-2 short, punchy sentences \
(maximum 25 words). Never output lengthy paragraphs or lists. This is a real-time voice call.
3. **Warm Female Persona**: Speak in a gentle, polite, and reassuring tone.
4. **Accuracy & No Hallucination**: Use the CONTEXT block below whenever available. If the \
information is not in the context, politely state that you can check further or transfer them.

If a CONTEXT block appears below, use it to answer the customer's question:
"""


# ──────────────────────────────────────────────────────────────────────────────
# RAG Context Injector  (custom FrameProcessor)
# ──────────────────────────────────────────────────────────────────────────────
class RAGContextInjector(FrameProcessor):
    """Intercepts user transcription frames, performs a RAG lookup, and injects
    the retrieved context into the LLM system message before the LLM processes
    the user's turn.
    """

    def __init__(self, context: LLMContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process incoming frames – enrich context on user transcriptions."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            user_text: str = frame.text
            if user_text and user_text.strip():
                logger.info("RAG lookup for: %s", user_text[:120])
                try:
                    rag_context = await rag_lookup(user_text, top_k=3, score_threshold=0.40)
                except Exception:
                    logger.exception("RAG lookup failed – proceeding without context")
                    rag_context = ""

                # Rebuild the system message with the fresh context block
                if rag_context:
                    enriched_system = (
                        f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{rag_context}"
                    )
                else:
                    enriched_system = SYSTEM_PROMPT

                # Update the first (system) message in the context
                messages = self._context.get_messages()
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = enriched_system
                else:
                    messages.insert(
                        0, {"role": "system", "content": enriched_system}
                    )
                self._context.set_messages(messages)

        # Always push the frame downstream so the pipeline continues
        await self.push_frame(frame, direction)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VoiceRAG Telephony Server",
    version="1.0.0",
    docs_url="/docs",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-warm the SentenceTransformer embedding model into memory on boot.
    Prevents the first caller from incurring the ~470MB cold-load latency stall.
    """
    logger.info("Pre-warming SentenceTransformer model on FastAPI startup...")
    await asyncio.to_thread(get_embedding_model)
    logger.info("SentenceTransformer model successfully warmed into memory.")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.api_route("/", methods=["GET", "POST"])
@app.api_route("/twilio-hook", methods=["GET", "POST"])
async def twilio_hook(request: Request) -> Response:
    """Twilio webhook that returns TwiML directing the call to a WebSocket
    media stream at ``/media-stream`` on the same host.

    Twilio will open a bi-directional WebSocket and stream audio in real time.
    """
    host: str = request.headers.get("host", "localhost:8765")
    ws_url = f"wss://{host}/media-stream"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "  <Connect>"
        f'    <Stream url="{ws_url}" />'
        "  </Connect>"
        "</Response>"
    )

    logger.info("Twilio hook called (%s %s) – streaming to %s", request.method, request.url.path, ws_url)
    return Response(content=twiml, media_type="application/xml")


# ──────────────────────────────────────────────────────────────────────────────
# Browser Audio Serializer (Zero-Twilio Direct Testing)
# ──────────────────────────────────────────────────────────────────────────────
class BrowserAudioSerializer(FrameSerializer):
    """Serializes frames to/from the browser WebSocket client.

    Inbound:
      - Raw 16-bit linear PCM audio bytes (16kHz mono) from browser microphone
        are converted to InputAudioRawFrame.
      - JSON string control messages (e.g. ping, config).
    Outbound:
      - AudioRawFrame (from ElevenLabs TTS) -> raw PCM16 bytes sent to browser.
      - InterruptionFrame -> JSON {"event": "clear", "type": "interruption"}.
      - TranscriptionFrame -> JSON {"event": "transcription", "text": frame.text, "user_id": getattr(frame, "user_id", "user")}.
    """

    def __init__(self, sample_rate: int = 16000, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sample_rate = sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, AudioRawFrame):
            return frame.audio
        elif isinstance(frame, InterruptionFrame):
            return json.dumps({"event": "clear", "type": "interruption"})
        elif isinstance(frame, TranscriptionFrame):
            return json.dumps({
                "event": "transcription",
                "text": frame.text,
                "user_id": getattr(frame, "user_id", "user"),
            })
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            if not data:
                return None
            return InputAudioRawFrame(
                audio=data,
                num_channels=1,
                sample_rate=self._sample_rate,
            )
        elif isinstance(data, str):
            try:
                # Browser control message if needed
                return None
            except Exception:
                return None
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Shared Voice Pipeline Runner
# ──────────────────────────────────────────────────────────────────────────────
async def run_voice_pipeline(
    transport: FastAPIWebsocketTransport,
    client_description: str = "Client",
) -> None:
    """Core Pipecat voice pipeline shared across Twilio PSTN and Browser WebSockets."""
    # ── STT (Deepgram Nova-2 with low-latency multilingual endpointing)
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        settings=DeepgramSTTService.Settings(
            model="nova-2",
            language="multi",
            endpointing=100,
            smart_format=True,
        ),
    )

    # ── LLM (Gemini preferred, or Groq fallback) ─────────────────────
    if GEMINI_API_KEY:
        logger.info("Using Google Gemini LLM: model=%s", GEMINI_MODEL)
        llm = OpenAILLMService(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            settings=OpenAILLMService.Settings(
                model=GEMINI_MODEL,
                temperature=0.2,
                top_p=0.9,
            ),
        )
    else:
        logger.info("Using Groq LLM: model=%s", GROQ_MODEL)
        llm = GroqLLMService(
            api_key=GROQ_API_KEY,
            settings=GroqLLMService.Settings(
                model=GROQ_MODEL,
                temperature=0.2,
                top_p=0.9,
            ),
        )

    # ── TTS (ElevenLabs Turbo v2.5 with natural female voice) ───────
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        sample_rate=16000,
        settings=ElevenLabsTTSService.Settings(
            model="eleven_turbo_v2_5",
            voice=ELEVENLABS_VOICE_ID or "21m00Tcm4TlvDq8ikWAM",
            stability=0.5,
            similarity_boost=0.8,
        ),
    )

    # ── LLM Context & Aggregators ───────────────────────────────────
    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
    )
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.7,
            start_secs=0.2,
            stop_secs=0.2,
            min_volume=0.6,
        ),
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
        ),
    )

    # ── RAG Processor ────────────────────────────────────────────────
    rag_processor = RAGContextInjector(context=context, name="RAGContextInjector")

    # ── Pipeline ─────────────────────────────────────────────────────
    # Order:
    #   Transport.input → STT → RAG Injector → User Aggregator (VAD)
    #       → LLM → TTS → Assistant Aggregator → Transport.output
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rag_processor,
            user_aggregator,
            llm,
            tts,
            assistant_aggregator,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    # ── Run ──────────────────────────────────────────────────────────
    runner = PipelineRunner()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport_inst, client):
        logger.info("%s connected: %s", client_description, client)
        # Greet the caller as soon as the call connects
        await task.queue_frame(
            LLMMessagesAppendFrame(
                messages=[
                    {
                        "role": "user",
                        "content": "Please greet the caller warmly and concisely with: 'Hello, Namaste! Thank you for calling customer support. How can I assist you today?'",
                    }
                ],
                run_llm=True,
            )
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_inst, client):
        logger.info("%s disconnected: %s", client_description, client)
        await task.cancel()

    logger.info("Starting Pipecat pipeline for %s", client_description)
    await runner.run(task)
    logger.info("Pipecat pipeline finished for %s", client_description)


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/browser", response_class=HTMLResponse)
@app.get("/test", response_class=HTMLResponse)
async def browser_page() -> HTMLResponse:
    """Serve the interactive browser-based testing console."""
    html_file = Path(__file__).resolve().parent.parent / "browser_voice_client.html"
    if not html_file.exists():
        html_file = Path(__file__).resolve().parent / "browser_voice_client.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>browser_voice_client.html not found</h2>", status_code=404)


@app.websocket("/browser-stream")
async def browser_stream(websocket: WebSocket) -> None:
    """Handle direct browser WebSocket audio streaming for local testing.
    Bypasses Twilio and PSTN carrier restrictions completely.
    """
    await websocket.accept()
    logger.info("Browser audio stream WebSocket connected")

    try:
        serializer = BrowserAudioSerializer(sample_rate=16000)
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=serializer,
            ),
        )
        await run_voice_pipeline(transport, client_description="Browser Microphone Client")
    except Exception:
        logger.exception("Error in browser audio stream pipeline")
    finally:
        logger.info("Browser audio stream WebSocket closed")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    """Handle the bi-directional Twilio media stream WebSocket.

    Sets up and runs the full Pipecat pipeline for real-time voice AI.
    """
    await websocket.accept()
    logger.info("Twilio media stream WebSocket connected")

    try:
        # ── Parse Twilio Initial Connection Parameters ───────────────────
        _, call_data = await parse_telephony_websocket(websocket)
        stream_sid = call_data.get("stream_id", "")
        call_sid = call_data.get("call_id", "")
        logger.info("Twilio call established. stream_sid=%s call_sid=%s", stream_sid, call_sid)

        # ── Serializer & Transport ───────────────────────────────────────
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            account_sid=TWILIO_ACCOUNT_SID or None,
            auth_token=TWILIO_AUTH_TOKEN or None,
            params=TwilioFrameSerializer.InputParams(auto_hang_up=bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)),
        )

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=serializer,
            ),
        )

        await run_voice_pipeline(transport, client_description=f"Twilio Call ({call_sid})")

    except Exception:
        logger.exception("Error in Twilio media stream pipeline")
    finally:
        logger.info("Twilio media stream WebSocket closed")


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # If NGROK_AUTHTOKEN is provided in .env or environment, start tunnel automatically
    ngrok_token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if ngrok_token and ngrok_token != "your_ngrok_authtoken_here":
        try:
            import ngrok
            ngrok_domain = os.getenv("NGROK_DOMAIN", "").strip()
            kwargs: dict[str, Any] = {"authtoken": ngrok_token}
            if ngrok_domain and ngrok_domain != "your_optional_static_domain":
                kwargs["domain"] = ngrok_domain
            listener = ngrok.forward(8765, **kwargs)
            public_url = listener.url()
            logger.info("=" * 65)
            logger.info("✨ Ngrok tunnel online: %s", public_url)
            logger.info("👉 Paste into Twilio Webhook: %s/twilio-hook", public_url)
            logger.info("=" * 65)
        except Exception as e:
            logger.error("Failed to automatically start ngrok tunnel: %s", e)

    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8765,
        log_level="info",
        reload=False,
    )
