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
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, TranscriptionFrame, TTSSpeakFrame
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
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from src.rag_tool import rag_lookup

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
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")

# Validate that critical keys are present
_REQUIRED_KEYS = {
    "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
    "ELEVENLABS_VOICE_ID": ELEVENLABS_VOICE_ID,
}
for _key, _val in _REQUIRED_KEYS.items():
    if not _val:
        logger.warning("Environment variable %s is not set – calls will fail.", _key)

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a friendly, professional, and multilingual telecom customer \
support agent working for a telecommunications company. Your job is to help callers \
with their questions about plans, billing, network issues, SIM cards, roaming, and \
any other telecom-related topics.

IMPORTANT RULES:
1. **Language matching** – Always reply in the same language the customer is speaking. \
If they switch languages mid-conversation, switch with them seamlessly.
2. **Conciseness** – Keep answers short and to the point. This is a phone call, not \
an essay. Aim for 1-3 sentences per turn unless the customer asks for more detail.
3. **Accuracy** – When context information is provided below, base your answers on \
that context. If the context does not cover the question, say so honestly and offer \
to transfer to a specialist.
4. **Tone** – Be warm, patient, and empathetic. Acknowledge frustrations before \
offering solutions.
5. **No hallucination** – Never invent plan prices, policy details, or technical \
specs. Stick to the provided context or clearly state that you need to check.

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
                    rag_context = await rag_lookup(user_text, top_k=3)
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

        # ── STT (Deepgram) with auto-detect multi-language ───────────────
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            settings=DeepgramSTTService.Settings(
                language=None,  # Auto-detect language
            ),
        )

        # ── LLM (Groq) ──────────────────────────────────────────────────
        llm = GroqLLMService(
            api_key=GROQ_API_KEY,
            settings=GroqLLMService.Settings(
                model=GROQ_MODEL,
            ),
        )

        # ── TTS (ElevenLabs) ─────────────────────────────────────────────
        tts = ElevenLabsTTSService(
            api_key=ELEVENLABS_API_KEY,
            settings=ElevenLabsTTSService.Settings(
                voice=ELEVENLABS_VOICE_ID,
            ),
        )

        # ── LLM Context & Aggregators ───────────────────────────────────
        context = LLMContext(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
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
            logger.info("Twilio client connected: %s", client)
            # Greet the caller as soon as the call connects
            await task.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": "Please greet the caller warmly and concisely with: 'Hello! Thank you for calling customer support. How can I assist you today?'",
                        }
                    ],
                    run_llm=True,
                )
            )

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport_inst, client):
            logger.info("Twilio client disconnected: %s", client)
            await task.cancel()

        logger.info("Starting Pipecat pipeline")
        await runner.run(task)
        logger.info("Pipecat pipeline finished")

    except Exception:
        logger.exception("Error in media stream pipeline")
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
