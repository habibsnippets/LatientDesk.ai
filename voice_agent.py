"""LatientDesk.ai voice receptionist — Pipecat STT -> LLM -> TTS pipeline.

Wires the tool-calling receptionist from agent.py into a streaming voice
pipeline. The brain (system prompt, tool schemas, tool executor, conversation
strategy) is imported unchanged from agent.py / pms.py / insurance.py.

Slots:
  STT: Groq Whisper (`whisper-large-v3-turbo`) — free tier (~2,000 req/day),
       reuses GROQ_API_KEY. VAD-gated HTTP transcription (Groq's Whisper
       endpoint is HTTP, so transcription happens per speech turn).
  LLM: Groq Llama (default llama-3.1-8b-instant, --model to switch).
  TTS: Piper (local, no key, $0) — voice model auto-downloads on first run.

Run:
    GROQ_API_KEY=gsk_... .venv/bin/python voice_agent.py --reseed

Requirements (see requirements-voice.txt):
    pipecat-ai[silero,local,piper], websockets, piper-tts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

# Suppress harmless HuggingFace warning about local PyTorch
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Silence noisy ALSA soundcard warnings on Linux
try:
    import ctypes
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(None)
except Exception:
    pass

import pms
import agent as core

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

STT_MODEL = "whisper-large-v3-turbo"


# ---------------------------------------------------------------------------
# Tool schemas for Pipecat (FunctionSchema instead of raw OpenAI dicts)
# ---------------------------------------------------------------------------

def receptionist_tools_schema() -> ToolsSchema:
    """Convert agent.TOOLS (OpenAI-style dicts) into Pipecat's ToolsSchema."""
    schemas = []
    for tool in core.TOOLS:
        fn = tool["function"]
        schemas.append(
            FunctionSchema(
                name=fn["name"],
                description=fn["description"],
                properties=fn["parameters"].get("properties", {}),
                required=fn["parameters"].get("required", []),
            )
        )
    return ToolsSchema(standard_tools=schemas)


# ---------------------------------------------------------------------------
# The same receptionist brain, as a Pipecat LLMService
# ---------------------------------------------------------------------------

class DentalReceptionistLLM(GroqLLMService):
    """GroqLLMService with the receptionist tools registered.

    Tool execution goes through core.execute_tool, the exact same executor the
    text CLI uses — pms.py and insurance.py are untouched.
    """

    def __init__(self, conn: sqlite3.Connection, api_key: str, model: str, **kwargs: Any):
        super().__init__(api_key=api_key, settings=self.Settings(model=model), **kwargs)
        self._conn = conn
        self._register_receptionist_tools()

    def _register_receptionist_tools(self) -> None:
        self.register_function("get_open_slots", self._tool_get_open_slots)
        self.register_function("create_appointment", self._tool_create_appointment)
        self.register_function("verify_insurance", self._tool_verify_insurance)

    async def _tool_get_open_slots(self, params: Any) -> None:
        result = core.execute_tool("get_open_slots", dict(params.arguments), self._conn)
        await params.result_callback(result)

    async def _tool_create_appointment(self, params: Any) -> None:
        result = core.execute_tool("create_appointment", dict(params.arguments), self._conn)
        await params.result_callback(result)

    async def _tool_verify_insurance(self, params: Any) -> None:
        result = core.execute_tool("verify_insurance", dict(params.arguments), self._conn)
        await params.result_callback(result)


# ---------------------------------------------------------------------------
# Console observer: prints caller speech and tool traffic
# ---------------------------------------------------------------------------

class ConsoleVoiceObserver(FrameProcessor):
    """Pass-through processor that logs the voice conversation to the console."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            print(f"\n[caller] {frame.text}")
        elif isinstance(frame, FunctionCallInProgressFrame):
            print(f"[tool] {frame.function_name}({dict(frame.arguments)})")
        elif isinstance(frame, FunctionCallResultFrame):
            print(f"[tool result] {str(frame.result)[:120]}")
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

def build_task(conn: sqlite3.Connection, model: str, voice: str, api_key: str) -> PipelineTask:
    # Local mic/speaker transport with Silero VAD gating STT.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    stt = GroqSTTService(api_key=api_key, settings=GroqSTTService.Settings(model=STT_MODEL))
    llm = DentalReceptionistLLM(conn=conn, api_key=api_key, model=model)

    # Local Piper TTS: no API key. The voice model downloads on first use.
    from pipecat.services.piper.tts import PiperTTSService

    voices_dir = Path(__file__).resolve().parent / "voices"
    voices_dir.mkdir(exist_ok=True)  # piper's downloader won't create it
    tts = PiperTTSService(
        download_dir=voices_dir,
        settings=PiperTTSService.Settings(voice=voice),
    )

    messages = [{"role": "system", "content": core.SYSTEM_PROMPT}]
    context = LLMContext(messages, tools=receptionist_tools_schema())
    aggregators = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),           # mic audio in
            stt,                         # Groq Whisper: speech -> text
            aggregators.user(),          # aggregates caller turns into context
            llm,                         # receptionist brain + tool calls
            ConsoleVoiceObserver(),      # console echo of speech/tools
            tts,                         # Piper: text -> speech
            transport.output(),          # speaker audio out
            aggregators.assistant(),     # aggregates agent turns into context
        ]
    )

    return PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LatientDesk.ai voice receptionist (Pipecat)")
    parser.add_argument("--model", default=core.DEFAULT_MODEL, help=f"Groq LLM (default {core.DEFAULT_MODEL})")
    parser.add_argument("--voice", default="en_US-amy-medium", help="Piper voice id")
    parser.add_argument("--db", default=str(pms.DB_PATH), help="SQLite database path")
    parser.add_argument("--reseed", action="store_true", help="Reseed the demo database first")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is not set — it is required for both Whisper STT and the LLM.")
        print("Get a free key at https://console.groq.com then: export GROQ_API_KEY=gsk_...")
        return

    conn = pms.connect(args.db)
    if args.reseed or not Path(args.db).exists():
        n = pms.seed(conn)
        print(f"[db] seeded {n} open slots at {args.db}")

    async def _run() -> None:
        task = build_task(conn, args.model, args.voice, api_key)
        print(f"[voice] STT={STT_MODEL} (Groq) | LLM={args.model} (Groq) | TTS=Piper/{args.voice}")
        print("[voice] Speak when ready. Ctrl-C to exit.")
        runner = PipelineRunner()
        await runner.run(task)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n[voice] goodbye")
    except Exception as exc:  # noqa: BLE001 — surface mic/device errors readably
        logger.error(f"Pipeline failed: {exc}")
        print(f"[voice] pipeline error: {exc}")
        print("If this is a microphone error, check audio devices (PyAudio) and try again.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
