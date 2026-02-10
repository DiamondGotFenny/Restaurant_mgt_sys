from __future__ import annotations

import os
import struct
import tempfile
import uuid
from datetime import datetime, timezone
from threading import Lock, RLock
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from .logger_config import setup_logger
from .text_to_sql.text_to_sql_engine import TextToSQLEngine
from .vectorDB_Agent.vectorDB_Engine import VectorDBEngine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ = load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

logger = setup_logger(os.path.join(BASE_DIR, "logs", "app.log"), logger_name="server.app")


def _parse_origins(value: str | None) -> list[str]:
    if not value:
        # Vite defaults (dev). Adjust via CLIENT_ORIGINS env var.
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    return [o.strip() for o in value.split(",") if o.strip()]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Citation(BaseModel):
    source: str
    page: int | str | None = None
    chunk_id: str | None = None
    note: str | None = None


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    sender: Literal["system", "user", "assistant"]
    timestamp: datetime = Field(default_factory=_now)
    citations: list[Citation] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str


SOPHIE_SYSTEM_PROMPT = """You are Sophie, an NYC restaurant assistant.

Rules:
- You must answer ONLY using the provided CONTEXT or DATABASE_RESULT.
- If you do not have enough information to answer, say so clearly.
- Do not invent facts or cite sources you were not given.
- Be helpful and practical (suggest what to ask next when needed).
- Do not include a "Sources" section in the answer text; citations are returned separately.
"""

SOPHIE_GREETING = (
    "Hi, I'm Sophie. Ask me anything about NYC restaurants, and I'll answer using the data I have."
)


def _initial_history() -> list[Message]:
    return [
        Message(text=SOPHIE_SYSTEM_PROMPT, sender="system"),
        Message(text=SOPHIE_GREETING, sender="assistant"),
    ]


class ChatSessionStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, list[Message]] = {}

    def get_or_create(self, session_id: str) -> list[Message]:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = _initial_history()
            return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = _initial_history()


sessions = ChatSessionStore()


def get_session_id(request: Request, response: Response) -> str:
    """
    Session isolation via header (frontend stores it in localStorage).
    """
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        session_id = str(uuid.uuid4())
    response.headers["X-Session-Id"] = session_id
    return session_id


def _make_llm() -> AzureChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_API_VERSION")
    deployment = (
        os.getenv("OPENAI_MODEL_4OMINI")
        or os.getenv("OPENAI_MODEL_4o")
        or os.getenv("OPENAI_MODEL_35")
    )

    missing = []
    if not api_key:
        missing.append("OPENAI_API_KEY")
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_version:
        missing.append("AZURE_API_VERSION")
    if not deployment:
        missing.append("OPENAI_MODEL_4OMINI (or OPENAI_MODEL_4o / OPENAI_MODEL_35)")
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        temperature=0.2,
        max_tokens=800,
    )


_llm_lock = Lock()
_llm: AzureChatOpenAI | None = None


def get_llm() -> AzureChatOpenAI:
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = _make_llm()
    return _llm


_rag_lock = Lock()
_rag_engine: VectorDBEngine | None = None


def get_rag_engine() -> VectorDBEngine | None:
    global _rag_engine
    if _rag_engine is None:
        with _rag_lock:
            if _rag_engine is None:
                try:
                    _rag_engine = VectorDBEngine(
                        os.path.join(BASE_DIR, "logs", "vectorDB_Engine.log")
                    )
                except Exception as e:
                    logger.error("Failed to initialize RAG engine: %s", e)
                    _rag_engine = None
    return _rag_engine


_sql_lock = Lock()
_sql_engine: TextToSQLEngine | None = None


def get_sql_engine() -> TextToSQLEngine | None:
    global _sql_engine
    if _sql_engine is None:
        with _sql_lock:
            if _sql_engine is None:
                try:
                    _sql_engine = TextToSQLEngine(
                        os.path.join(BASE_DIR, "logs", "text_to_sql_engine.log")
                    )
                except Exception as e:
                    logger.error("Failed to initialize SQL engine: %s", e)
                    _sql_engine = None
    return _sql_engine


def _route(question: str) -> Literal["sql", "rag"]:
    q = question.lower()

    # Heuristic: "compute/sort/filter" style questions tend to be SQL-friendly.
    sql_hints = [
        "how many",
        "count",
        "number of",
        "average",
        "avg",
        "top",
        "highest",
        "lowest",
        "rank",
        "sorted",
        "order by",
        "filter",
        "between",
        "inspection",
        "grade",
        "score",
        "rating",
        "reviews",
        "menu",
        "price",
        "cheapest",
        "most expensive",
    ]
    if any(h in q for h in sql_hints):
        return "sql"
    return "rag"


def _history_to_lc_messages(history: list[Message], max_turns: int = 12) -> list:
    # Keep the initial system prompt, then last N messages to bound context.
    system_msgs = [m for m in history if m.sender == "system"]
    rest = [m for m in history if m.sender != "system"]
    rest = rest[-max_turns:]

    out = []
    if system_msgs:
        out.append(SystemMessage(content=system_msgs[0].text))
    for m in rest:
        if m.sender == "user":
            out.append(HumanMessage(content=m.text))
        elif m.sender == "assistant":
            out.append(AIMessage(content=m.text))
    return out


def _run_rag(question: str) -> tuple[str, list[Citation]]:
    engine = get_rag_engine()
    if engine is None or engine.ensemble_retriever is None:
        return (
            "I don't have the document search data available right now, so I can't answer that reliably.",
            [],
        )

    try:
        docs = engine.ensemble_retriever.invoke(question)
    except Exception as e:
        logger.error("RAG retrieval failed: %s", e)
        return ("I hit an error while searching the documents.", [])

    citations: list[Citation] = []
    context_blocks: list[str] = []
    max_chars_per_doc = 1500

    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        source = str(md.get("source", "Unknown"))
        page = md.get("page", None)
        chunk_id = md.get("chunk_id", None)

        citations.append(Citation(source=source, page=page, chunk_id=chunk_id))

        content = (getattr(d, "page_content", "") or "").strip()
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."

        context_blocks.append(
            f"source={source} page={page} chunk_id={chunk_id}\n{content}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    if not context.strip():
        return ("I couldn't find relevant information in my documents for that question.", [])

    llm = get_llm()
    prompt = (
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n"
    )
    try:
        resp = llm.invoke([SystemMessage(content=SOPHIE_SYSTEM_PROMPT), HumanMessage(content=prompt)])
        answer = (resp.content or "").strip()
    except Exception as e:
        logger.error("RAG LLM generation failed: %s", e)
        return ("I hit an error while generating the answer.", citations)

    answer = answer or "I don't have enough information to answer that."
    return answer, citations


def _run_sql(question: str) -> tuple[str, list[Citation]]:
    engine = get_sql_engine()
    if engine is None:
        return (
            "I don't have the database connection available right now, so I can't answer that reliably.",
            [],
        )

    try:
        state = engine.process_query(question)
    except Exception as e:
        logger.error("SQL tool failed: %s", e)
        return ("I hit an error while querying the database.", [])

    sql_query = state.get("query") or ""
    sql_result = state.get("result") or ""
    answer = state.get("answer") or ""

    # Keep sensitive/raw data out of the main chat text; attach as a citation note instead.
    note_parts = []
    if sql_query:
        note_parts.append(f"SQL:\n{sql_query}")
    if sql_result:
        max_result_chars = 1500
        trimmed = sql_result if len(sql_result) <= max_result_chars else (sql_result[:max_result_chars] + "...")
        note_parts.append(f"Result (raw):\n{trimmed}")

    citations = [Citation(source="database", note="\n\n".join(note_parts) or None)]

    if not answer:
        answer = "I couldn't produce an answer from the database result."

    return answer, citations


def _chat(question: str) -> tuple[str, list[Citation], Literal["sql", "rag"]]:
    route = _route(question)
    if route == "sql":
        answer, citations = _run_sql(question)
        # Fallback to RAG if SQL isn't available.
        if "don't have the database connection available" in answer:
            route = "rag"
            answer, citations = _run_rag(question)
        return answer, citations, route

    answer, citations = _run_rag(question)
    # Fallback to SQL if RAG can't run but SQL can.
    if "don't have the document search data available" in answer:
        sql_answer, sql_citations = _run_sql(question)
        return sql_answer, sql_citations, "sql"
    return answer, citations, "rag"


# Azure Speech setup (lazy init to avoid startup failures when env vars are missing).
_speech_lock = Lock()
_speech_config: speechsdk.SpeechConfig | None = None
_speech_synthesizer: speechsdk.SpeechSynthesizer | None = None


def _import_speechsdk():
    try:
        import azure.cognitiveservices.speech as speechsdk  # type: ignore
    except ModuleNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Azure Speech SDK is not installed. "
                "Install 'azure-cognitiveservices-speech' to use /chat-speech."
            ),
        )
    return speechsdk


def _init_speech() -> None:
    global _speech_config, _speech_synthesizer
    if _speech_config is not None and _speech_synthesizer is not None:
        return

    speechsdk = _import_speechsdk()

    speech_key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not speech_key or not region:
        raise ValueError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION.")

    _speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    _speech_config.speech_recognition_language = "en-US"
    _speech_config.speech_synthesis_language = "en-US"
    _speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    _speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=_speech_config, audio_config=None
    )


def get_speech() -> tuple[speechsdk.SpeechConfig, speechsdk.SpeechSynthesizer]:
    global _speech_config, _speech_synthesizer
    if _speech_config is None or _speech_synthesizer is None:
        with _speech_lock:
            if _speech_config is None or _speech_synthesizer is None:
                _init_speech()
    assert _speech_config is not None
    assert _speech_synthesizer is not None
    return _speech_config, _speech_synthesizer


ACCEPTED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/x-pn-wav",
}

MAX_MESSAGE_CHARS = 4000
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MiB


def _is_audio_file(file: UploadFile) -> bool:
    return (file.content_type or "").lower() in ACCEPTED_AUDIO_MIME_TYPES


async def _speech_to_text(audio_file: UploadFile) -> str | None:
    if not _is_audio_file(audio_file):
        return None

    speechsdk = _import_speechsdk()
    speech_config, _ = get_speech()
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        return None
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio_input = speechsdk.AudioConfig(filename=tmp_path)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_input
        )
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        return None
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _create_wav_header(channels: int, sample_rate: int, bits_per_sample: int) -> bytes:
    bytes_per_sample = bits_per_sample // 8
    block_align = channels * bytes_per_sample
    byte_rate = sample_rate * block_align

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        0,
    )
    return header


async def _text_to_speech_stream(text: str):
    speechsdk = _import_speechsdk()
    _, synthesizer = get_speech()
    try:
        result = synthesizer.speak_text_async(text).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError("TTS synthesis did not complete successfully.")

        audio_stream = speechsdk.AudioDataStream(result)
        audio_stream.position = 0

        # WAV header (mono/16k/16-bit)
        yield _create_wav_header(channels=1, sample_rate=16000, bits_per_sample=16)

        chunk_size = 16000
        audio_buffer = bytes(chunk_size)
        while True:
            filled_size = audio_stream.read_data(audio_buffer)
            if filled_size > 0:
                yield audio_buffer[:filled_size]
            else:
                break
    except Exception as ex:
        logger.error("Error synthesizing audio: %s", ex)
        yield b""


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(os.getenv("CLIENT_ORIGINS")),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)


@app.post("/chat-text/")
async def chat_text(chat: ChatRequest, session_id: str = Depends(get_session_id)):
    question = (chat.message or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="message must be non-empty")
    if len(question) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=413, detail="message too large")

    history = sessions.get_or_create(session_id)
    history.append(Message(text=question, sender="user"))

    answer, citations, route = _chat(question)
    assistant_msg = Message(text=answer, sender="assistant", citations=citations)
    history.append(assistant_msg)

    return {"response": assistant_msg.model_dump(), "route": route, "session_id": session_id}


@app.post("/chat")
async def chat(chat: ChatRequest, session_id: str = Depends(get_session_id)):
    # Alias for a unified entry point (recommended).
    return await chat_text(chat, session_id)


@app.post("/chat-speech")
async def chat_speech(
    request: Request,
    data: UploadFile = File(...),
):
    # Session header passthrough for StreamingResponse
    session_id = request.headers.get("X-Session-Id") or str(uuid.uuid4())

    if not _is_audio_file(data):
        raise HTTPException(status_code=415, detail="Unsupported audio type. Send WAV.")

    input_text = await _speech_to_text(data)
    if not input_text:
        raise HTTPException(status_code=400, detail="Speech not recognized.")

    history = sessions.get_or_create(session_id)
    history.append(Message(text=input_text, sender="user"))

    answer, citations, _ = _chat(input_text)
    assistant_msg = Message(text=answer, sender="assistant", citations=citations)
    history.append(assistant_msg)

    stream = _text_to_speech_stream(answer)
    return StreamingResponse(
        stream,
        media_type="audio/wav",
        headers={"X-Session-Id": session_id},
    )


@app.get("/chat_history/")
async def get_chat_history(session_id: str = Depends(get_session_id)):
    history = sessions.get_or_create(session_id)
    return {
        "chat_history": [m.model_dump() for m in history if m.sender != "system"],
        "session_id": session_id,
    }


@app.post("/clear_chat_history/")
async def clear_chat_history(session_id: str = Depends(get_session_id)):
    sessions.clear(session_id)
    return {"message": "Chat history has been cleared.", "session_id": session_id}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
