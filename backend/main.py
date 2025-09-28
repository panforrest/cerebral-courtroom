from fastapi import FastAPI
import logging
from pydantic import BaseModel
import os

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from . import prompts
from fastapi.responses import StreamingResponse, FileResponse
import json
from pathlib import Path
from fastapi import WebSocket, WebSocketDisconnect
from .agent_manager import AgentManager

# single global manager for demo
manager = AgentManager()

# basic logger for the backend module
logger = logging.getLogger('cerebral')
if not logger.handlers:
    # default handler for local runs/tests
    h = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

app = FastAPI(title="Cerebral Courtroom - Backend")

class CaseSubmission(BaseModel):
    title: str
    facts: str
    user_arguments: list[str] | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    """Return available OpenAI models for the configured API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    if OpenAI is None:
        return {"error": "openai package not installed in this env"}
    client = OpenAI(api_key=api_key)
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        return {"models": ids}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/case")
async def submit_case(case: CaseSubmission):
    # In the scaffold we simply echo back a session_id placeholder.
    return {"session_id": "demo-session-1", "title": case.title}


class DemoArg(BaseModel):
    facts: str
    argument: str


@app.post('/api/demo/opposing')
async def demo_opposing(payload: DemoArg):
    """Call Opposing Counsel (single-agent) and return its reply."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    if OpenAI is None:
        return {"error": "openai package not installed"}

    client = OpenAI(api_key=api_key)
    prompt = prompts.OPPOSING_PROMPT_TEMPLATE.format(facts=payload.facts, argument=payload.argument)
    try:
        resp = client.responses.create(
            model="gpt-5-codex",
            input=prompt,
            max_tokens=300
        )
        # responses.create returns a structured object; try to retrieve text.
        text = getattr(resp, 'output_text', None) or str(resp)
        return {"reply": text}
    except Exception as e:
        return {"error": str(e)}



@app.get('/api/demo/opposing-stream')
async def demo_opposing_stream(facts: str, argument: str):
    """Stream Opposing Counsel output as Server-Sent Events (SSE).
    Provide `facts` and `argument` as query parameters.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    if OpenAI is None:
        return {"error": "openai package not installed"}

    client = OpenAI(api_key=api_key)
    prompt = prompts.OPPOSING_PROMPT_TEMPLATE.format(facts=facts, argument=argument)

    def event_generator():
        try:
            with client.responses.stream(model="gpt-5-codex", input=prompt) as stream:
                for event in stream:
                    # stream output text deltas to client
                    if getattr(event, 'type', None) == 'response.output_text.delta':
                        delta = getattr(event, 'delta', '')
                        text = str(delta)
                        payload = {'type': 'delta', 'delta': text}
                        yield f"data: {json.dumps(payload)}\n\n"
            # final event
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.get('/demo.html')
async def demo_page():
    """Serve the demo HTML page so the demo is a single URL (no CORS needed)."""
    base = Path(__file__).resolve().parent.parent
    demo_path = base / 'frontend' / 'demo.html'
    if demo_path.exists():
        return FileResponse(str(demo_path))
    return {"error": "demo page not found"}


@app.get('/')
async def root_redirect():
    """Serve demo at root for convenience"""
    return FileResponse(str((Path(__file__).resolve().parent.parent / 'frontend' / 'demo.html')))


@app.post('/api/session')
async def create_session(payload: CaseSubmission):
    sid = manager.create_session(payload.title, payload.facts)
    return {"session_id": sid}


@app.websocket('/ws/session/{session_id}')
async def ws_session(ws: WebSocket, session_id: str):
    await ws.accept()
    logger.debug("[ws] accepted connection for session %s", session_id)
    try:
        while True:
            data = await ws.receive_json()
            logger.debug("[ws] recv for %s: %s", session_id, data)
            # data: {type: 'present', text: '...'}
            if data.get('type') == 'present':
                text = data.get('text', '')
                manager.add_user_presentation(session_id, text)
                # run the multi-agent sequence in a background thread so we don't block the event loop
                import asyncio

                def run_seq():
                    try:
                        return manager.run_turn_sequence(session_id, text)
                    except Exception as e:
                        logger.exception("[ws] run_seq exception for %s", session_id)
                        raise

                results = await asyncio.to_thread(run_seq)
                # send each agent's reply as it becomes available
                for r in results:
                    logger.debug("[ws] send to %s: %s", session_id, r)
                    await ws.send_json({'type': 'agent_reply', 'agent': r.get('agent'), 'text': r.get('text')})
    except WebSocketDisconnect:
        return
