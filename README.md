# Restaurant_mgt_sys

NYC restaurant assistant demo with:
- Text chat and speech chat (Azure Speech STT/TTS)
- RAG over local PDFs (hybrid retrieval)
- Text-to-SQL over a Postgres database

## Repo Layout
- `client/`: React + TypeScript + Vite frontend
- `server/`: FastAPI backend (RAG + Text-to-SQL + speech)

## Quick Start

### Backend (FastAPI)
1. Create a Python venv and install deps:
   - `python -m venv .venv`
   - Windows PowerShell: `.\\.venv\\Scripts\\pip install -r server\\requirements.txt`
2. Configure env:
   - Copy `server/.env.example` to `server/.env` and fill in values
3. Run:
   - `python -m uvicorn server.app:app --reload --port 8000`

Health check: `GET /healthz`

### Frontend (React)
1. Install:
   - `cd client`
   - `npm install`
2. Configure env:
   - Copy `client/.env.example` to `client/.env`
3. Run:
   - `npm run dev`

## API Endpoints
- `POST /chat-text/` body: `{ "message": "..." }`
- `POST /chat-speech` multipart/form-data field: `data` (WAV)
- `GET /chat_history/`
- `POST /clear_chat_history/`

Session isolation is done via the `X-Session-Id` header (the frontend stores it in `localStorage`).

## Notes
- RAG uses PDFs under `server/data/Restaurants_data/` (if present).
- Do not commit `server/.env` / `client/.env` (they may contain secrets). Rotate keys if they were ever exposed.
