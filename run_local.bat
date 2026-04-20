@echo off
REM Quick-start script for Agentic DS Ledger (Phase 1 local mode)
REM Run this from the agentic-ds-ledger\ folder

echo Starting Agentic DS Ledger (local mode)...

REM Copy .env.example to .env if it doesn't exist
if not exist .env (
    copy .env.example .env
    echo Created .env from .env.example — add your OPENAI_API_KEY if you have one.
)

REM Start FastAPI backend in new window
start "DS Ledger Backend" cmd /k "venv\Scripts\python -m uvicorn backend.api:app --reload --port 8000"

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Start Streamlit frontend
echo Opening Streamlit frontend at http://localhost:8501
venv\Scripts\streamlit run frontend/app.py --server.port 8501
