Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env — add OPENAI_API_KEY if you have one."
}

Write-Host "Starting FastAPI backend on http://localhost:8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot'; .\venv\Scripts\python -m uvicorn backend.api:app --reload --port 8000"

Start-Sleep -Seconds 3

Write-Host "Starting Streamlit on http://localhost:8501 — opening in browser..."
.\venv\Scripts\streamlit run frontend/app.py --server.port 8501
