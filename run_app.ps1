# Starts the app. Uses the GPU environment (~/rag-gpu-venv) if it exists, else .venv.
Set-Location $PSScriptRoot
$gpu = Join-Path $HOME "rag-gpu-venv\Scripts\python.exe"
$python = if (Test-Path $gpu) { $gpu } else { ".\.venv\Scripts\python.exe" }
& $python -m streamlit run app.py
