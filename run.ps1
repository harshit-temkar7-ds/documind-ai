# One-click setup + run for Windows PowerShell.
#   Right-click > Run with PowerShell   or   .\run.ps1
# Requires Python 3.10, 3.11 or 3.12 (NOT 3.13/3.14).
Set-Location $PSScriptRoot

# pick a supported Python
$py = $null
foreach ($v in @("3.12", "3.11", "3.10")) {
    try {
        $null = & py "-$v" --version 2>&1
        if ($LASTEXITCODE -eq 0) { $py = "-$v"; break }
    } catch { }
}
if (-not $py) {
    Write-Host "Python 3.10-3.12 was not found. Install Python 3.12 from https://www.python.org/downloads/ and run again." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment with Python $py ..."
    & py $py -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dependency installation failed - see the error above." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "`nCreated .env - opening it so you can paste your GROQ_API_KEY." -ForegroundColor Yellow
    notepad .env
    Read-Host "Press Enter after saving .env"
}
New-Item -ItemType Directory -Force data\uploads | Out-Null

Write-Host "`nStarting DocuMind AI on http://localhost:8000 (first upload downloads the embedding model, ~90 MB)" -ForegroundColor Green
python main.py
