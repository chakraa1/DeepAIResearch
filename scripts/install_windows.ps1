param(
    [string]$ProjectPath = "G:\Outskill\Hackathon\DeepAIResearch",
    [switch]$Run
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

Write-Step "Checking project path"
if (-not (Test-Path $ProjectPath)) {
    throw "Project path '$ProjectPath' does not exist. Clone or copy the repository there first."
}

Set-Location $ProjectPath

$streamlitApp = "src\deep_researcher\main\streamlit_app.py"
$envExample = "src\deep_researcher\config\.env.example"

if (-not (Test-Path "requirements.txt") -or -not (Test-Path $streamlitApp)) {
    throw "This folder does not look like the DeepAIResearch project. Missing requirements.txt or $streamlitApp."
}

Write-Step "Checking Python"
$pythonCommand = $null
foreach ($candidate in @("py -3.12", "py -3", "python")) {
    try {
        $versionOutput = Invoke-Expression "$candidate --version" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonCommand = $candidate
            Write-Host "Using $candidate ($versionOutput)"
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCommand) {
    throw "Python 3.10+ was not found. Install Python from https://www.python.org/downloads/ and retry."
}

Write-Step "Creating virtual environment"
if (-not (Test-Path ".venv")) {
    Invoke-Expression "$pythonCommand -m venv .venv"
} else {
    Write-Host ".venv already exists; reusing it."
}

$venvPython = Join-Path $ProjectPath ".venv\Scripts\python.exe"
$venvPip = Join-Path $ProjectPath ".venv\Scripts\pip.exe"
$venvStreamlit = Join-Path $ProjectPath ".venv\Scripts\streamlit.exe"

Write-Step "Installing Python dependencies"
& $venvPython -m pip install --upgrade pip
& $venvPip install -r requirements.txt

Write-Step "Preparing environment file"
if (-not (Test-Path ".env") -and (Test-Path $envExample)) {
    Copy-Item $envExample ".env"
    Write-Host "Created .env from .env.example. Add OPENAI_API_KEY and TAVILY_API_KEY for full web research."
} elseif (Test-Path ".env") {
    Write-Host ".env already exists; leaving it unchanged."
}

Write-Step "Verifying installation"
& $venvPython -m compileall $streamlitApp src tests

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "To start the app:"
Write-Host "  cd `"$ProjectPath`""
Write-Host "  .\.venv\Scripts\streamlit.exe run $streamlitApp"

if ($Run) {
    Write-Step "Starting Streamlit"
    & $venvStreamlit run $streamlitApp
}
