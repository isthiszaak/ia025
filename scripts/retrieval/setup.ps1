[CmdletBinding()]
param([string]$Python)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Push-Location -LiteralPath $repoRoot
try {
    if (-not $Python) {
        $candidates = @(
            (Join-Path $env:LOCALAPPDATA 'Python\pythoncore-3.12-64\python.exe'),
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
            (Join-Path $repoRoot '.tools\python\python.exe')
        )
        $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if (-not $Python) {
            $command = Get-Command python -ErrorAction SilentlyContinue
            if ($command) { $Python = $command.Source }
        }
        if (-not $Python) { throw 'Python 3.10+ is required. Run setup.ps1 -Python "C:\path\to\python.exe".' }
    }
    $venvPython = Join-Path $repoRoot '.venv-retrieval\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $Python -m venv (Join-Path $repoRoot '.venv-retrieval')
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv-retrieval.' }
    }
    & $venvPython -m pip install --disable-pip-version-check --no-compile --cache-dir (Join-Path $repoRoot '.cache\pip') -r (Join-Path $repoRoot 'app\retrieval\requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Retrieval dependency installation failed.' }
    Write-Output "Retrieval environment ready: $venvPython"
} finally { Pop-Location }
