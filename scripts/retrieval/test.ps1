[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python = Join-Path $repoRoot '.venv-retrieval\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts\retrieval\setup.ps1 first.' }
Push-Location -LiteralPath $repoRoot
try {
    & $python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
} finally { Pop-Location }
