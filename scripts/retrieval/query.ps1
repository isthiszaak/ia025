[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Question,
    [string]$Cases,
    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python = Join-Path $repoRoot '.venv-retrieval\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts\retrieval\setup.ps1 first.' }
$cliArgs = @('-m', 'app.retrieval', 'query')
if ($Question) { $cliArgs += $Question }
if ($Cases) { $cliArgs += @('--cases', $Cases) }
if ($OutputDir) { $cliArgs += @('--output-dir', $OutputDir) }
Push-Location -LiteralPath $repoRoot
try {
    & $python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw 'Retrieval query failed.' }
} finally { Pop-Location }
