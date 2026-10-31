[CmdletBinding()]
param(
    [ValidateSet('tokens', 'characters')][string]$ChunkUnit = 'tokens',
    [int]$ChunkSize = 512,
    [int]$ChunkOverlap = 50,
    [string]$Selection,
    [string]$DataDir,
    [string]$IndexDir,
    [switch]$IncludeCarf,
    [switch]$UsePrepared
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python = Join-Path $repoRoot '.venv-retrieval\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts\retrieval\setup.ps1 first.' }
Push-Location -LiteralPath $repoRoot
try {
    $cliArgs = @('-m', 'app.retrieval', 'build', '--chunk-unit', $ChunkUnit,
        '--chunk-size', $ChunkSize, '--chunk-overlap', $ChunkOverlap)
    if ($Selection) { $cliArgs += @('--selection', $Selection) }
    if ($DataDir) { $cliArgs += @('--data-dir', $DataDir) }
    if ($IndexDir) { $cliArgs += @('--index-dir', $IndexDir) }
    if ($IncludeCarf) { $cliArgs += '--include-carf' }
    if ($UsePrepared) { $cliArgs += '--use-prepared' }
    & $python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw 'Retrieval build failed.' }
} finally { Pop-Location }
