[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Question,
    [string]$Cases,
    [string]$OutputDir,
    [ValidateSet('tokens', 'characters')][string]$ChunkUnit = 'tokens',
    [int]$ChunkSize = 512,
    [int]$ChunkOverlap = 50,
    [int]$TopK = 5,
    [int]$Limit,
    [string]$Selection,
    [string]$DataDir,
    [string]$IndexDir
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python = Join-Path $repoRoot '.venv-retrieval\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts\retrieval\setup.ps1 first.' }
$cliArgs = @('-m', 'app.retrieval', 'query', '--chunk-unit', $ChunkUnit,
    '--chunk-size', $ChunkSize, '--chunk-overlap', $ChunkOverlap, '--top-k', $TopK)
if ($Question) { $cliArgs += $Question }
if ($Cases) { $cliArgs += @('--cases', $Cases) }
if ($OutputDir) { $cliArgs += @('--output-dir', $OutputDir) }
if ($PSBoundParameters.ContainsKey('Limit')) { $cliArgs += @('--limit', $Limit) }
if ($Selection) { $cliArgs += @('--selection', $Selection) }
if ($DataDir) { $cliArgs += @('--data-dir', $DataDir) }
if ($IndexDir) { $cliArgs += @('--index-dir', $IndexDir) }
Push-Location -LiteralPath $repoRoot
try {
    & $python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw 'Retrieval query failed.' }
} finally { Pop-Location }
