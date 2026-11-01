[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Cases,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [ValidateRange(1, 2147483647)][int]$Limit,
    [string]$Selection,
    [ValidateRange(0, 255)][int]$ChunkOverlap = 50,
    [switch]$BuildIndexes
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python = Join-Path $repoRoot '.venv-retrieval\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts\retrieval\setup.ps1 first.' }
$cliArgs = @('-m', 'app.retrieval', 'compare', '--cases', $Cases, '--output-dir', $OutputDir,
    '--chunk-unit', 'tokens', '--chunk-overlap', "$ChunkOverlap")
if ($PSBoundParameters.ContainsKey('Limit')) { $cliArgs += @('--limit', "$Limit") }
if ($Selection) { $cliArgs += @('--selection', $Selection) }
if ($BuildIndexes) { $cliArgs += '--build-indexes' }
Push-Location -LiteralPath $repoRoot
try {
    & $python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw 'Retrieval comparisons contain failures; inspect the output manifests.' }
} finally { Pop-Location }
