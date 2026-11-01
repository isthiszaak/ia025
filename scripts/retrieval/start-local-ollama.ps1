[CmdletBinding()]
param(
    [string]$OllamaPath,
    [ValidateRange(1024, 65535)][int]$Port = 11435,
    [switch]$PullModel,
    [string]$Model = 'embeddinggemma:300m'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $OllamaPath) { $OllamaPath = Join-Path $repoRoot '.tools\ollama\ollama.exe' }
if (-not (Test-Path -LiteralPath $OllamaPath)) {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) { $OllamaPath = $command.Source }
    else { throw 'Provide -OllamaPath or extract the official Windows portable ZIP into .tools\ollama.' }
}
$OllamaPath = (Resolve-Path -LiteralPath $OllamaPath).Path
$baseUrl = "http://127.0.0.1:$Port"
function Test-LocalOllama {
    try {
        $request = [System.Net.WebRequest]::Create("$baseUrl/api/version")
        $request.Proxy = $null
        $request.Timeout = 2500
        $response = $request.GetResponse()
        $response.Close()
        return $true
    } catch { return $false }
}
$ready = Test-LocalOllama
$previousEnvironment = @{}
foreach ($name in @('OLLAMA_HOST', 'OLLAMA_MODELS', 'OLLAMA_NO_CLOUD', 'USERPROFILE')) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $env:OLLAMA_HOST = "127.0.0.1:$Port"
    $env:OLLAMA_MODELS = Join-Path $repoRoot 'data\retrieval\models'
    $env:OLLAMA_NO_CLOUD = '1'
    # Ollama uses USERPROFILE for its signing key on Windows. Keep that state local.
    $env:USERPROFILE = Join-Path $repoRoot '.cache\retrieval\ollama-home'
    if (-not $ready) {
        $logDir = Join-Path $repoRoot '.cache\retrieval'
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        New-Item -ItemType Directory -Path $env:OLLAMA_MODELS -Force | Out-Null
        $server = Start-Process -FilePath $OllamaPath -ArgumentList 'serve' -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'ollama.stdout.log') -RedirectStandardError (Join-Path $logDir 'ollama.stderr.log')
        $deadline = (Get-Date).AddSeconds(45)
        while ((Get-Date) -lt $deadline) {
            if (Test-LocalOllama) { $ready = $true; break }
            $server.Refresh()
            if ($server.HasExited) { throw "Ollama exited; inspect $logDir\ollama.stderr.log." }
            Start-Sleep -Milliseconds 300
        }
        if (-not $ready) { throw "Ollama startup timed out; inspect $logDir\ollama.stderr.log." }
        Write-Output "Local Ollama started (PID $($server.Id)): $baseUrl"
    } else { Write-Output "Using Ollama already listening at $baseUrl" }
    if ($PullModel) {
        Write-Output "Pulling $Model into the local Ollama model store..."
        $pullBody = [System.Text.Encoding]::UTF8.GetBytes((@{ model = $Model; stream = $false } | ConvertTo-Json -Compress))
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            try {
                $pullRequest = [System.Net.WebRequest]::Create("$baseUrl/api/pull")
                $pullRequest.Proxy = $null
                $pullRequest.Method = 'POST'
                $pullRequest.ContentType = 'application/json'
                $pullRequest.Timeout = 600000
                $pullRequest.ContentLength = $pullBody.Length
                $stream = $pullRequest.GetRequestStream()
                try { $stream.Write($pullBody, 0, $pullBody.Length) } finally { $stream.Close() }
                $response = $pullRequest.GetResponse()
                $reader = [System.IO.StreamReader]::new($response.GetResponseStream())
                try { $result = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Close(); $response.Close() }
                if ($result.status -ne 'success') { throw "Unexpected pull result: $($result.status)" }
                Write-Output "Model ready: $Model"
                break
            } catch {
                if ($attempt -eq 3) { throw }
                # Windows antivirus can briefly hold downloaded blob files open.
                Start-Sleep -Seconds 2
            }
        }
    }
    Write-Output "OLLAMA_BASE_URL=$baseUrl"
} finally {
    foreach ($name in $previousEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process') }
}
