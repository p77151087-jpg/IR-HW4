param([switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$projectApp = Join-Path $projectRoot 'app.py'
$serviceRecord = Join-Path $projectRoot 'reports\server.json'
$listener = Get-NetTCPConnection -LocalPort 8504 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $running = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener[0].OwningProcess)"
    if (-not $running.CommandLine.Contains($projectApp)) {
        throw 'Port 8504 is already used by another application. No process was stopped.'
    }
} else {
    $process = Start-Process -FilePath $projectPython -ArgumentList @('-B', '-m', 'streamlit', 'run', $projectApp, '--server.address', '127.0.0.1', '--server.port', '8504') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectRoot 'reports\server.stdout.log') -RedirectStandardError (Join-Path $projectRoot 'reports\server.stderr.log') -PassThru
    @{ process_id = $process.Id; project = $projectRoot; url = 'http://127.0.0.1:8504' } | ConvertTo-Json | Set-Content -LiteralPath $serviceRecord -Encoding UTF8
}
$ready = $false
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    try {
        $response = Invoke-WebRequest 'http://127.0.0.1:8504/_stcore/health' -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) { throw 'HW4 did not become ready. Check reports\server.stderr.log.' }
Write-Output 'HW4 ready: http://127.0.0.1:8504'
if ($OpenBrowser) { Start-Process 'http://127.0.0.1:8504' }
