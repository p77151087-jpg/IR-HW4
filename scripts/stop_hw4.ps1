$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$recordPath = Join-Path $projectRoot 'reports\server.json'
if (-not (Test-Path -LiteralPath $recordPath)) { Write-Output 'No recorded HW4 background service.'; exit 0 }
$record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($record.process_id)" -ErrorAction SilentlyContinue
if (-not $process) { Write-Output 'Recorded HW4 process has already stopped.'; exit 0 }
$expectedApp = Join-Path $projectRoot 'app.py'
$expectedPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if ($process.ExecutablePath -ne $expectedPython -or -not $process.CommandLine.Contains($expectedApp) -or -not $process.CommandLine.Contains('streamlit')) {
    throw 'Process identity does not match this HW4 project; refusing to stop it.'
}
# Windows venv launchers keep a base-Python child which owns the listening socket.
# Stop only children whose full command line also identifies this exact app.
$children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($process.ProcessId)" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($expectedApp) -and $_.CommandLine.Contains('streamlit') })
foreach ($child in $children) { Stop-Process -Id $child.ProcessId }
if (Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue) { Stop-Process -Id $process.ProcessId }
Write-Output 'Stopped this HW4 background service.'
