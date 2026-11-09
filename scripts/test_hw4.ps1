param([string]$ReportPath = 'reports/pytest.xml')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
Push-Location -LiteralPath $projectRoot
try {
    & $projectPython -B -X utf8 -m pytest tests -q -p no:cacheprovider "--junitxml=$ReportPath"
    if ($LASTEXITCODE -ne 0) { throw 'HW4 tests failed.' }
} finally {
    Pop-Location
}
