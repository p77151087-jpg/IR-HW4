param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $projectPython)) {
        & $Python -m venv (Join-Path $projectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create HW4 virtual environment.' }
    }
    $lock = Join-Path $projectRoot 'requirements-lock.txt'
    if (-not (Test-Path -LiteralPath $lock)) { $lock = Join-Path $projectRoot 'requirements.txt' }
    & $projectPython -m pip install -r $lock
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install HW4 dependencies.' }
    & $projectPython -B -X utf8 -m ir_hw4.cli prepare
    if ($LASTEXITCODE -ne 0) { throw 'Failed to prepare Reuters.' }
    $modelManifestPath = 'artifacts\reuters-modapte-v1\manifest.json'
    $dataManifestPath = 'data\reuters-21578\manifest.json'
    $referenceManifestPath = 'resources\reuters-21578-manifest.json'
    if ((Test-Path -LiteralPath $modelManifestPath) -and (Test-Path -LiteralPath $referenceManifestPath)) {
        $modelManifest = Get-Content -LiteralPath $modelManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $referenceHash = (Get-FileHash -LiteralPath $referenceManifestPath -Algorithm SHA256).Hash
        $dataHash = (Get-FileHash -LiteralPath $dataManifestPath -Algorithm SHA256).Hash
        if ($modelManifest.dataset_manifest_sha256 -eq $referenceHash -and $dataHash -ne $referenceHash) {
            $referenceData = Get-Content -LiteralPath $referenceManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $preparedData = Get-Content -LiteralPath $dataManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
            # Reuse the reference only when the timestamp is the sole difference.
            $referenceData.PSObject.Properties.Remove('downloaded_at_utc')
            $preparedData.PSObject.Properties.Remove('downloaded_at_utc')
            if (($preparedData | ConvertTo-Json -Depth 10 -Compress) -cne ($referenceData | ConvertTo-Json -Depth 10 -Compress)) {
                throw 'The prepared Reuters data does not match the published model inputs.'
            }
            Copy-Item -LiteralPath $referenceManifestPath -Destination $dataManifestPath
        }
    }
    if (-not (Test-Path -LiteralPath $modelManifestPath)) {
        & $projectPython -B -X utf8 -m ir_hw4.cli train
        if ($LASTEXITCODE -ne 0) { throw 'Word2Vec training failed.' }
    }
    if (-not (Test-Path -LiteralPath 'reports\evaluation-v1')) {
        & $projectPython -B -X utf8 -m ir_hw4.cli evaluate
        if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed.' }
    }
} finally {
    Pop-Location
}
