$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = '..\lab2\.venv\Scripts\python.exe'
foreach ($split in @('dev','holdout')) {
    if (-not (Test-Path "results/${split}_rankings.jsonl")) {
        & $python -u search.py $split
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & $python -u generate.py $split
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
