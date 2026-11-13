param([switch]$QualityReview)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath 'data/test_2/annotations.csv')) {
    throw 'Run .venv/Scripts/python.exe -m balltrack.two_clip prepare first.'
}
if ($QualityReview) {
    if (-not (Test-Path -LiteralPath 'data/test_2/quality_review.md')) {
        throw 'Complete the first review, then run .venv/Scripts/python.exe -m balltrack.two_clip quality-plan.'
    }
    & $python -m balltrack annotate --video data/test_2/denmark_wales_clip_1min.mp4 --output data/test_2/annotations.csv --stride 1 --frames-file data/test_2/quality_review.md
} else {
    & $python -m balltrack annotate --video data/test_2/denmark_wales_clip_1min.mp4 --output data/test_2/annotations.csv --stride 1 --pending-only
}
if ($LASTEXITCODE -ne 0) { throw 'Annotation tool failed; saved annotations are preserved.' }
