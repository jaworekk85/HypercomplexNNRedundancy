param(
    [string]$Tectonic = "",
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
$paperRoot = $PSScriptRoot
$repoRoot = Split-Path -Parent $paperRoot
if (-not $Python) {
    $Python = Join-Path $repoRoot '.venv-cuda/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
}
& $Python (Join-Path $paperRoot 'prepare.py')
if ($LASTEXITCODE -ne 0) { throw 'Manuscript data preparation failed.' }
& $Python (Join-Path $paperRoot 'plot_overview.py')
if ($LASTEXITCODE -ne 0) { throw 'Manuscript figure preparation failed.' }
& $Python (Join-Path $paperRoot 'plot_grouping_concept.py')
if ($LASTEXITCODE -ne 0) { throw 'Concept figure preparation failed.' }
if (-not $Tectonic) {
    $localCompiler = Join-Path $repoRoot '.cache/paper-tooling/tectonic/tectonic.exe'
    if (Test-Path -LiteralPath $localCompiler) { $Tectonic = $localCompiler }
    elseif (Get-Command tectonic -ErrorAction SilentlyContinue) { $Tectonic = 'tectonic' }
    else { throw 'Install Tectonic or use -Tectonic <path>. See paper/README.md.' }
}
$previousCache = $env:TECTONIC_CACHE_DIR
try {
    $env:TECTONIC_CACHE_DIR = Join-Path $repoRoot '.cache/paper-tooling/tex-cache'
    New-Item -ItemType Directory -Force -Path (Join-Path $paperRoot 'build') | Out-Null
    Push-Location $paperRoot
    try {
        & $Tectonic --untrusted --keep-logs --keep-intermediates --outdir build main.tex
        if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed.' }
        Copy-Item -LiteralPath 'build/main.pdf' -Destination 'draft.pdf' -Force
    }
    finally { Pop-Location }
}
finally { $env:TECTONIC_CACHE_DIR = $previousCache }
