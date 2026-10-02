param(
    [ValidateSet("cpu", "cu126", "cu128", "rocm6.4")]
    [string]$Backend = "cpu",
    [string]$Environment = "reinvent4"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root "third_party\reinvent4\repo"
$thirdParty = Join-Path $root "third_party\reinvent4"

conda env list | Out-Null
$existing = conda env list | Select-String ("^\s*" + [regex]::Escape($Environment) + "\s")
if (-not $existing) {
    conda create -y -n $Environment python=3.11
}

if (-not (Test-Path (Join-Path $source "install.py"))) {
    New-Item -ItemType Directory -Path $thirdParty -Force | Out-Null
    git clone --depth 1 https://github.com/MolecularAI/REINVENT4.git $source
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $source "install.py"))) {
        if (Test-Path $source) {
            Remove-Item -LiteralPath $source -Recurse -Force
        }
        $archive = Join-Path $thirdParty "reinvent4-main.zip"
        Invoke-WebRequest -Uri "https://codeload.github.com/MolecularAI/REINVENT4/zip/refs/heads/main" -OutFile $archive -UseBasicParsing
        Expand-Archive -LiteralPath $archive -DestinationPath $thirdParty -Force
        if (Test-Path (Join-Path $thirdParty "REINVENT4-main")) {
            Rename-Item -LiteralPath (Join-Path $thirdParty "REINVENT4-main") -NewName "repo"
        }
        Remove-Item -LiteralPath $archive -Force
    }
}

if (-not (Test-Path (Join-Path $source "install.py"))) {
    throw "REINVENT4 source is unavailable; check network access to GitHub and codeload.github.com."
}

conda run -n $Environment python (Join-Path $source "install.py") $Backend -d none
if ($LASTEXITCODE -ne 0) {
    throw "REINVENT4 dependency installation failed."
}
conda run -n $Environment python -m pip install -e $root
if ($LASTEXITCODE -ne 0) {
    throw "Installing this project into the REINVENT4 environment failed."
}
conda run -n $Environment reinvent --help | Select-Object -First 5
if ($LASTEXITCODE -ne 0) {
    throw "The reinvent executable is not available in the environment."
}
Write-Host "REINVENT4 environment '$Environment' is ready."
Write-Host "Use: conda run -n $Environment reinvent <config.toml>"
