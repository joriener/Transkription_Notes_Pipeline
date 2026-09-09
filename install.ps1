# =============================================================
#  Transkription_Notes_Pipeline - install.ps1
#  PowerShell 7+ assumed.
#  One-shot dependency installer, merged from the Audio and Video
#  pipeline installers, plus V1.1 additions (Playwright, python-docx).
#
#  Usage:
#    .\install.ps1              # Core deps + CPU torch + whisperx
#    .\install.ps1 -GPU         # Core deps + CUDA torch + whisperx
#    .\install.ps1 -CoreOnly    # Core deps only (no torch/whisperx)
#    .\install.ps1 -Verify      # Verify installed packages without installing
# =============================================================
param(
    [switch]$GPU,
    [switch]$CoreOnly,
    [switch]$Verify
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Python = "C:\Python\Python311\python.exe"

function Write-Step { param([string]$Msg) Write-Host "`n=== $Msg ===" -ForegroundColor Cyan }
function Write-OK   { param([string]$Msg) Write-Host "  OK  $Msg" -ForegroundColor Green }
function Write-Fail { param([string]$Msg) Write-Host "  FAIL  $Msg" -ForegroundColor Red }

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Fail "Python not found at $Python"
    Write-Host "Edit the `$Python variable in install.ps1 if your path differs." -ForegroundColor Yellow
    exit 1
}
$version = & $Python --version 2>&1
Write-Host "Using: $Python ($version)" -ForegroundColor White

# --- Verify mode ---
if ($Verify) {
    Write-Step "Verifying installed packages"
    $checks = @(
        @{ Module = "imagehash";   Label = "ImageHash" },
        @{ Module = "PIL";         Label = "Pillow" },
        @{ Module = "numpy";       Label = "numpy" },
        @{ Module = "scipy";       Label = "scipy" },
        @{ Module = "pywt";        Label = "PyWavelets" },
        @{ Module = "requests";    Label = "requests" },
        @{ Module = "whisperx";    Label = "whisperx" },
        @{ Module = "torch";       Label = "torch" },
        @{ Module = "torchaudio";  Label = "torchaudio" },
        @{ Module = "docx";        Label = "python-docx" }
    )
    $allOK = $true
    foreach ($c in $checks) {
        & $Python -c "import $($c.Module)" 2>$null
        if ($LASTEXITCODE -eq 0) { Write-OK $c.Label }
        else { Write-Fail "$($c.Label) -- not found"; $allOK = $false }
    }
    & $Python -c "import tkinter" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-OK "tkinter (GUI)" }
    else { Write-Fail "tkinter -- not found (GUI unavailable, CLI still works)" }
    & $Python -c "import playwright" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-OK "playwright (PDF export, preferred backend)" }
    else { Write-Fail "playwright -- not found (falls back to weasyprint/pdfkit for PDF)" }
    & $Python -c "import weasyprint" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-OK "weasyprint (PDF export fallback)" }
    else { Write-Fail "weasyprint -- not found (PDF export unavailable if Playwright missing too)" }
    & $Python -c "import torch; print('  CUDA available:', torch.cuda.is_available())" 2>$null
    $ffmpegBinDir = Join-Path $PSScriptRoot "ffmpegin"
    foreach ($exe in @("ffmpeg.exe", "ffplay.exe", "ffprobe.exe")) {
        if (Test-Path -LiteralPath (Join-Path $ffmpegBinDir $exe)) { Write-OK "$exe (bundled)" }
        elseif (Get-Command ($exe -replace '\.exe$', '') -ErrorAction SilentlyContinue) { Write-OK "$exe (on PATH)" }
        else { Write-Fail "$exe -- not bundled and not on PATH"; $allOK = $false }
    }
    if ($allOK) { Write-Host "`nAll core packages verified." -ForegroundColor Green }
    else { Write-Host "`nSome packages missing. Run .\install.ps1 to fix." -ForegroundColor Yellow }
    exit 0
}

# --- Step 1: core dependencies ---
Write-Step "Installing core dependencies"
& $Python -m pip install --upgrade pip
& $Python -m pip install `
    "Pillow>=10.4.0" `
    "ImageHash==4.3.2" `
    "numpy>=1.26.0" `
    "scipy>=1.12.0" `
    "PyWavelets>=1.6.0" `
    "requests>=2.32.0" `
    "python-docx>=1.1.0"
if ($LASTEXITCODE -ne 0) { Write-Fail "Core dependency install failed."; exit 1 }
Write-OK "Core dependencies installed."

Write-Step "Verifying core imports"
& $Python -c "import imagehash, PIL, numpy, scipy, pywt, requests, docx; print('Core imports OK')"
if ($LASTEXITCODE -ne 0) { Write-Fail "Core import check failed."; exit 1 }
Write-OK "Core imports verified."

if ($CoreOnly) {
    Write-Host "`nDone (core only). Skipping torch, whisperx, and Playwright." -ForegroundColor Yellow
    Write-Host "Run .\install.ps1 when ready to add transcription support." -ForegroundColor Yellow
    exit 0
}

# --- Step 2: torch ---
Write-Step "Installing torch + torchaudio"
if ($GPU) {
    Write-Host "  GPU mode: installing CUDA 12.8 variant." -ForegroundColor Yellow
    Write-Host "  Ensure CUDA 12.8 toolkit is installed: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    & $Python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "  CPU mode (default). Use -GPU flag for CUDA variant." -ForegroundColor Yellow
    & $Python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { Write-Fail "torch install failed."; exit 1 }
Write-OK "torch installed."

# --- Step 3: whisperx ---
Write-Step "Installing whisperx 3.8.6"
& $Python -m pip install "whisperx==3.8.6"
if ($LASTEXITCODE -ne 0) { Write-Fail "whisperx install failed."; exit 1 }
Write-OK "whisperx installed."

# --- Step 3b: torchcodec fix ---
# whisperx 3.8.5 downgrades torchcodec to 0.7.0 which breaks DLL loading on
# Windows with PyTorch 2.8 + CUDA 12.8. Restore the compatible version.
# Non-fatal: whisperx falls back to PyAV for audio decoding if this fails.
Write-Step "Restoring torchcodec 0.10.0 (whisperx compatibility fix)"
if ($GPU) {
    & $Python -m pip install "torchcodec==0.10.0" --index-url https://download.pytorch.org/whl/cu128
} else {
    & $Python -m pip install "torchcodec==0.10.0" --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "  torchcodec restore failed (non-fatal). whisperx will use PyAV for audio." -ForegroundColor Yellow
} else {
    Write-OK "torchcodec 0.10.0 restored."
}

# --- Step 4: Playwright (preferred PDF backend) ---
# Recommended over weasyprint on Windows: no GTK3 native dependency, so it
# does not collide with Tesseract-OCR's bundled libgobject-2.0-0.dll (the
# cause of "cannot load library ...: error 0x7e" if you have Tesseract-OCR
# installed for other projects, e.g. DocsSorter or Reisekosten_Scans).
Write-Step "Installing Playwright (PDF export, preferred backend)"
& $Python -m pip install "playwright>=1.45.0"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Playwright install failed (non-fatal). Falling back to weasyprint/pdfkit for PDF." -ForegroundColor Yellow
} else {
    Write-OK "Playwright installed."
    Write-Step "Downloading Chromium for Playwright (~300 MB, one-time)"
    & $Python -m playwright install chromium
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Chromium download failed (non-fatal). Falling back to weasyprint/pdfkit for PDF." -ForegroundColor Yellow
    } else {
        Write-OK "Chromium installed. PDF export will use Playwright."
    }
}

# --- Step 5: weasyprint (PDF export fallback) ---
Write-Step "Installing weasyprint (PDF export fallback)"
& $Python -m pip install "weasyprint>=62.0"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  weasyprint install failed (non-fatal). PDF export still works via Playwright if installed above." -ForegroundColor Yellow
} else {
    Write-OK "weasyprint installed."
}

# --- Step 5b: bundled ffmpeg/ffplay/ffprobe ---
# config.py's _BUNDLED_FFMPEG_DIR prefers <repo>fmpegin\ over PATH, so the
# pipeline needs no system-wide ffmpeg install and no PATH edit. The three .exe
# files are ~87 MB each and the gyan.dev essentials build is GPL v3, so they are
# gitignored and fetched here instead of being stored in the repository.
# Non-fatal: config.get_ffmpeg_path() falls back to ffmpeg on PATH, then to None.
$FfmpegVersion = "7.1"
$FfmpegUrl     = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-$FfmpegVersion-essentials_build.zip"
$FfmpegBinDir  = Join-Path $PSScriptRoot "ffmpegin"
$FfmpegExes    = @("ffmpeg.exe", "ffplay.exe", "ffprobe.exe")

Write-Step "Checking bundled ffmpeg"
$missingExes = @($FfmpegExes | Where-Object { -not (Test-Path -LiteralPath (Join-Path $FfmpegBinDir $_)) })
if ($missingExes.Count -eq 0) {
    Write-OK "ffmpeg, ffplay and ffprobe already in ffmpegin - skipping download."
} else {
    Write-Host "  Missing: $($missingExes -join ', ')" -ForegroundColor Yellow
    Write-Host "  Downloading ffmpeg $FfmpegVersion essentials (~80 MB) from gyan.dev..." -ForegroundColor Yellow
    $tmpZip = Join-Path $env:TEMP "ffmpeg-$FfmpegVersion-essentials_build.zip"
    $tmpDir = Join-Path $env:TEMP "ffmpeg-extract-$PID"
    try {
        Invoke-WebRequest -Uri $FfmpegUrl -OutFile $tmpZip -UseBasicParsing
        Expand-Archive -LiteralPath $tmpZip -DestinationPath $tmpDir -Force
        New-Item -ItemType Directory -Force -Path $FfmpegBinDir | Out-Null
        # The archive holds one top-level folder, e.g. ffmpeg-7.1-essentials_buildin\.
        $srcBin = Get-ChildItem -LiteralPath $tmpDir -Directory |
                  ForEach-Object { Join-Path $_.FullName "bin" } |
                  Where-Object { Test-Path -LiteralPath $_ } |
                  Select-Object -First 1
        if (-not $srcBin) { throw "No bin\ folder found inside the downloaded archive." }
        foreach ($exe in $FfmpegExes) {
            $src = Join-Path $srcBin $exe
            if (Test-Path -LiteralPath $src) {
                Copy-Item -LiteralPath $src -Destination (Join-Path $FfmpegBinDir $exe) -Force
                Write-OK $exe
            } else {
                Write-Host "  $exe not in the archive - skipped." -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host "  ffmpeg download failed (non-fatal): $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "  The pipeline will use ffmpeg from PATH instead, if present." -ForegroundColor Yellow
        Write-Host "  To install manually:" -ForegroundColor Yellow
        Write-Host "    1. Download $FfmpegUrl" -ForegroundColor Yellow
        Write-Host "    2. Copy binfmpeg.exe, ffplay.exe and ffprobe.exe into:" -ForegroundColor Yellow
        Write-Host "       $FfmpegBinDir" -ForegroundColor Yellow
    } finally {
        Remove-Item -LiteralPath $tmpZip -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# --- Step 6: tkinter check (GUI) ---
Write-Step "Checking tkinter (GUI)"
& $Python -c "import tkinter" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-OK "tkinter available - GUI (gui.py / --gui) will work."
} else {
    Write-Host "  tkinter not found. Reinstall Python from python.org with the 'tcl/tk' option enabled." -ForegroundColor Yellow
    Write-Host "  The CLI (run_pipeline.py) works without it." -ForegroundColor Yellow
}

# --- Final verification ---
Write-Step "Final import check"
& $Python -c "
import imagehash, PIL, numpy, scipy, pywt, requests, docx
import torch, whisperx
print(f'torch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}')
print('All imports OK')
"
if ($LASTEXITCODE -ne 0) { Write-Fail "Final import check failed."; exit 1 }

Write-Host "`n=== Transkription_Notes_Pipeline install complete ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Copy keys.cfg.example to keys.cfg and fill in your values" -ForegroundColor White
Write-Host "     (or use the Settings tab in the GUI once launched)" -ForegroundColor White
Write-Host "  2. Copy vocabulary.py.example to vocabulary.py and add your terms" -ForegroundColor White
Write-Host "  3. Launch the GUI:" -ForegroundColor White
Write-Host "     C:\Python\Python311\python.exe run_pipeline.py --gui" -ForegroundColor White
Write-Host "  4. Or run from the command line:" -ForegroundColor White
Write-Host "     C:\Python\Python311\python.exe run_pipeline.py --no-summary your_file.mp4" -ForegroundColor White
Write-Host "  5. Verify the transcript log:" -ForegroundColor White
Write-Host "     C:\Python\Python311\python.exe db.py" -ForegroundColor White
