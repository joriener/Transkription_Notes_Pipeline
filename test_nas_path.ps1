# =============================================================
#  Transkription_Notes_Pipeline - test_nas_path.ps1
#  PowerShell 7+ assumed.
#
#  Standalone NAS/UNC-path smoke test. Run this on your machine
#  (not executable from the sandbox that generated this project:
#  \\192.168.0.104 is a private LAN address, unreachable from
#  outside your network). Checks the specific failure points that
#  matter for this pipeline before you point --output-dir, a
#  source file, or db_path at the NAS.
#
#  Usage:
#    .\test_nas_path.ps1
#    .\test_nas_path.ps1 -UncPath "\\192.168.0.104\Public\Agilent\02_GC"
#    .\test_nas_path.ps1 -SampleVideo "\\192.168.0.104\Public\Agilent\02_GC\sample.mp4"
# =============================================================
param(
    [string]$UncPath = "\\192.168.0.104\Public\Agilent",
    [string]$SampleVideo = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
$Python = "C:\Python\Python311\python.exe"

function Write-Step { param([string]$Msg) Write-Host "`n=== $Msg ===" -ForegroundColor Cyan }
function Write-OK   { param([string]$Msg) Write-Host "  OK    $Msg" -ForegroundColor Green }
function Write-Fail { param([string]$Msg) Write-Host "  FAIL  $Msg" -ForegroundColor Red }
function Write-Warn { param([string]$Msg) Write-Host "  WARN  $Msg" -ForegroundColor Yellow }

Write-Host "Transkription_Notes_Pipeline - NAS/UNC path check" -ForegroundColor White
Write-Host "Target: $UncPath" -ForegroundColor White

# --- 1. Reachability ---
Write-Step "1. Share reachability"
if (Test-Path -LiteralPath $UncPath) {
    Write-OK "Path reachable: $UncPath"
} else {
    Write-Fail "Cannot reach $UncPath. Check VPN/network, share permissions, and that the path is spelled correctly."
    Write-Host "  Remaining checks will likely fail too; fix reachability first." -ForegroundColor Yellow
}

# --- 2. Write access (needed for --output-dir on the share) ---
Write-Step "2. Write access"
$testFile = Join-Path $UncPath "_pipeline_write_test.tmp"
try {
    "test" | Out-File -LiteralPath $testFile -Encoding utf8 -ErrorAction Stop
    Remove-Item -LiteralPath $testFile -ErrorAction Stop
    Write-OK "Write + delete succeeded on $UncPath"
} catch {
    Write-Fail "Write test failed: $($_.Exception.Message)"
    Write-Host "  If you only need to READ source files from the NAS and write outputs" -ForegroundColor Yellow
    Write-Host "  locally, this is fine: leave --output-dir unset or point it at a local folder." -ForegroundColor Yellow
}

# --- 3. Python pathlib round-trip on the UNC path ---
Write-Step "3. Python pathlib handling (config.py / run_pipeline.py path logic)"
if (Test-Path -LiteralPath $Python) {
    $pyTest = @"
from pathlib import Path
p = Path(r'$UncPath')
print('parent:', p.parent)
print('exists:', p.exists())
stem_test = p / 'sample.mp4'
print('stem:', stem_test.stem)
print('as prefix:', str(p / 'sample'))
"@
    $pyTest | & $Python -
    Write-OK "pathlib handled the UNC path without raising (see output above)."
} else {
    Write-Warn "Python not found at $Python; skipped pathlib check."
}

# --- 4. ffmpeg/ffprobe on a real file (only if -SampleVideo given) ---
Write-Step "4. ffmpeg/ffprobe on a NAS file"
if ($SampleVideo -eq "") {
    Write-Warn "No -SampleVideo given; skipped. Re-run with -SampleVideo pointing at a real "
    Write-Warn "file on the share to check frame extraction, e.g.:"
    Write-Warn "  .\test_nas_path.ps1 -SampleVideo `"$UncPath\02_GC\some_webinar.mp4`""
} elseif (-not (Test-Path -LiteralPath $SampleVideo)) {
    Write-Fail "Sample file not found: $SampleVideo"
} else {
    $ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
    if (-not $ffprobe) {
        Write-Fail "ffprobe not found in PATH. Install ffmpeg first (see requirements.txt)."
    } else {
        $duration = & ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 -- "$SampleVideo" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-OK "ffprobe read the file directly from the share (duration: $duration sec)."
        } else {
            Write-Fail "ffprobe could not read the file directly from the UNC path: $duration"
            Write-Host "  Known workaround for some ffmpeg builds: map the share to a drive letter" -ForegroundColor Yellow
            Write-Host "  first (net use Z: $UncPath) and use Z:\... instead of the UNC form." -ForegroundColor Yellow
        }
    }
}

# --- 5. SQLite over the network share (important caveat, not just a check) ---
Write-Step "5. SQLite database location"
Write-Warn "Do NOT point db_path (config.py) at a NAS/UNC path."
Write-Warn "SQLite's WAL journal mode (used by db.py for concurrent-safe writes) is"
Write-Warn "explicitly documented as unreliable over network filesystems/SMB shares:"
Write-Warn "https://www.sqlite.org/wal.html (see 'the WAL file needs shared memory')."
Write-Warn "Keep transkription_notes_pipeline.db on local disk (the default, D:\Claude\...)."
Write-Warn "Source videos and --output-dir on the NAS are fine; only the .db file should stay local."

Write-Host "`n=== Summary ===" -ForegroundColor White
Write-Host "If steps 1-4 passed: reading source files from $UncPath and writing outputs" -ForegroundColor White
Write-Host "there (--output-dir) should work normally. Keep the SQLite database local regardless." -ForegroundColor White
