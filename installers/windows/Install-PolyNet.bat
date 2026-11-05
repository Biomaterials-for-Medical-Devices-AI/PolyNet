@echo off
rem ===========================================================================
rem  PolyNet one-click installer for Windows.
rem
rem  Double-click this file. It needs no Python and no administrator rights:
rem  it downloads a private copy of Python, installs PolyNet into its own
rem  virtual environment and leaves PolyNet shortcuts on the Desktop and in
rem  the Start menu.
rem
rem  Everything lives in %LOCALAPPDATA%\PolyNet, so the "Uninstall PolyNet"
rem  shortcut (or -Uninstall below) removes all of it.
rem
rem  Advanced use, from a Command Prompt in this folder:
rem      Install-PolyNet.bat -Uninstall
rem      Install-PolyNet.bat -Version 1.0.5
rem      Install-PolyNet.bat -Check
rem      Install-PolyNet.bat -NoShortcut
rem      Install-PolyNet.bat -Icon C:\path\to\icon.png
rem
rem  This file is both a batch script and a PowerShell script: cmd.exe stops
rem  reading at the `exit /b` below, and the PowerShell part after the
rem  #:PS_START:# marker is what actually does the work. Keeping it in one
rem  file means there is only one thing to download and nothing to unzip.
rem ===========================================================================

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $t=[IO.File]::ReadAllText('%~f0'); $i=$t.LastIndexOf('#:PS_START:#'); & ([scriptblock]::Create($t.Substring($i+12)))" %*
if errorlevel 1 (
    echo.
    echo The installer did not finish. Please scroll up to see why.
    pause
)
exit /b

#:PS_START:#
param(
    [switch]$Uninstall,
    [switch]$Check,
    [switch]$NoShortcut,
    [string]$Version = "",
    [string]$Icon = ""
)

# Stop on cmdlet errors. Mind native commands: Windows PowerShell turns their
# stderr into error records whenever it is redirected (2>$null, 2>&1, or 2>file),
# and under "Stop" the first stderr line aborts the script even when the command
# succeeded. uv, pip and python all report progress on stderr, so never redirect
# their stderr here -- let it reach the console, or relax this preference around
# the call, as the cache prune in step 7 does.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# --------------------------------------------------------------- settings ---

$Root = $env:POLYNET_HOME
if (-not $Root) { $Root = Join-Path $env:LOCALAPPDATA "PolyNet" }

$PythonVersion = $env:POLYNET_PYTHON_VERSION
if (-not $PythonVersion) { $PythonVersion = "3.11" }

$UvVersion = $env:POLYNET_UV_VERSION
if (-not $UvVersion) { $UvVersion = "latest" }

$EnvDir      = Join-Path $Root "env"
$ToolsDir    = Join-Path $Root "tools"
$Launcher    = Join-Path $Root "PolyNet.cmd"
$Uninstaller = Join-Path $Root "Uninstall-PolyNet.cmd"
$IconPath    = Join-Path $Root "polynet.ico"
$Log         = Join-Path $Root "install.log"

$DesktopDir   = [Environment]::GetFolderPath("Desktop")
$StartMenuDir = Join-Path ([Environment]::GetFolderPath("Programs")) "PolyNet"

$PackageSpec = "polynet"
if ($Version) { $PackageSpec = "polynet==$Version" }

$IconSource = $Icon
if (-not $IconSource) { $IconSource = $env:POLYNET_ICON }

$script:StepNumber = 0
$script:TotalSteps = 7

# ---------------------------------------------------------------- output ----

function Write-Step([string]$Message) {
    $script:StepNumber++
    Write-Host ""
    Write-Host ("[{0}/{1}] {2}" -f $script:StepNumber, $script:TotalSteps, $Message) -ForegroundColor Cyan
}

function Write-Info([string]$Message) { Write-Host "      $Message" }
function Write-Warn([string]$Message) { Write-Host "      ! $Message" -ForegroundColor Yellow }

function Wait-ForUser {
    # The same guard as the macOS script's "[ -t 0 ]": when stdin is not a
    # console -- a CI run, or a piped invocation -- there is nobody to press
    # Enter, so do not stop for a keypress that will never come.
    if ([Console]::IsInputRedirected) { return }
    Write-Host ""
    Write-Host "  Press Enter to close this window." -NoNewline
    try { Read-Host | Out-Null } catch { }
}

function Invoke-Tool([string]$Exe, [string[]]$Arguments, [string]$What) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed (exit code $LASTEXITCODE)."
    }
}

# ------------------------------------------------------------- uninstall ----

if ($Uninstall) {
    Write-Host ""
    Write-Host "  Removing PolyNet" -ForegroundColor Cyan
    Write-Host ""
    foreach ($target in @(
        (Join-Path $DesktopDir "PolyNet.lnk"),
        $StartMenuDir,
        $Root
    )) {
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  removed $target"
        }
    }
    Write-Host ""
    Write-Host "  PolyNet has been removed." -ForegroundColor Green
    Write-Host "  Your experiments and data were not touched."
    Wait-ForUser
    exit 0
}

# ----------------------------------------------------------------- icons ----

# A .ico file may hold a PNG directly (Windows Vista and later), so a 256x256
# re-encoding of the PolyNet logo wrapped in an icon header is all that is
# needed -- no image tooling to install.
function New-IconFromPng([string]$PngPath, [string]$IcoPath) {
    Add-Type -AssemblyName System.Drawing
    $source = [System.Drawing.Image]::FromFile($PngPath)
    try {
        $bitmap = New-Object System.Drawing.Bitmap -ArgumentList 256, 256
        try {
            $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
            $graphics.InterpolationMode = "HighQualityBicubic"
            $graphics.Clear([System.Drawing.Color]::Transparent)
            $graphics.DrawImage($source, 0, 0, 256, 256)
            $graphics.Dispose()

            $memory = New-Object System.IO.MemoryStream
            $bitmap.Save($memory, [System.Drawing.Imaging.ImageFormat]::Png)
            $png = $memory.ToArray()
            $memory.Dispose()
        } finally { $bitmap.Dispose() }
    } finally { $source.Dispose() }

    # ICONDIR followed by one ICONDIRENTRY, then the PNG itself.
    $header = [byte[]]@(
        0, 0,        # reserved
        1, 0,        # type: icon
        1, 0,        # number of images
        0,           # width  (0 means 256)
        0,           # height (0 means 256)
        0,           # palette size
        0,           # reserved
        1, 0,        # colour planes
        32, 0        # bits per pixel
    )
    $size = [BitConverter]::GetBytes([uint32]$png.Length)
    $offset = [BitConverter]::GetBytes([uint32]22)
    [System.IO.File]::WriteAllBytes($IcoPath, [byte[]]($header + $size + $offset + $png))
}

function New-Shortcut([string]$Path, [string]$Target, [string]$Description) {
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($Path)
    $shortcut.TargetPath = $Target
    $shortcut.WorkingDirectory = $env:USERPROFILE
    $shortcut.Description = $Description
    if (Test-Path -LiteralPath $IconPath) { $shortcut.IconLocation = "$IconPath,0" }
    $shortcut.Save()
}

# ----------------------------------------------------------------- start ----

New-Item -ItemType Directory -Force -Path $Root, $ToolsDir | Out-Null
try { Start-Transcript -Path $Log -Force | Out-Null } catch { }

Write-Host ""
Write-Host "  PolyNet installer" -ForegroundColor Cyan
Write-Host "  polymer property prediction with graph neural networks" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  This will install PolyNet and its own private copy of Python into"
Write-Host "  $Root. Nothing else on this computer is changed,"
Write-Host "  and no administrator rights are needed."
Write-Host ""
Write-Host "  The download is a few gigabytes (PyTorch, RDKit and friends), so"
Write-Host "  this usually takes 5-15 minutes on a normal connection. You can"
Write-Host "  leave it running."

try {
    # PyPI has no Windows/ARM64 wheels for PyTorch, so on an ARM machine ask uv
    # for the x64 build of Python and let Windows emulate it: the x64 wheels
    # then install and run as they do on an Intel or AMD machine.
    $PythonRequest = $PythonVersion
    switch ($env:PROCESSOR_ARCHITECTURE) {
        "AMD64" { $UvTarget = "x86_64-pc-windows-msvc" }
        "ARM64" {
            $UvTarget = "aarch64-pc-windows-msvc"
            $PythonRequest = "cpython-$PythonVersion-windows-x86_64"
        }
        default { throw "Unsupported processor architecture: $env:PROCESSOR_ARCHITECTURE" }
    }

    # uv keeps its managed Python builds and its download cache inside $Root so
    # that uninstalling really removes everything.
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "python"
    $env:UV_CACHE_DIR = Join-Path $Root "cache"
    $env:UV_NO_MODIFY_PATH = "1"
    # Build the environment on the private Python even when the computer
    # already has a python3.11 on its PATH: PolyNet should not depend on an
    # interpreter it does not control.
    $env:UV_MANAGED_PYTHON = "1"

    # ------------------------------------------- 1. the package manager ----

    Write-Step "Fetching the installer tool (uv)"

    $uv = Join-Path $ToolsDir "uv.exe"
    if (Test-Path -LiteralPath $uv) {
        Write-Info "already present: $(& $uv --version)"
    } else {
        if ($UvVersion -eq "latest") {
            $uvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-$UvTarget.zip"
        } else {
            $uvUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-$UvTarget.zip"
        }
        Write-Info "downloading $uvUrl"
        $zip = Join-Path $env:TEMP "uv-polynet.zip"
        try {
            Invoke-WebRequest -Uri $uvUrl -OutFile $zip -UseBasicParsing
        } catch {
            throw "Could not download uv. Check your internet connection, then run this installer again. ($($_.Exception.Message))"
        }
        Expand-Archive -LiteralPath $zip -DestinationPath $ToolsDir -Force
        Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue

        if (-not (Test-Path -LiteralPath $uv)) {
            $found = Get-ChildItem -LiteralPath $ToolsDir -Recurse -Filter "uv.exe" |
                Select-Object -First 1
            if (-not $found) { throw "uv was downloaded but uv.exe was not found in $ToolsDir." }
            Copy-Item -LiteralPath $found.FullName -Destination $uv -Force
        }
        Write-Info "installed $(& $uv --version)"
    }

    # ------------------------------------------------------- 2. Python ----

    Write-Step "Installing Python $PythonVersion (a private copy, just for PolyNet)"
    # --no-bin keeps uv from adding a python3.11 shim to the user's bin
    # directory: PolyNet uses its own environment, and the shim would only
    # warn about PATH.
    Invoke-Tool $uv @("python", "install", "--no-bin", $PythonRequest) "Installing Python $PythonVersion"
    Write-Info "done"

    # -------------------------------------------------- 3. environment ----

    Write-Step "Creating PolyNet's environment"

    $python = Join-Path $EnvDir "Scripts\python.exe"
    if (Test-Path -LiteralPath $python) {
        Write-Info "reusing the existing environment in $EnvDir"
    } else {
        # --seed installs pip, which the GUI's "install PSMILES" button needs.
        Invoke-Tool $uv @("venv", "--seed", "--python", $PythonRequest, $EnvDir) "Creating the environment"
    }
    if (-not (Test-Path -LiteralPath $python)) { throw "No Python interpreter at $python." }

    # ----------------------------------------------------- 4. PolyNet ----

    Write-Step "Installing PolyNet and its dependencies (this is the long part)"
    Invoke-Tool $uv @("pip", "install", "--python", $python, "--upgrade", $PackageSpec) "Installing $PackageSpec"

    $installedVersion = (& $python -c "import polynet; print(polynet.__version__)")
    if ($LASTEXITCODE -ne 0) { throw "PolyNet was installed but cannot be imported." }
    Write-Info "installed PolyNet $installedVersion"

    # ------------------------------------------ 5. Visual C++ runtime ----

    # The PyTorch and XGBoost wheels are built with MSVC and need its
    # redistributable runtime, which is present on most machines but is not
    # part of a clean Windows install.
    Write-Step "Checking the Visual C++ runtime"

    $missing = @("vcruntime140.dll", "vcruntime140_1.dll") | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $env:SystemRoot "System32\$_"))
    }
    if ($missing) {
        Write-Warn "Microsoft's Visual C++ runtime is missing ($($missing -join ', '))."
        Write-Warn "PyTorch will not load without it. Install it from:"
        Write-Warn "  https://aka.ms/vs/17/release/vc_redist.x64.exe"
        Write-Warn "then run this installer again."
    } else {
        Write-Info "present"
    }

    # ------------------------------------------ 6. launcher and icons ----

    Write-Step "Creating the PolyNet shortcuts"

    @"
@echo off
rem Starts the PolyNet GUI. Created by the PolyNet installer.
title PolyNet
echo.
echo   ======================================================
echo     PolyNet is starting.
echo.
echo     Your web browser will open in a few seconds. The
echo     first start takes a little longer than the rest.
echo.
echo     KEEP THIS WINDOW OPEN while you use PolyNet.
echo     To stop PolyNet, close this window or press Ctrl-C.
echo   ======================================================
echo.
cd /d "%USERPROFILE%"
"$EnvDir\Scripts\polynet.exe" %*
if errorlevel 1 pause
"@ | Set-Content -LiteralPath $Launcher -Encoding ASCII

    @"
@echo off
rem Removes PolyNet. Created by the PolyNet installer.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Remove-Item -LiteralPath '$(Join-Path $DesktopDir 'PolyNet.lnk')','$StartMenuDir','$Root' -Recurse -Force -ErrorAction SilentlyContinue; Write-Host ''; Write-Host '  PolyNet has been removed. Your experiments and data were not touched.'; Write-Host ''; Read-Host '  Press Enter to close this window'"
"@ | Set-Content -LiteralPath $Uninstaller -Encoding ASCII

    # The icon comes from the logo that ships inside the installed package, so
    # it always matches the version that was installed. -Icon overrides it.
    $logo = & $python -c "import pathlib, polynet; print(pathlib.Path(polynet.__file__).parent / 'app' / 'static' / 'polynet.png')"
    if ($LASTEXITCODE -ne 0) { $logo = "" }
    if ($IconSource) {
        if (-not (Test-Path -LiteralPath $IconSource)) { throw "No such icon file: $IconSource" }
        $logo = $IconSource
        Write-Info "using the icon you supplied: $IconSource"
    }
    if ($logo -and (Test-Path -LiteralPath $logo)) {
        try {
            New-IconFromPng $logo $IconPath
            Write-Info "icon built from $(Split-Path -Leaf $logo)"
        } catch {
            Write-Warn "could not build the icon; the shortcuts will use the default one"
        }
    }

    New-Item -ItemType Directory -Force -Path $StartMenuDir | Out-Null
    New-Shortcut (Join-Path $StartMenuDir "PolyNet.lnk") $Launcher "Open the PolyNet GUI"
    New-Shortcut (Join-Path $StartMenuDir "Uninstall PolyNet.lnk") $Uninstaller "Remove PolyNet"
    Write-Info "added PolyNet to the Start menu"

    if (-not $NoShortcut) {
        New-Shortcut (Join-Path $DesktopDir "PolyNet.lnk") $Launcher "Open the PolyNet GUI"
        Write-Info "created a PolyNet shortcut on your Desktop"
    }

    # ------------------------------------------------------- 7. check ----

    Write-Step "Checking that everything works"

    $modules = "polynet,torch,rdkit,xgboost,sklearn,streamlit,torch_geometric"
    # Capture the output instead of letting it print straight through:
    # Start-Transcript does not record a child process's stdout, so the one
    # step most likely to need diagnosing was the one step missing from the
    # log. Assigning to a variable redirects stdout only, so it cannot trip
    # the stderr behaviour described above.
    $smoke = & $python -c "for m in '$modules'.split(','): __import__(m); print('      ' + m + ' ok')"
    $smokeExit = $LASTEXITCODE
    foreach ($line in $smoke) { Write-Host $line }
    if ($smokeExit -ne 0) {
        throw "PolyNet is installed but some of its parts do not load. See $Log for details."
    }

    if ($Check) {
        Write-Info "running polynet check (a short training run on synthetic data)"
        Invoke-Tool (Join-Path $EnvDir "Scripts\polynet.exe") @("check") "polynet check"
    }

    # Reclaim the download cache. Two things matter here, and getting either
    # wrong failed an install that had otherwise completely succeeded:
    #
    #   * uv reports progress on stderr ("Pruning cache at: ..."), so silence
    #     it at the source with --quiet. The "2>$null" that used to be here was
    #     meant to do that, but in Windows PowerShell redirecting a native
    #     command's stderr makes PowerShell wrap each line in an error record
    #     instead of discarding it -- and under $ErrorActionPreference = "Stop"
    #     the first line aborts the script, however well the command did.
    #   * pruning is optional tidying, so it must never end the run either way.
    try {
        $ErrorActionPreference = "Continue"
        & $uv cache prune --quiet | Out-Null
    } catch {
        Write-Info "could not prune the download cache (harmless)"
    } finally {
        $ErrorActionPreference = "Stop"
    }

    # ------------------------------------------------------- finished ----

    Write-Host ""
    Write-Host "  PolyNet $installedVersion is installed." -ForegroundColor Green
    Write-Host ""
    Write-Host "  To start it, double-click PolyNet on your Desktop, or find"
    Write-Host "  PolyNet in the Start menu."
    Write-Host ""
    Write-Host "  It is also available from a Command Prompt:" -ForegroundColor DarkGray
    Write-Host "    $EnvDir\Scripts\polynet.exe"
    Write-Host "  To remove it again: use 'Uninstall PolyNet' in the Start menu." -ForegroundColor DarkGray
    Write-Host "  Install log: $Log" -ForegroundColor DarkGray
    Wait-ForUser
} catch {
    Write-Host ""
    Write-Host "  Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  The full log is at: $Log"
    Write-Host "  Please send that file to the PolyNet maintainers if you need help."
    Wait-ForUser
    exit 1
} finally {
    try { Stop-Transcript | Out-Null } catch { }
}
