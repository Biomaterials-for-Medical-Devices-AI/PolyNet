# One-click installers

These installers set up PolyNet on a computer that has **no Python at all**.
They download a private copy of Python, install PolyNet into its own virtual
environment and leave a clickable PolyNet icon behind. No administrator
password, no Homebrew, no conda, and nothing else on the computer is changed.

| Platform | File | Installs to |
|---|---|---|
| macOS (Apple silicon and Intel) | [`macos/Install-PolyNet.command`](macos/Install-PolyNet.command) | `~/.polynet` |
| Windows 10/11 (x64 and ARM64) | [`windows/Install-PolyNet.bat`](windows/Install-PolyNet.bat) | `%LOCALAPPDATA%\PolyNet` |
| Linux | [`macos/Install-PolyNet.command`](macos/Install-PolyNet.command) | `~/.polynet` |

Allow **5–15 minutes** and about **4 GB of disk space**: PyTorch, RDKit, Ray
and the other dependencies are large downloads.

---

## For users

### macOS

1. Download **Install-PolyNet.command**.
2. **Control-click** (or right-click) the downloaded file and choose **Open**,
   then **Open** again in the dialog that appears.

   The extra step is needed because the file is not signed with an Apple
   Developer certificate, so double-clicking it shows *"cannot be opened
   because it is from an unidentified developer"*. If no **Open** button
   appears, go to  **System Settings → Privacy & Security**, scroll to the
   message about Install-PolyNet.command and click **Open Anyway**.

   To skip Gatekeeper entirely, paste this into the Terminal app instead —
   files fetched with `curl` are not quarantined:

   ```bash
   curl -fsSLO https://raw.githubusercontent.com/Biomaterials-for-Medical-Devices-AI/PolyNet/main/installers/macos/Install-PolyNet.command && bash Install-PolyNet.command
   ```

3. A Terminal window reports progress. When it finishes, **PolyNet** is on the
   Desktop, in the Applications folder and in Launchpad.

### Windows

1. Download **Install-PolyNet.bat**.
2. Double-click it. If Windows SmartScreen warns about an unrecognised file,
   choose **More info → Run anyway**.
3. A Command Prompt window reports progress. When it finishes, **PolyNet** is
   on the Desktop and in the Start menu.

### Linux

Download `Install-PolyNet.command`, then run `bash Install-PolyNet.command`.
PolyNet is added to the applications menu.

### Using PolyNet

Click the PolyNet icon. A terminal window opens, the browser follows a few
seconds later, and the GUI runs there. **Keep that terminal window open while
using PolyNet** — closing it stops the app.

### Removing PolyNet

- **macOS / Linux** — open `~/.polynet/Uninstall-PolyNet.command`, or run the
  installer again with `--uninstall`.
- **Windows** — use **Uninstall PolyNet** in the Start menu, or run the
  installer again with `-Uninstall`.

Either way, experiments and data saved outside the install directory are left
alone.

---

## What the installers actually do

Both follow the same seven steps, and both are safe to run again — a second
run upgrades PolyNet in place.

1. Download [uv](https://github.com/astral-sh/uv) (a single static binary)
   from its GitHub release into the install directory.
2. `uv python install --no-bin 3.11` — a private, relocatable CPython build.
3. `uv venv --seed` — the environment. `--seed` provides `pip`, which the
   GUI's *install PSMILES* button needs.
4. `uv pip install --upgrade polynet` from PyPI.
5. Platform fix-ups (see below).
6. A launcher script, an icon built from `polynet/app/static/polynet.png`, and
   the shortcuts.
7. Import every heavy dependency to prove the install works. With `--check` /
   `-Check` it also runs `polynet check`, a short training run on synthetic
   data.

uv's managed Python builds and its download cache are redirected into the
install directory (`UV_PYTHON_INSTALL_DIR`, `UV_CACHE_DIR`) so that
uninstalling removes everything.

### Platform fix-ups

**macOS — OpenMP.** The XGBoost wheel records a single search path for
`libomp.dylib`, Homebrew's `/opt/homebrew/opt/libomp/lib`, so `import xgboost`
fails on a machine without Homebrew — which is the whole point of this
installer. PyTorch ships a copy of that library under the same install name,
so step 5 copies it to `site-packages/polynet_runtime/libomp.dylib` and adds a
`.pth` file that loads it at interpreter start-up. XGBoost's lookup is then
satisfied by the already-loaded image, and the process ends up with exactly
one OpenMP runtime — which is what PyTorch and XGBoost need in order to
coexist (see the `torch` version pins in `pyproject.toml`).

**Windows — Visual C++ runtime.** The PyTorch and XGBoost wheels need
Microsoft's redistributable runtime, which is on most machines but is not part
of a clean Windows install and cannot be installed without administrator
rights. The installer checks for it and prints the download link if it is
missing.

**Windows — never redirect a native command's stderr.** Windows PowerShell
wraps each line a native program writes to stderr in an error record, but only
when that stderr is redirected (`2>$null`, `2>&1`, `2>file`); left alone it
passes straight through to the console. The script runs under
`$ErrorActionPreference = "Stop"`, which promotes that record to a terminating
error, so one informational line from a command that exited 0 is enough to fail
the install. This is exactly how a real install died at step 7 of 7: the line
was `& $uv cache prune 2>$null`, written to hide uv's `Pruning cache at: ...`
progress message, and the redirection meant to silence it is what made it
fatal. Silence such output at the source instead (`uv ... --quiet`), and if a
command must be redirected, relax the preference around the call and check
`$LASTEXITCODE` yourself.

**Windows — the transcript misses child stdout.** `Start-Transcript` records
PowerShell's own streams and a native program's stderr, but not a child
process's stdout. That is why uv's progress appears in the install log while
the import check's `... ok` lines did not. Where the log needs that output, the
script captures it into a variable and re-emits it with `Write-Host`.

### The clickable icon

The icon is built from `polynet/app/static/polynet.png` — the logo, taken from
the *installed package* rather than from this repository, so it always matches
the version that was installed. To use a different image, pass `--icon
my-logo.png` (macOS) or `-Icon my-logo.png` (Windows). Any square PNG works;
512×512 or larger gives the sharpest result.

**macOS** builds `~/Applications/PolyNet.app` with `osacompile`, an AppleScript
applet that runs `open -a Terminal <launcher>`. Three things matter here:

- the bundle is compiled *on the user's machine*, so Gatekeeper does not
  quarantine it — unlike a downloaded `.app`, it just opens;
- an applet is a real app bundle, so it takes a custom `.icns` icon (built
  from the logo with `sips` and `iconutil`) and appears in Launchpad and
  Spotlight;
- `open -a Terminal` rather than `tell application "Terminal"` avoids the
  *"PolyNet wants to control Terminal"* automation prompt.

One trap is worth knowing about. `osacompile` writes **two** icon keys into
`Info.plist`: `CFBundleIconFile`, pointing at `Resources/applet.icns`, and
`CFBundleIconName`, pointing at the stock AppleScript applet artwork inside
`Resources/Assets.car`. macOS prefers the asset catalogue, so replacing
`applet.icns` alone changes nothing on screen — Finder keeps drawing the grey
applet-on-a-pedestal icon. The installer therefore also removes
`CFBundleIconName` and deletes `Assets.car`. Because `osacompile` ad-hoc signs
the bundle and editing `Resources/` invalidates that signature, it then
re-signs with `codesign --force --sign -`, and nudges LaunchServices with
`lsregister -f` so a reinstall does not show the previous icon from cache.

The Desktop item is a **copy** of that bundle, not a symlink: Finder draws a
symlink with a generic document icon, so the logo would not show on the
Desktop, which is exactly where most people look for it. Both copies run the
same launcher from `~/.polynet`, and the uninstaller removes both.

**Windows** writes `PolyNet.cmd` and `.lnk` shortcuts via `WScript.Shell`. The
icon is a 256×256 PNG wrapped in an ICO header, which Windows has accepted
since Vista — no image tooling needed.

---

## For maintainers

### Distributing them

Attach both files to the GitHub release:

```bash
gh release upload v1.0.5 \
  installers/macos/Install-PolyNet.command \
  installers/windows/Install-PolyNet.bat
```

The installers always install the latest PolyNet from PyPI, so they do not
need to be rebuilt for a new release — only re-uploaded if the installers
themselves changed. To pin a version, users pass `--version 1.0.5` (macOS) or
`-Version 1.0.5` (Windows).

### Options

| macOS / Linux | Windows | Effect |
|---|---|---|
| `--uninstall` | `-Uninstall` | Remove PolyNet, its Python and its icons |
| `--version X.Y.Z` | `-Version X.Y.Z` | Install that version instead of the latest |
| `--check` | `-Check` | Also run `polynet check` at the end |
| `--icon FILE` | `-Icon FILE` | Build the icon from this image instead of the logo |
| `--no-shortcut` | `-NoShortcut` | Do not put PolyNet on the Desktop |
| `--help` | — | Usage |

Environment overrides, both platforms: `POLYNET_HOME`, `POLYNET_ICON`,
`POLYNET_PYTHON_VERSION`, `POLYNET_UV_VERSION`.

### Why Windows is a single `.bat`

`Install-PolyNet.bat` is both a batch file and a PowerShell script: `cmd.exe`
stops at `exit /b`, and the bootstrap line re-reads the file and runs
everything after the `#:PS_START:#` marker as PowerShell. Keeping it in one
file means there is one thing to download and nothing to unzip — a `.bat` plus
a separate `.ps1` breaks when someone double-clicks the `.bat` straight out of
a zip viewer.

### Known limitations

- **Neither installer is code-signed.** Users have to click past Gatekeeper on
  macOS and SmartScreen on Windows. Fixing this properly means an Apple
  Developer ID (`codesign` + `notarytool`, a signed `.pkg`) and an
  Authenticode certificate for Windows.
- The GUI's *install PSMILES* button needs `git` on the machine, because
  `canonicalize-psmiles` is installed from a git URL. On a Mac without the
  Xcode command line tools, that button triggers macOS's developer-tools
  prompt.
- `pip install polynet` is a few gigabytes. Most of that is PyTorch, Ray and
  the PolyMetriX dependency chain.
- **The Windows installer cannot be tested on a Mac.** There is no `arm64`
  PowerShell container image, and the `amd64` one crashes under emulation, so
  the Windows path can only be checked by reading it or by running it on
  Windows. A `windows-latest` CI job that runs the installer end to end would
  have caught the stderr bug above before a user did.
