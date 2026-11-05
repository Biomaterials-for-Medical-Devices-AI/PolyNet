#!/bin/bash
#
# PolyNet one-click installer for macOS (and Linux).
#
# Double-click this file. It needs no Python, no admin password and no
# Homebrew: it downloads a private copy of Python, installs PolyNet into its
# own virtual environment and leaves a clickable PolyNet icon behind.
#
#   macOS   ~/Applications/PolyNet.app  (plus a Desktop shortcut)
#   Linux   PolyNet in the applications menu
#
# Everything lives under ~/.polynet, so `--uninstall` removes all of it.
#
# usage: Install-PolyNet.command [options]
#
#   --uninstall      Remove PolyNet, its Python and its icons.
#   --version X.Y.Z  Install this PolyNet version instead of the latest.
#   --check          Also run `polynet check` (a short training run on
#                    synthetic data) at the end.
#   --icon FILE      Use this image for the icon instead of the PolyNet logo.
#                    Any square PNG works; bigger than 512x512 is best.
#   --no-shortcut    Do not put PolyNet on the Desktop.
#   -h, --help       Show this message.
#
# Environment overrides: POLYNET_HOME (install directory), POLYNET_ICON,
# POLYNET_PYTHON_VERSION, POLYNET_UV_VERSION.

set -u -o pipefail

POLYNET_HOME="${POLYNET_HOME:-$HOME/.polynet}"
PYTHON_VERSION="${POLYNET_PYTHON_VERSION:-3.11}"
UV_VERSION="${POLYNET_UV_VERSION:-latest}"

ENV_DIR="$POLYNET_HOME/env"
TOOLS_DIR="$POLYNET_HOME/tools"
BIN_DIR="$POLYNET_HOME/bin"
LAUNCHER="$BIN_DIR/polynet-launch"
LOG="$POLYNET_HOME/install.log"

APP_DIR="$HOME/Applications/PolyNet.app"
DESKTOP_APP="$HOME/Desktop/PolyNet.app"
DESKTOP_ENTRY="$HOME/.local/share/applications/polynet.desktop"

# Installers before this one put a symlink here instead of an app bundle.
# Finder draws a symlink with a generic icon, which is why it is now a copy.
LEGACY_DESKTOP_LINK="$HOME/Desktop/PolyNet"

OS="$(uname -s)"
ARCH="$(uname -m)"

PKG_SPEC="polynet"
ICON_SOURCE="${POLYNET_ICON:-}"
DO_UNINSTALL=0
DO_CHECK=0
MAKE_SHORTCUT=1

# ---------------------------------------------------------------- output ----

if [ -t 1 ]; then
    B=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; N=$'\033[0m'
else
    B=""; DIM=""; RED=""; GREEN=""; YELLOW=""; N=""
fi

STEP=0
TOTAL=7

step() {
    STEP=$((STEP + 1))
    printf '\n%s[%d/%d] %s%s\n' "$B" "$STEP" "$TOTAL" "$1" "$N"
}

info() { printf '      %s\n' "$1"; }
warn() { printf '%s      ! %s%s\n' "$YELLOW" "$1" "$N"; }

die() {
    printf '\n%s  Installation failed: %s%s\n' "$RED" "$1" "$N"
    printf '  The full log is at: %s\n' "$LOG"
    printf '  Please send that file to the PolyNet maintainers if you need help.\n'
    pause
    exit 1
}

pause() {
    if [ -t 0 ]; then
        printf '\n  Press Return to close this window. '
        read -r _ || true
    fi
}

usage() {
    sed -n '/^# usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

# ------------------------------------------------------------- arguments ----

while [ $# -gt 0 ]; do
    case "$1" in
        --uninstall) DO_UNINSTALL=1 ;;
        --check) DO_CHECK=1 ;;
        --no-shortcut) MAKE_SHORTCUT=0 ;;
        --version)
            [ $# -ge 2 ] || { echo "--version needs a version number" >&2; exit 2; }
            PKG_SPEC="polynet==$2"; shift ;;
        --version=*) PKG_SPEC="polynet==${1#--version=}" ;;
        --icon)
            [ $# -ge 2 ] || { echo "--icon needs a file" >&2; exit 2; }
            ICON_SOURCE="$2"; shift ;;
        --icon=*) ICON_SOURCE="${1#--icon=}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# ------------------------------------------------------------- uninstall ----

if [ "$DO_UNINSTALL" = 1 ]; then
    printf '%sRemoving PolyNet%s\n\n' "$B" "$N"
    for target in "$APP_DIR" "$DESKTOP_APP" "$LEGACY_DESKTOP_LINK" \
                  "$DESKTOP_ENTRY" "$POLYNET_HOME"; do
        if [ -e "$target" ] || [ -L "$target" ]; then
            rm -rf "$target" && printf '  removed %s\n' "$target"
        fi
    done
    printf '\n%s  PolyNet has been removed.%s\n' "$GREEN" "$N"
    printf '  Your experiments and data were not touched.\n'
    pause
    exit 0
fi

# ----------------------------------------------------------------- start ----

mkdir -p "$POLYNET_HOME" "$TOOLS_DIR" "$BIN_DIR" || {
    echo "Cannot write to $POLYNET_HOME" >&2; exit 1; }

# Mirror everything from here on into the log file.
exec > >(tee "$LOG") 2>&1

cat <<BANNER

${B}  PolyNet installer${N}
  ${DIM}polymer property prediction with graph neural networks${N}

  This will install PolyNet and its own private copy of Python into
  ${POLYNET_HOME}. Nothing else on this computer is changed, and no
  admin password is needed.

  The download is a few gigabytes (PyTorch, RDKit and friends), so this
  usually takes 5-15 minutes on a normal connection. You can leave it
  running.

BANNER

case "$OS" in
    Darwin)
        case "$ARCH" in
            arm64) UV_TARGET="aarch64-apple-darwin" ;;
            x86_64) UV_TARGET="x86_64-apple-darwin" ;;
            *) die "unsupported Mac architecture: $ARCH" ;;
        esac ;;
    Linux)
        case "$ARCH" in
            aarch64|arm64) UV_TARGET="aarch64-unknown-linux-gnu" ;;
            x86_64) UV_TARGET="x86_64-unknown-linux-gnu" ;;
            *) die "unsupported Linux architecture: $ARCH" ;;
        esac ;;
    *) die "this installer supports macOS and Linux; on Windows use Install-PolyNet.bat" ;;
esac

command -v curl >/dev/null 2>&1 || die "curl is not available on this computer"

# uv keeps its managed Python builds and its download cache inside
# POLYNET_HOME so that --uninstall really removes everything.
export UV_PYTHON_INSTALL_DIR="$POLYNET_HOME/python"
export UV_CACHE_DIR="$POLYNET_HOME/cache"
export UV_NO_MODIFY_PATH=1
# Build the environment on the private Python even when the computer already
# has a python3.11 on its PATH, from conda or Homebrew say: PolyNet should not
# depend on an interpreter it does not control.
export UV_MANAGED_PYTHON=1

# ------------------------------------------------- 1. the package manager ----

step "Fetching the installer tool (uv)"

UV="$TOOLS_DIR/uv"
if [ -x "$UV" ] && "$UV" --version >/dev/null 2>&1; then
    info "already present: $("$UV" --version)"
else
    if [ "$UV_VERSION" = "latest" ]; then
        UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_TARGET}.tar.gz"
    else
        UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${UV_TARGET}.tar.gz"
    fi
    info "downloading ${UV_URL}"
    curl -fsSL "$UV_URL" | tar -xz -C "$TOOLS_DIR" --strip-components=1 \
        || die "could not download uv. Check your internet connection, then run this installer again."
    [ -x "$UV" ] || die "uv was downloaded but is not where it was expected ($UV)"
    info "installed $("$UV" --version)"
fi

# --------------------------------------------------------------- 2. Python ----

step "Installing Python ${PYTHON_VERSION} (a private copy, just for PolyNet)"

# --no-bin keeps uv from adding a python3.11 shim to ~/.local/bin: PolyNet
# uses its own environment, and the shim would warn about PATH.
"$UV" python install --no-bin "$PYTHON_VERSION" \
    || die "could not install Python ${PYTHON_VERSION}"
info "done"

# ------------------------------------------------------ 3. the environment ----

step "Creating PolyNet's environment"

if [ -x "$ENV_DIR/bin/python" ]; then
    info "reusing the existing environment in $ENV_DIR"
else
    # --seed installs pip, which the GUI's "install PSMILES" button needs.
    "$UV" venv --seed --python "$PYTHON_VERSION" "$ENV_DIR" \
        || die "could not create the environment in $ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
[ -x "$PY" ] || die "no Python interpreter at $PY"

# ------------------------------------------------------------- 4. PolyNet ----

step "Installing PolyNet and its dependencies (this is the long part)"

"$UV" pip install --python "$PY" --upgrade "$PKG_SPEC" \
    || die "could not install $PKG_SPEC"

INSTALLED_VERSION="$("$PY" -c 'import polynet; print(polynet.__version__)' 2>/dev/null)" \
    || die "PolyNet was installed but cannot be imported"
info "installed PolyNet $INSTALLED_VERSION"

# -------------------------------------------------------------- 5. OpenMP ----

# On macOS the XGBoost wheel looks for libomp.dylib only in Homebrew's
# directory, so `import xgboost` fails on a machine without Homebrew. PyTorch
# and scikit-learn both ship a compatible copy of that library, so put one
# where PolyNet's Python can find it and load it at interpreter start-up. This
# also guarantees a single OpenMP runtime in the process, which is what
# PyTorch and XGBoost need in order to coexist.

step "Making XGBoost work without Homebrew"

if [ "$OS" = "Darwin" ]; then
    SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
        || die "could not locate the environment's site-packages directory"

    LIBOMP=""
    for candidate in \
        "$SITE/torch/lib/libomp.dylib" \
        "$SITE/sklearn/.dylibs/libomp.dylib" \
        "$SITE/xgboost/lib/libomp.dylib"
    do
        if [ -f "$candidate" ]; then LIBOMP="$candidate"; break; fi
    done
    [ -n "$LIBOMP" ] || die "no bundled libomp.dylib was found in $SITE"

    mkdir -p "$SITE/polynet_runtime"
    cp "$LIBOMP" "$SITE/polynet_runtime/libomp.dylib" \
        || die "could not copy $LIBOMP"

    cat > "$SITE/polynet_openmp_preload.py" <<'PREEOF'
"""Load PolyNet's OpenMP runtime before anything else imports XGBoost.

The XGBoost wheel for macOS looks for libomp.dylib only in Homebrew's
directory, so importing it fails on a machine without Homebrew. Loading a
bundled copy here, under the install name XGBoost asks for, satisfies that
lookup and keeps a single OpenMP runtime in the process, which is what PyTorch
and XGBoost need in order to coexist.

Written by the PolyNet installer (installers/macos/Install-PolyNet.command) and
run at interpreter start-up by zz_polynet_openmp.pth in the same directory.
"""

import ctypes
import os

_LIBOMP = os.path.join(os.path.dirname(__file__), "polynet_runtime", "libomp.dylib")

try:
    ctypes.CDLL(_LIBOMP, mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass
PREEOF

    printf 'import polynet_openmp_preload\n' > "$SITE/zz_polynet_openmp.pth"
    info "using the copy of libomp.dylib bundled with $(basename "$(dirname "$(dirname "$LIBOMP")")")"
else
    info "not needed on $OS"
fi

# ------------------------------------------------------------ 6. launcher ----

step "Creating the PolyNet icon"

cat > "$LAUNCHER" <<LAUNCHEOF
#!/bin/bash
# Starts the PolyNet GUI. Created by the PolyNet installer.
cat <<'MSG'

  ======================================================
    PolyNet is starting.

    Your web browser will open in a few seconds. The
    first start takes a little longer than the rest.

    KEEP THIS WINDOW OPEN while you use PolyNet.
    To stop PolyNet, close this window or press Ctrl-C.
  ======================================================

MSG
cd "\$HOME" || exit 1
exec "$ENV_DIR/bin/polynet" "\$@"
LAUNCHEOF
chmod +x "$LAUNCHER"

# A double-clickable uninstaller, for people who would rather not hunt down
# this installer again. It is deliberately short: it deletes the directory it
# lives in, so bash must have read all of it before the first rm.
cat > "$POLYNET_HOME/Uninstall-PolyNet.command" <<UNINSTALLEOF
#!/bin/bash
# Removes PolyNet. Created by the PolyNet installer.
printf '\n  Removing PolyNet...\n'
rm -rf "$APP_DIR" "$DESKTOP_APP" "$LEGACY_DESKTOP_LINK" "$DESKTOP_ENTRY" "$POLYNET_HOME"
printf '\n  PolyNet has been removed.\n'
printf '  Your experiments and data were not touched.\n\n'
if [ -t 0 ]; then printf '  Press Return to close this window. '; read -r _; fi
UNINSTALLEOF
chmod +x "$POLYNET_HOME/Uninstall-PolyNet.command"

make_icns() {
    # $1 = source png, $2 = destination icns
    local src="$1" dest="$2" iconset size
    iconset="$(mktemp -d)/PolyNet.iconset"
    mkdir -p "$iconset" || return 1
    for size in 16 32 64 128 256 512 1024; do
        sips -z "$size" "$size" "$src" --out "$iconset/icon_${size}x${size}.png" \
            >/dev/null 2>&1 || return 1
    done
    # Retina variants are the next size up under an @2x name.
    for size in 16 32 128 256 512; do
        cp "$iconset/icon_$((size * 2))x$((size * 2)).png" \
            "$iconset/icon_${size}x${size}@2x.png" || return 1
    done
    rm -f "$iconset/icon_64x64.png" "$iconset/icon_1024x1024.png"
    iconutil -c icns "$iconset" -o "$dest" >/dev/null 2>&1
}

# The icon comes from the logo that ships inside the installed package, so it
# always matches the version that was installed. --icon overrides it.
sign_app() {
    # Ad-hoc signature. Not a Developer ID -- it just keeps the bundle
    # internally consistent after the icon swap.
    codesign --force --sign - "$1" >/dev/null 2>&1 \
        || warn "could not re-sign $1"
}

LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister

register_app() {
    # Finder caches icons per bundle, so a reinstall over an old PolyNet.app
    # can keep showing the previous icon until LaunchServices re-reads it.
    [ -x "$LSREGISTER" ] && "$LSREGISTER" -f "$1" >/dev/null 2>&1
    return 0
}

LOGO="$("$PY" -c 'import pathlib, polynet; print(pathlib.Path(polynet.__file__).parent / "app" / "static" / "polynet.png")' 2>/dev/null)"
if [ -n "$ICON_SOURCE" ]; then
    [ -f "$ICON_SOURCE" ] || die "no such icon file: $ICON_SOURCE"
    LOGO="$ICON_SOURCE"
    info "using the icon you supplied: $ICON_SOURCE"
fi

if [ "$OS" = "Darwin" ]; then
    mkdir -p "$HOME/Applications"
    rm -rf "$APP_DIR"

    # An AppleScript applet is a real app bundle: it takes a custom icon, it
    # shows up in Launchpad and Spotlight, and -- because it is built here on
    # this machine rather than downloaded -- Gatekeeper does not block it.
    # `open -a Terminal` avoids the "wants to control Terminal" permission
    # prompt that `tell application "Terminal"` would trigger.
    APPLESCRIPT="$(mktemp -d)/PolyNet.applescript"
    cat > "$APPLESCRIPT" <<APPLEEOF
on run
	do shell script "open -a Terminal " & quoted form of "$LAUNCHER"
end run
APPLEEOF
    osacompile -o "$APP_DIR" "$APPLESCRIPT" >/dev/null 2>&1 \
        || die "could not create $APP_DIR"

    plutil -replace CFBundleIdentifier -string "ai.polynet.launcher" \
        "$APP_DIR/Contents/Info.plist" >/dev/null 2>&1
    plutil -replace CFBundleShortVersionString -string "$INSTALLED_VERSION" \
        "$APP_DIR/Contents/Info.plist" >/dev/null 2>&1

    if [ -f "$LOGO" ] && make_icns "$LOGO" "$APP_DIR/Contents/Resources/applet.icns"; then
        # osacompile sets CFBundleIconFile (-> applet.icns) AND CFBundleIconName
        # (-> the applet artwork in Assets.car), and macOS prefers the asset
        # catalogue. Replacing applet.icns on its own therefore changes nothing
        # on screen: Finder keeps drawing the stock AppleScript applet icon.
        # Dropping the key and the catalogue makes the icns the icon.
        plutil -remove CFBundleIconName "$APP_DIR/Contents/Info.plist" >/dev/null 2>&1
        rm -f "${APP_DIR:?}/Contents/Resources/Assets.car"
        info "icon built from $(basename "$LOGO")"
    else
        warn "could not build the app icon; PolyNet will use the default one"
    fi

    # osacompile ad-hoc signs the applet, and editing Resources afterwards
    # invalidates that signature, so sign it again.
    sign_app "$APP_DIR"
    touch "$APP_DIR"
    register_app "$APP_DIR"
    info "created $APP_DIR"

    rm -rf "$LEGACY_DESKTOP_LINK"
    if [ "$MAKE_SHORTCUT" = 1 ] && [ -d "$HOME/Desktop" ]; then
        # A copy of the bundle rather than a symlink: Finder resolves a symlink
        # to a generic icon, so the logo would not show on the Desktop. Both
        # copies run the same launcher, and the uninstaller removes both.
        rm -rf "$DESKTOP_APP"
        if cp -R "$APP_DIR" "$DESKTOP_APP"; then
            sign_app "$DESKTOP_APP"
            touch "$DESKTOP_APP"
            register_app "$DESKTOP_APP"
            info "put PolyNet on your Desktop"
        fi
    fi
else
    mkdir -p "$(dirname "$DESKTOP_ENTRY")"
    ICON_PATH="$POLYNET_HOME/polynet.png"
    [ -f "$LOGO" ] && cp "$LOGO" "$ICON_PATH"
    cat > "$DESKTOP_ENTRY" <<ENTRYEOF
[Desktop Entry]
Type=Application
Name=PolyNet
Comment=Polymer property prediction with graph neural networks
Exec=$LAUNCHER
Icon=$ICON_PATH
Terminal=true
Categories=Science;Education;
ENTRYEOF
    chmod +x "$DESKTOP_ENTRY"
    info "created $DESKTOP_ENTRY"
    if [ "$MAKE_SHORTCUT" = 1 ] && [ -d "$HOME/Desktop" ]; then
        cp "$DESKTOP_ENTRY" "$HOME/Desktop/polynet.desktop" 2>/dev/null \
            && chmod +x "$HOME/Desktop/polynet.desktop" \
            && info "created a PolyNet shortcut on your Desktop"
    fi
fi

# --------------------------------------------------------------- 7. check ----

step "Checking that everything works"

"$PY" - <<'PYEOF' || die "PolyNet is installed but some of its parts do not load. See the log for details."
import importlib

for module in ("polynet", "torch", "rdkit", "xgboost", "sklearn", "streamlit", "torch_geometric"):
    importlib.import_module(module)
    print(f"      {module} ok")
PYEOF

if [ "$DO_CHECK" = 1 ]; then
    info "running polynet check (a short training run on synthetic data)"
    "$ENV_DIR/bin/polynet" check || die "polynet check failed"
fi

"$UV" cache prune >/dev/null 2>&1 || true

# ---------------------------------------------------------------- finish ----

printf '\n%s  PolyNet %s is installed.%s\n\n' "$GREEN" "$INSTALLED_VERSION" "$N"
if [ "$OS" = "Darwin" ]; then
    printf '  To start it, double-click %sPolyNet%s on your Desktop, or find\n' "$B" "$N"
    printf '  PolyNet in your Applications folder or in Launchpad.\n'
else
    printf '  To start it, look for %sPolyNet%s in your applications menu.\n' "$B" "$N"
fi
printf '\n  %sIt is also available in the Terminal:%s %s\n' "$DIM" "$N" "$ENV_DIR/bin/polynet"
printf '  %sTo remove it again:%s open %s\n' "$DIM" "$N" "$POLYNET_HOME/Uninstall-PolyNet.command"
printf '  %sInstall log:%s %s\n' "$DIM" "$N" "$LOG"
pause
