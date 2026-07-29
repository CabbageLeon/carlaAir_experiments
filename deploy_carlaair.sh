#!/bin/bash
# ============================================================
#  deploy_carlaair.sh — CarlaAir v0.1.7 One-Click Deployment
# ============================================================
#  This script clones CarlaAir and sets up everything:
#    system deps, uv, Python venv, AirSim, Carla binary.
#
#  Usage:
#    curl -sSL <raw-url> | bash              # Default install
#    bash deploy_carlaair.sh                 # From repo root
#    bash deploy_carlaair.sh --dir ~/carla   # Custom directory
#    bash deploy_carlaair.sh --skip-binary   # Skip 17GB download
#    bash deploy_carlaair.sh --help
# ============================================================
set -e

# ── Configuration ──────────────────────────────────────────
CARLAAIR_VERSION="v0.1.7"
CARLAAIR_GIT_REPO="https://github.com/louiszengCN/CarlaAir.git"
CARLAAIR_GIT_BRANCH="main"
CARLAAIR_HF_REPO="tianlezeng/CarlaAIr-v0.1.7"
AIRSIM_REPO="https://github.com/microsoft/AirSim.git"
AIRSIM_TAG="v1.8.1"
PYTHON_VERSION="3.10"
VENV_DIR=".venv"
INSTALL_DIR=""                    # set via --dir, defaults to ./CarlaAir

# ── Flags ──────────────────────────────────────────────────
SKIP_BINARY=false
SKIP_AIRSIM=false
SKIP_CLONE=false
AUTO_YES=false

# ── Colors ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ── Helpers ────────────────────────────────────────────────
BANNER="============================================"
ok()   { echo -e "  ${GREEN}[✓]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[!]${NC} $1"; }
fail() { echo -e "  ${RED}[✗]${NC} $1"; }
info() { echo -e "  ${BLUE}[i]${NC} $1"; }
step() { echo -e "\n${BOLD}${BLUE}── Step $1: $2 ──${NC}\n"; }
header() {
    echo -e "\n${BOLD}${BLUE}${BANNER}${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}${BANNER}${NC}\n"
}

die() {
    echo -e "\n${RED}${BANNER}${NC}"
    echo -e "${RED}  DEPLOYMENT FAILED at Step ${CURRENT_STEP}: $1${NC}"
    echo -e "${RED}${BANNER}${NC}"
    if [ -n "${SCRIPT_DIR}" ]; then
        echo -e "${YELLOW}  Fix the issue above and re-run:${NC}"
        echo -e "  ${BOLD}cd ${SCRIPT_DIR} && bash deploy_carlaair.sh --skip-clone${NC}"
    fi
    exit 1
}

check_cmd() {
    command -v "$1" &>/dev/null
}

# ── Argument Parsing ───────────────────────────────────────
CURRENT_STEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dir)
            shift; INSTALL_DIR="$1" ;;
        --skip-binary) SKIP_BINARY=true ;;
        --skip-airsim) SKIP_AIRSIM=true ;;
        --skip-clone)  SKIP_CLONE=true ;;
        -y|--yes)      AUTO_YES=true ;;
        --help|-h)
            echo "Usage: bash deploy_carlaair.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dir PATH       Install directory (default: ./CarlaAir)"
            echo "  --skip-clone     Skip git clone (already inside repo)"
            echo "  --skip-binary    Skip CarlaAir binary download (17GB)"
            echo "  --skip-airsim    Skip AirSim build from source"
            echo "  -y, --yes        Auto-confirm all prompts"
            echo "  --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  curl -sSL <url> | bash                    # One-liner install"
            echo "  bash deploy_carlaair.sh --dir ~/carlaair  # Custom directory"
            echo "  bash deploy_carlaair.sh --skip-binary -y  # Dev machine with binaries"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ── Start ──────────────────────────────────────────────────
header "CarlaAir ${CARLAAIR_VERSION} — One-Click Deployment"
echo "  Install dir : ${INSTALL_DIR:-./CarlaAir}"
echo "  Skip clone  : ${SKIP_CLONE}"
echo "  Skip binary : ${SKIP_BINARY}"
echo "  Skip AirSim : ${SKIP_AIRSIM}"
echo ""

# ───────────────────────────────────────────────────────────
#  Step 0: Clone CarlaAir Repository
# ───────────────────────────────────────────────────────────
CURRENT_STEP=0
step 0 "Clone CarlaAir Repository"

# Detect if we're already inside the repo
SCRIPT_LOCATION="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
IN_REPO=false

if [ -f "${SCRIPT_LOCATION}/carlaAir.sh" ] && [ -f "${SCRIPT_LOCATION}/env_setup/carla_python_module.tar.gz" ]; then
    IN_REPO=true
    SCRIPT_DIR="${SCRIPT_LOCATION}"
fi

if [ "$SKIP_CLONE" = true ]; then
    if [ "$IN_REPO" = true ]; then
        ok "Already inside CarlaAir repo: ${SCRIPT_DIR}"
    else
        # User said skip-clone but we're not in a repo
        if [ -n "$INSTALL_DIR" ] && [ -f "${INSTALL_DIR}/carlaAir.sh" ]; then
            SCRIPT_DIR="$(cd "$INSTALL_DIR" && pwd)"
            ok "Using existing repo: ${SCRIPT_DIR}"
        else
            die "--skip-clone specified but not inside a CarlaAir repo"
        fi
    fi
elif [ "$IN_REPO" = true ]; then
    ok "Already inside CarlaAir repo: ${SCRIPT_DIR}"
    info "  (use --skip-clone to avoid this detection message)"
else
    # Need to clone
    TARGET="${INSTALL_DIR:-${PWD}/CarlaAir}"

    if [ -d "$TARGET" ] && [ -f "${TARGET}/carlaAir.sh" ]; then
        ok "CarlaAir already cloned at: ${TARGET}"
        SCRIPT_DIR="$(cd "$TARGET" && pwd)"
    else
        if [ -d "$TARGET" ] && [ "$(ls -A "$TARGET" 2>/dev/null)" ]; then
            warn "Directory ${TARGET} exists and is not empty"
            if [ "$AUTO_YES" = true ]; then
                info "Using existing directory..."
                SCRIPT_DIR="$(cd "$TARGET" && pwd)"
            else
                info "Clone into it anyway? (git clone will refuse if not empty)"
                info "Or specify another directory with --dir (y=clone, n=abort)"
                read -r response
                if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
                    die "Aborted. Use --dir to specify a different directory."
                fi
            fi
        fi

        if [ -z "$SCRIPT_DIR" ]; then
            info "Cloning ${CARLAAIR_GIT_REPO} -> ${TARGET}..."
            git clone --depth 1 --branch "${CARLAAIR_GIT_BRANCH}" \
                "${CARLAAIR_GIT_REPO}" "${TARGET}" 2>&1 || {
                # Try without branch (repo may use 'main' or 'master')
                warn "Branch '${CARLAAIR_GIT_BRANCH}' failed, trying default branch..."
                git clone --depth 1 "${CARLAAIR_GIT_REPO}" "${TARGET}" 2>&1 || \
                    die "Failed to clone CarlaAir. Check network and GitHub access."
            }
            SCRIPT_DIR="$(cd "$TARGET" && pwd)"
            ok "CarlaAir cloned to: ${SCRIPT_DIR}"
        fi
    fi
fi

info "Project directory: ${SCRIPT_DIR}"

# ───────────────────────────────────────────────────────────
#  Step 1: System Check
# ───────────────────────────────────────────────────────────
CURRENT_STEP=1
step 1 "System Requirements Check"

# OS check
if [ -f /etc/os-release ]; then
    . /etc/os-release
    info "OS: ${NAME} ${VERSION_ID}"
    case "$VERSION_ID" in
        20.04|22.04|24.04) ok "Ubuntu ${VERSION_ID} is supported" ;;
        *) warn "Untested Ubuntu version: ${VERSION_ID}. Proceeding anyway..." ;;
    esac
else
    warn "Cannot detect OS. CarlaAir requires Ubuntu 20.04/22.04."
fi

# Architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    die "CarlaAir requires x86_64 architecture, got: $ARCH"
fi
ok "Architecture: ${ARCH}"

# Disk space (need ~50GB for binary + venv)
AVAIL_GB=$(df -BG "${SCRIPT_DIR}" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAIL_GB" -lt 30 ]; then
    warn "Only ${AVAIL_GB}GB free at ${SCRIPT_DIR}. CarlaAir needs ~30GB (50GB recommended)."
    if [ "$SKIP_BINARY" = false ]; then
        warn "Consider using --skip-binary if you already have CarlaUE4/"
    fi
else
    ok "Disk space: ${AVAIL_GB}GB free"
fi

# RAM
TOTAL_RAM_GB=$(free -g | awk '/Mem:/ {print $2}')
if [ "$TOTAL_RAM_GB" -lt 16 ]; then
    warn "Only ${TOTAL_RAM_GB}GB RAM. 16GB minimum, 32GB recommended."
else
    ok "RAM: ${TOTAL_RAM_GB}GB"
fi

# GPU check (non-fatal)
if check_cmd nvidia-smi; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: ${GPU_INFO}"
else
    warn "nvidia-smi not found. Install NVIDIA drivers for GPU acceleration."
fi

# ───────────────────────────────────────────────────────────
#  Step 2: System Dependencies
# ───────────────────────────────────────────────────────────
CURRENT_STEP=2
step 2 "System Dependencies"

PACKAGES="build-essential git curl wget ca-certificates unzip"
PACKAGES="${PACKAGES} libvulkan1 vulkan-utils libsdl2-2.0-0 libomp5 xdg-user-dirs"

info "Installing: ${PACKAGES}"
echo ""

if [ "$AUTO_YES" = true ]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ${PACKAGES}
else
    sudo apt-get update
    sudo apt-get install -y ${PACKAGES}
fi
ok "System packages installed"

# ───────────────────────────────────────────────────────────
#  Step 3: Install uv
# ───────────────────────────────────────────────────────────
CURRENT_STEP=3
step 3 "Install uv (Python Package Manager)"

if check_cmd uv; then
    UV_VER=$(uv --version)
    ok "uv already installed: ${UV_VER}"
else
    info "Downloading and installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if check_cmd uv; then
        ok "uv installed: $(uv --version)"
    else
        die "uv installation failed. Install manually: https://docs.astral.sh/uv/"
    fi
fi

export PATH="$HOME/.local/bin:$PATH"

# ───────────────────────────────────────────────────────────
#  Step 4: Download CarlaAir Binary
# ───────────────────────────────────────────────────────────
CURRENT_STEP=4
step 4 "Download CarlaAir Binary (CarlaUE4 + Engine) ~17GB"

BINARY_PATH="${SCRIPT_DIR}/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"

if [ "$SKIP_BINARY" = true ]; then
    warn "Skipping binary download (--skip-binary)"
elif [ -f "$BINARY_PATH" ]; then
    ok "CarlaUE4 binary already exists, skipping download"
    info "  To re-download: rm -rf ${SCRIPT_DIR}/CarlaUE4 ${SCRIPT_DIR}/Engine"
else
    info "This is the largest step (~17GB, 10-30 min depending on network)."
    echo ""

    TMP_DIR="${SCRIPT_DIR}/.carlaair_download"
    rm -rf "${TMP_DIR}"
    mkdir -p "${TMP_DIR}"

    DOWNLOAD_OK=false

    # ── Method 1: huggingface-cli ──
    if ! check_cmd huggingface-cli; then
        info "Installing huggingface_hub for download..."
        pip install -q huggingface_hub 2>/dev/null || \
            uv pip install --system -q huggingface_hub 2>/dev/null || true
    fi

    if check_cmd huggingface-cli; then
        info "Downloading from Hugging Face (huggingface-cli)..."
        if huggingface-cli download "${CARLAAIR_HF_REPO}" \
            --local-dir "${TMP_DIR}" \
            --local-dir-use-symlinks False \
            --resume-download 2>&1; then
            DOWNLOAD_OK=true
            info "huggingface-cli download completed"
        else
            warn "huggingface-cli failed, trying wget fallback..."
        fi
    fi

    # ── Method 2: wget direct download ──
    if [ "$DOWNLOAD_OK" = false ]; then
        info "Trying direct wget download..."

        for FILENAME in \
            "CarlaAir_v0.1.7.tar.gz" \
            "CarlaAir_Ubuntu_v0.1.7.tar.gz" \
            "carlaair_v0.1.7.tar.gz" \
            "LinuxNoEditor.tar.gz" \
            "CarlaAir_Linux_v0.1.7.tar.gz"; do

            URL="https://huggingface.co/${CARLAAIR_HF_REPO}/resolve/main/${FILENAME}"
            info "Probing: ${URL}"
            HTTP_CODE=$(curl -sI -o /dev/null -w "%{http_code}" --max-time 10 "${URL}" 2>/dev/null || echo "000")
            if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
                info "Found: ${FILENAME} (HTTP ${HTTP_CODE})"
                wget -c --progress=bar:force -O "${TMP_DIR}/${FILENAME}" "${URL}" 2>&1 && {
                    DOWNLOAD_OK=true
                    break
                }
            fi
        done
    fi

    # ── Method 3: Manual download ──
    if [ "$DOWNLOAD_OK" = false ]; then
        warn ""
        warn "  Automatic download failed. Please download manually:"
        warn "    Hugging Face : https://huggingface.co/${CARLAAIR_HF_REPO}"
        warn "    Baidu Pan    : https://pan.baidu.com/s/1RguWqwKrN-3KEgyKvWiiug?pwd=d5ai"
        warn ""
        warn "  After downloading, extract CarlaUE4/ and Engine/ to:"
        warn "    ${SCRIPT_DIR}/"
        warn ""
        warn "  Then re-run:"
        warn "    cd ${SCRIPT_DIR} && bash deploy_carlaair.sh --skip-clone --skip-binary"
        die "Binary download failed — manual download needed"
    fi

    # ── Extract and install ──
    if [ "$DOWNLOAD_OK" = true ]; then
        info "Extracting files..."
        mkdir -p "${TMP_DIR}/extracted"

        # Try to find downloaded archive
        ARCHIVE=$(ls -t "${TMP_DIR}"/*.tar.gz "${TMP_DIR}"/*.tar.xz "${TMP_DIR}"/*.tar.bz2 "${TMP_DIR}"/*.zip 2>/dev/null | head -1)

        if [ -n "$ARCHIVE" ]; then
            info "Archive: $(basename "$ARCHIVE") ($(du -h "$ARCHIVE" | cut -f1))"
            case "$ARCHIVE" in
                *.tar.gz|*.tgz) tar xzf "$ARCHIVE" -C "${TMP_DIR}/extracted" ;;
                *.tar.xz)       tar xJf "$ARCHIVE" -C "${TMP_DIR}/extracted" ;;
                *.tar.bz2)      tar xjf "$ARCHIVE" -C "${TMP_DIR}/extracted" ;;
                *.zip)          unzip -q "$ARCHIVE" -d "${TMP_DIR}/extracted" ;;
            esac || die "Failed to extract archive"
        fi

        EXTRACT_DIR="${TMP_DIR}/extracted"

        # Handle top-level wrapper directory (LinuxNoEditor/ etc.)
        for SUBDIR in "LinuxNoEditor" "CarlaAir" "CarlaAir_${CARLAAIR_VERSION}"; do
            if [ -d "${EXTRACT_DIR}/${SUBDIR}" ]; then
                EXTRACT_DIR="${EXTRACT_DIR}/${SUBDIR}"
                info "Detected wrapper: ${SUBDIR}/"
                break
            fi
        done

        # Copy CarlaUE4
        if [ -d "${EXTRACT_DIR}/CarlaUE4" ]; then
            SIZE=$(du -sh "${EXTRACT_DIR}/CarlaUE4" 2>/dev/null | cut -f1)
            info "Copying CarlaUE4/ (${SIZE})..."
            rsync -a --info=progress2 "${EXTRACT_DIR}/CarlaUE4/" "${SCRIPT_DIR}/CarlaUE4/" 2>/dev/null || \
                cp -r "${EXTRACT_DIR}/CarlaUE4" "${SCRIPT_DIR}/CarlaUE4/"
            ok "CarlaUE4/ installed"
        else
            warn "CarlaUE4/ not found in archive"
            info "Contents of extracted directory:"
            ls "${EXTRACT_DIR}/" 2>/dev/null | head -20 || true
        fi

        # Copy Engine
        if [ -d "${EXTRACT_DIR}/Engine" ]; then
            SIZE=$(du -sh "${EXTRACT_DIR}/Engine" 2>/dev/null | cut -f1)
            info "Copying Engine/ (${SIZE})..."
            rsync -a --info=progress2 "${EXTRACT_DIR}/Engine/" "${SCRIPT_DIR}/Engine/" 2>/dev/null || \
                cp -r "${EXTRACT_DIR}/Engine" "${SCRIPT_DIR}/Engine/"
            ok "Engine/ installed"
        fi

        # Handle flat download (individual files from huggingface-cli)
        if [ ! -d "${EXTRACT_DIR}/CarlaUE4" ] && [ ! -d "${EXTRACT_DIR}/Engine" ]; then
            # Try the TMP_DIR directly too
            for dir in "${TMP_DIR}" "${EXTRACT_DIR}"; do
                for item in CarlaUE4 Engine CarlaUE4.sh carla_air.sh PythonAPI; do
                    SRC="${dir}/${item}"
                    DST="${SCRIPT_DIR}/${item}"
                    if [ -e "$SRC" ] && [ ! -e "$DST" ]; then
                        mv "$SRC" "$DST" 2>/dev/null || cp -r "$SRC" "$DST"
                        info "Moved: ${item}"
                    fi
                done
            done
        fi

        # Verify
        if [ -f "$BINARY_PATH" ]; then
            chmod +x "$BINARY_PATH"
            ok "Binary verified: CarlaUE4-Linux-Shipping ($(du -h "$BINARY_PATH" | cut -f1))"
        else
            warn "Binary not found at: ${BINARY_PATH}"
            warn "Files were extracted to: ${TMP_DIR}"
            warn "Please move CarlaUE4/ and Engine/ manually to: ${SCRIPT_DIR}/"
        fi

        rm -rf "${TMP_DIR}"
    fi
fi

# ───────────────────────────────────────────────────────────
#  Step 5: Create Python Virtual Environment
# ───────────────────────────────────────────────────────────
CURRENT_STEP=5
step 5 "Create Python Virtual Environment (uv)"

cd "${SCRIPT_DIR}"

# Find working Python
PYTHON_CMD=""
for py in python3.10 python3.11 python3.12 python3; do
    if check_cmd "$py"; then
        PY_VER=$("$py" --version 2>&1 | grep -oP '\d+\.\d+')
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
            PYTHON_CMD="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    info "Python 3.10+ not found, installing via uv..."
    uv python install "${PYTHON_VERSION}" 2>/dev/null || \
        die "Cannot install Python ${PYTHON_VERSION}. Try: sudo apt install python3.10"
    PYTHON_CMD="python3.10"
fi
ok "Python: $(${PYTHON_CMD} --version)"

if [ -d "${VENV_DIR}" ]; then
    warn "Virtual env already exists: ${VENV_DIR}"
    if [ "$AUTO_YES" = true ]; then
        rm -rf "${VENV_DIR}"
        uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
        ok "Virtual env recreated"
    else
        info "Recreate? [y/N]"
        read -r response
        if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
            rm -rf "${VENV_DIR}"
            uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
            ok "Virtual env recreated"
        else
            ok "Using existing virtual env"
        fi
    fi
else
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
    ok "Virtual env created: ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
ok "Virtual env activated"

# ───────────────────────────────────────────────────────────
#  Step 6: Install Python Dependencies
# ───────────────────────────────────────────────────────────
CURRENT_STEP=6
step 6 "Install Python Dependencies"

REQ_FILE="${SCRIPT_DIR}/requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
    warn "requirements.txt not found, creating minimal set..."
    cat > "$REQ_FILE" << 'PYEOF'
numpy>=2.0
opencv-contrib-python>=5.0
pillow>=12.0
pygame>=2.6
scipy>=1.15
openai>=2.0
httpx>=0.28
pydantic>=2.0
tqdm>=4.0
PYEOF
    info "Created minimal requirements.txt"
fi

info "Installing from requirements.txt..."
uv pip install -r "$REQ_FILE"
ok "Python dependencies installed"

# ───────────────────────────────────────────────────────────
#  Step 7: Install AirSim Python Client
# ───────────────────────────────────────────────────────────
CURRENT_STEP=7
step 7 "Install AirSim Python Client"

if [ "$SKIP_AIRSIM" = true ]; then
    warn "Skipping AirSim (--skip-airsim)"
elif python -c "import airsim" 2>/dev/null; then
    AIRSIM_VER=$(python -c "import airsim; print(getattr(airsim, '__version__', 'unknown'))" 2>/dev/null || echo "?")
    ok "AirSim already installed (v${AIRSIM_VER})"
else
    AIRSIM_BUILD_DIR="${SCRIPT_DIR}/.airsim_build"
    rm -rf "${AIRSIM_BUILD_DIR}"

    info "Cloning AirSim ${AIRSIM_TAG}..."
    git clone --depth 1 --branch "${AIRSIM_TAG}" "${AIRSIM_REPO}" "${AIRSIM_BUILD_DIR}" 2>&1 || {
        warn "Tag ${AIRSIM_TAG} not found, trying main branch..."
        git clone --depth 1 "${AIRSIM_REPO}" "${AIRSIM_BUILD_DIR}" 2>&1 || \
            die "Failed to clone AirSim repo. Check network."
    }
    ok "AirSim cloned"

    # Install Python client (try multiple locations)
    INSTALLED=false
    for SUBPATH in "PythonClient" "client" "airsim" "."; do
        AP="${AIRSIM_BUILD_DIR}/${SUBPATH}"
        if [ -f "${AP}/setup.py" ] || [ -f "${AP}/setup.cfg" ] || [ -f "${AP}/pyproject.toml" ]; then
            info "Installing from ${SUBPATH}..."
            cd "${AP}"
            uv pip install -e . 2>/dev/null && INSTALLED=true && break
            uv pip install . 2>/dev/null && INSTALLED=true && break
            python setup.py install 2>/dev/null && INSTALLED=true && break
            cd "${SCRIPT_DIR}"
        fi
    done

    cd "${SCRIPT_DIR}"

    if [ "$INSTALLED" = false ]; then
        warn "Source install failed, falling back to PyPI..."
        uv pip install airsim 2>/dev/null || die "AirSim installation failed completely"
        ok "AirSim installed from PyPI"
    else
        ok "AirSim Python client installed from source"
    fi

    rm -rf "${AIRSIM_BUILD_DIR}"
fi

python -c "import airsim; print('airsim OK:', airsim.__file__)" && \
    ok "AirSim import verified" || warn "AirSim import check failed"

# ───────────────────────────────────────────────────────────
#  Step 8: Install Carla Python Module
# ───────────────────────────────────────────────────────────
CURRENT_STEP=8
step 8 "Install Carla Python Module"

if python -c "import carla" 2>/dev/null; then
    CARLA_FILE=$(python -c "import carla; print(carla.__file__)" 2>/dev/null)
    ok "carla module already installed: ${CARLA_FILE}"
else
    CARLA_TARBALL="${SCRIPT_DIR}/env_setup/carla_python_module.tar.gz"
    [ ! -f "$CARLA_TARBALL" ] && die "carla_python_module.tar.gz not found at ${CARLA_TARBALL}"

    SITE_PKG=$(python -c 'import site; print(site.getsitepackages()[0])')
    info "Extracting to ${SITE_PKG}..."

    for d in "${SITE_PKG}/carla" "${SITE_PKG}/carla.libs"; do
        [ -d "$d" ] && rm -rf "$d"
    done

    tar xzf "$CARLA_TARBALL" -C "$SITE_PKG"
    ok "Carla module installed ($(ls "${SITE_PKG}/carla.libs/" 2>/dev/null | wc -l) shared libs)"
fi

python -c "import carla; print('carla OK — client:', carla.Client('localhost', 9999).get_client_version())" 2>/dev/null && \
    ok "Carla import verified" || \
    python -c "import carla; print('carla import OK')" 2>/dev/null && \
    ok "Carla import verified (no server needed)" || \
    die "Carla import failed"

# ───────────────────────────────────────────────────────────
#  Step 9: Configure AirSim Settings
# ───────────────────────────────────────────────────────────
CURRENT_STEP=9
step 9 "Configure AirSim Settings"

AIRSIM_DOC_DIR="${HOME}/Documents/AirSim"
AIRSIM_SETTINGS="${AIRSIM_DOC_DIR}/settings.json"
SRC_SETTINGS="${SCRIPT_DIR}/AirSimConfig/settings.json"

mkdir -p "${AIRSIM_DOC_DIR}"

if [ -f "$AIRSIM_SETTINGS" ]; then
    ok "AirSim settings already exist: ${AIRSIM_SETTINGS}"
elif [ -f "$SRC_SETTINGS" ]; then
    cp "$SRC_SETTINGS" "$AIRSIM_SETTINGS"
    ok "AirSim settings copied from AirSimConfig/"
else
    cat > "$AIRSIM_SETTINGS" << 'EOF'
{
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "Vehicles": {
        "SimpleFlight": {
            "VehicleType": "SimpleFlight",
            "AutoCreate": true,
            "Cameras": {
                "0": {
                    "CaptureSettings": [
                        { "ImageType": 0, "Width": 1280, "Height": 960 }
                    ],
                    "X": 0.5, "Y": 0.0, "Z": 0.1,
                    "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
                }
            }
        }
    }
}
EOF
    ok "Default AirSim settings created"
fi

# ───────────────────────────────────────────────────────────
#  Step 10: Verify Installation
# ───────────────────────────────────────────────────────────
CURRENT_STEP=10
step 10 "Verify Installation"

PASS=0
FAIL=0
echo ""

# Binary
if [ -f "$BINARY_PATH" ]; then
    ok "CarlaUE4 binary: $(du -h "$BINARY_PATH" | cut -f1)"
    ((PASS++))
else
    ALT="${SCRIPT_DIR}/LinuxNoEditor/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    if [ -f "$ALT" ]; then
        ok "CarlaUE4 binary (alt): ${ALT}"
        ((PASS++))
    else
        fail "CarlaUE4 binary not found"
        ((FAIL++))
    fi
fi

# Engine
[ -d "${SCRIPT_DIR}/Engine" ] && ok "Engine/" && ((PASS++)) || { fail "Engine/"; ((FAIL++)); }

# Python modules
for mod in carla airsim numpy cv2 pygame PIL scipy tqdm; do
    if python -c "import ${mod}" 2>/dev/null; then
        ok "Python: ${mod}"
        ((PASS++))
    else
        fail "Python: ${mod} — not importable"
        ((FAIL++))
    fi
done

# AirSim settings
[ -f "$AIRSIM_SETTINGS" ] && ok "AirSim settings" && ((PASS++)) || { fail "AirSim settings"; ((FAIL++)); }

# Launcher
[ -f "${SCRIPT_DIR}/carlaAir.sh" ] && ok "Launcher: carlaAir.sh" && ((PASS++)) || \
    warn "carlaAir.sh not found (binary release uses carla_air.sh)"

# ── Summary ──
echo ""
echo -e "${BLUE}${BANNER}${NC}"
if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}✓  DEPLOYMENT SUCCESSFUL — ${PASS} checks passed${NC}"
else
    echo -e "  ${YELLOW}${BOLD}⚠  DEPLOYED WITH WARNINGS — ${PASS} passed, ${FAIL} failed${NC}"
fi
echo -e "${BLUE}${BANNER}${NC}"

echo ""
echo -e "  ${BOLD}Start CarlaAir:${NC}"
echo -e "    ${GREEN}cd ${SCRIPT_DIR}${NC}"
echo -e "    ${GREEN}source ${VENV_DIR}/bin/activate${NC}"
echo -e "    ${GREEN}./carlaAir.sh${NC}"
echo ""
echo -e "  ${BOLD}Verify (after CarlaAir is running):${NC}"
echo -e "    ${GREEN}source ${VENV_DIR}/bin/activate${NC}"
echo -e "    ${GREEN}bash env_setup/test_env.sh${NC}"
echo ""
echo -e "  ${BOLD}Quick test:${NC}"
echo -e "    ${GREEN}python examples/quick_start_showcase.py${NC}"

exit ${FAIL}
