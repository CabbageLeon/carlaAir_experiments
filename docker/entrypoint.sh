#!/bin/bash
# ============================================================
#  CarlaAir Docker Entrypoint
# ============================================================
#  1. git clone CarlaAir (one time)
#  2. bash deploy_carlaair.sh --skip-clone (one time)
#  3. ./carlaAir.sh (every time)
# ============================================================
set -e

REPO_DIR="/workspace/CarlaAir"
export PATH="/root/.local/bin:${PATH}"
MAP="${1:-Town10HD}"

# ── Step 1: Clone project (first run) ──────────────────────
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo ""
    echo "================================================================="
    echo "  [1/3] Cloning CarlaAir from GitHub..."
    echo "================================================================="
    git clone --depth 1 https://github.com/louiszengCN/CarlaAir.git "$REPO_DIR"
    cd "$REPO_DIR"
    git lfs pull 2>/dev/null || true
    echo "  Done."
else
    cd "$REPO_DIR"
    echo "  Project already cloned."
fi

# ── Step 2: Run deploy script (first run) ──────────────────
if [ ! -f "${REPO_DIR}/.deploy_done" ]; then
    echo ""
    echo "================================================================="
    echo "  [2/3] Running deploy_carlaair.sh..."
    echo "  This downloads CarlaUE4 (~17GB) and installs Python deps."
    echo "  Estimated: 10–30 minutes depending on network."
    echo "================================================================="
    echo ""

    # Copy deploy script into the cloned project so it finds
    # requirements.txt and env_setup/ (SCRIPT_DIR = project root)
    cp /opt/deploy_carlaair.sh "${REPO_DIR}/deploy_carlaair.sh"
    cd "${REPO_DIR}"
    if bash deploy_carlaair.sh --skip-clone -y; then
        touch "${REPO_DIR}/.deploy_done"
        echo ""
        echo "================================================================="
        echo "  Deployment successful!"
        echo "================================================================="
    else
        echo ""
        echo "================================================================="
        echo "  Deployment FAILED."
        echo "  Fix the issue and restart the container (docker compose restart)."
        echo "  Keeping container alive for debugging..."
        echo "================================================================="
        exec bash
    fi
fi

# ── Step 3: Verify binary exists ───────────────────────────
BINARY="${REPO_DIR}/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
if [ ! -f "$BINARY" ]; then
    echo ""
    echo "================================================================="
    echo "  ERROR: CarlaUE4 binary not found!"
    echo "  ${BINARY}"
    echo ""
    echo "  Remove .deploy_done and re-run deploy:"
    echo "    rm ${REPO_DIR}/.deploy_done"
    echo "    bash ${REPO_DIR}/deploy_carlaair.sh --skip-clone -y"
    echo "================================================================="
    exec bash
fi

# ── Step 4: Launch ─────────────────────────────────────────
cd "$REPO_DIR"
source .venv/bin/activate

echo ""
echo "================================================================="
echo "  CarlaAir v0.1.7"
echo "  Map:        ${MAP}"
echo "  CARLA port: ${CARLA_PORT:-2000}"
echo "  AirSim port: ${AIRSIM_PORT:-41451}"
echo "================================================================="
echo ""

exec ./carlaAir.sh "$MAP"
