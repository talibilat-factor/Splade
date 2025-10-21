#!/usr/bin/env bash
# readme_workflow.sh
# End-to-end helper replicating README steps for SPLADE.
# Features:
#  * Environment setup (conda or venv fallback)
#  * Toy run (default config)
#  * Full data run (downloads triplets + config_splade)
#  * Idempotent: safe to re-run, will skip what already exists
#
# Usage examples:
#  ./scripts/readme_workflow.sh --setup-env --toy-run
#  ./scripts/readme_workflow.sh --full-run  # assumes env already active
#  ./scripts/readme_workflow.sh --download-data-only
#
# Options:
#  --setup-env              Create & activate conda env (splade) or python venv if conda missing
#  --env-name NAME          Name for conda env (default: splade)
#  --python VERSION         Python version for env (default: 3.11)
#  --toy-run                Run toy example (config_default.yaml)
#  --full-run               Run full training (downloads if needed + config_splade)
#  --download-data-only     Only fetch and extract triplets data
#  --no-conda               Force using python -m venv instead of conda
#  --keep-existing          Do not recreate environment if it already exists
#  --dry-run                Print commands only
#  -h|--help                Show help
#
# After environment setup this script prints an activation hint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Default parameters
ENV_NAME="splade"
PY_VERSION="3.11"
DO_SETUP_ENV=false
USE_CONDA=true
FORCE_NO_RECREATE=false
TOY_RUN=false
FULL_RUN=false
DOWNLOAD_ONLY=false
DRY_RUN=false
KEEP_EXISTING=false

TRIPLETS_URL="https://download.europe.naverlabs.com/splade/sigir22/triplets.tar.gz"
TRIPLETS_ARCHIVE="triplets.tar.gz"
DATA_DIR="triplets" # archive produces directories; we just mark extraction by existence of some file maybe

print_help() {
  grep '^# ' "$0" | sed 's/^# \{0,1\}//'
}

run_cmd() {
  if $DRY_RUN; then
    echo "[dry-run] $*"
  else
    echo "+ $*"
    eval "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-env) DO_SETUP_ENV=true; shift ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --python) PY_VERSION="$2"; shift 2 ;;
    --toy-run) TOY_RUN=true; shift ;;
    --full-run) FULL_RUN=true; shift ;;
    --download-data-only) DOWNLOAD_ONLY=true; shift ;;
    --no-conda) USE_CONDA=false; shift ;;
    --keep-existing) KEEP_EXISTING=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 1 ;;
  esac
done

if $TOY_RUN && $FULL_RUN; then
  echo "Cannot request both --toy-run and --full-run" >&2
  exit 1
fi

if $DOWNLOAD_ONLY && ($TOY_RUN || $FULL_RUN); then
  echo "--download-data-only cannot be combined with run flags" >&2
  exit 1
fi

# Detect conda
if $DO_SETUP_ENV; then
  if ! command -v conda >/dev/null 2>&1 || ! $USE_CONDA; then
    echo "Conda not available or --no-conda specified; will use python -m venv"
    USE_CONDA=false
  fi
  if $USE_CONDA; then
    # Check if env exists
    if conda env list | grep -E "^$ENV_NAME " >/dev/null 2>&1; then
      if ! $KEEP_EXISTING; then
        echo "Environment $ENV_NAME already exists. Recreate? (y/N)"; read -r ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
          run_cmd conda remove -y -n "$ENV_NAME" --all
        fi
      fi
    fi
    if ! conda env list | grep -E "^$ENV_NAME " >/dev/null 2>&1; then
      run_cmd conda create -n "$ENV_NAME" -y python="$PY_VERSION" pip
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    run_cmd conda activate "$ENV_NAME"
  else
    VENV_PATH=".venv_$ENV_NAME"
    if [[ -d "$VENV_PATH" ]] && ! $KEEP_EXISTING; then
      echo "Virtualenv $VENV_PATH exists. Recreate? (y/N)"; read -r ans
      if [[ "$ans" =~ ^[Yy]$ ]]; then
        run_cmd rm -rf "$VENV_PATH"
      fi
    fi
    if [[ ! -d "$VENV_PATH" ]]; then
      run_cmd python"$PY_VERSION" -m venv "$VENV_PATH" || run_cmd python -m venv "$VENV_PATH"
    fi
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
  fi
  echo "Installing Python dependencies..."
  run_cmd pip install --upgrade pip
  run_cmd pip install hydra-core transformers numba h5py pytrec-eval tensorboard matplotlib
fi

# Data download
if $DOWNLOAD_ONLY || $FULL_RUN; then
  if [[ ! -f "$TRIPLETS_ARCHIVE" ]]; then
    run_cmd wget -O "$TRIPLETS_ARCHIVE" "$TRIPLETS_URL"
  else
    echo "Archive already present: $TRIPLETS_ARCHIVE"
  fi
  # Extract only if not yet extracted (check a representative file)
  if [[ ! -d triplets ]] && ! ls -1 | grep -q "triplets"; then
    run_cmd tar -zxvf "$TRIPLETS_ARCHIVE"
  else
    echo "Triplets data seems already extracted."
  fi
fi

if $DOWNLOAD_ONLY; then
  echo "Download-only operation complete."; exit 0
fi

export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

run_toy() {
  export SPLADE_CONFIG_NAME="config_default.yaml"
  run_cmd python3 -m splade.all \
    config.checkpoint_dir=experiments/debug/checkpoint \
    config.index_dir=experiments/debug/index \
    config.out_dir=experiments/debug/out
}

run_full() {
  export SPLADE_CONFIG_NAME="config_splade"
  run_cmd python3 -m splade.all \
    config.checkpoint_dir=experiments/splade/checkpoint \
    config.index_dir=experiments/splade/index \
    config.out_dir=experiments/splade/out
}

if $TOY_RUN; then
  run_toy
elif $FULL_RUN; then
  run_full
else
  echo "No run flag provided. Use --toy-run or --full-run (or --help)."
fi

cat <<EOF

[INFO] Script finished.
If you created an environment, remember to activate it in new shells:
  conda activate $ENV_NAME   # if conda
or
  source .venv_$ENV_NAME/bin/activate  # if venv
EOF
