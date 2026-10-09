#!/usr/bin/env bash
# 一次性环境安装：conda 环境 guitarize（Python 3.11）+ Homebrew 工具 + Node 依赖
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"
ENV_NAME=guitarize
ENV_PY="$("$CONDA" info --base)/envs/$ENV_NAME/bin/python"

if [ ! -x "$ENV_PY" ]; then
  "$CONDA" create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.11
fi
"$ENV_PY" -m pip install --upgrade pip
"$ENV_PY" -m pip install torch torchaudio
"$ENV_PY" -m pip install demucs librosa mir_eval soundfile pretty_midi pyyaml \
  yt-dlp pytest scipy matplotlib torchcrepe
"$ENV_PY" -m pip install "git+https://github.com/CPJKU/beat_this.git"

brew list fluid-synth >/dev/null 2>&1 || brew install fluid-synth
brew list gh >/dev/null 2>&1 || brew install gh

(cd "$ROOT/tools/alphatab" && npm install)
echo "setup done: $ENV_PY"
