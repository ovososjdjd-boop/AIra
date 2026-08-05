#!/usr/bin/env bash
# Восстановление среды AIra после сброса песочницы (.venv не версионируется).
#
#   bash scripts/setup_env.sh           # полная стойка (numpy + matplotlib + torch CPU)
#   AIRA_NO_TORCH=1 bash scripts/setup_env.sh   # лёгкий вариант без torch (EXP-01/02, корпус)
#
# torch 2.1.2 ставится с pypi вместе с nvidia-* библиотеками (~2.5 ГБ). download.pytorch.org
# из среды недоступен, а CPU-колёса pypi требуют nvidia-пакеты для import — это ожидаемо.
set -e
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
if [ "${AIRA_NO_TORCH:-0}" != "1" ]; then
  .venv/bin/pip install --quiet "torch==2.1.2"
  .venv/bin/python -c "import torch; print('torch OK', torch.__version__)"
fi
echo "ok: .venv готово ($(./.venv/bin/python -V 2>&1))"
