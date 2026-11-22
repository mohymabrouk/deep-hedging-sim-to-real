#!/usr/bin/env bash
# Residual hedgers: validation (4 worlds, one at a time) -> freeze -> final retrain. 2 CPU threads, one process.
cd "$(dirname "$0")/.." || exit 1  # project root
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "start $(date -u +%FT%TZ)"
for g in rough garch bootstrap diffusion; do
  echo "validate_resid $g $(date -u +%FT%TZ)"
  uv run python -m dh.cli validate_resid --only "$g" > "logs/validate_resid_$g.out" 2>&1 || { echo "FAIL validate_resid $g"; exit 1; }
done
echo "freeze $(date -u +%FT%TZ)"
uv run python -m dh.cli freeze > logs/freeze_run3.out 2>&1 || { echo "FAIL freeze"; exit 1; }
echo "final_resid $(date -u +%FT%TZ)"
uv run python -m dh.cli final_resid > logs/final_resid.out 2>&1 || { echo "FAIL final_resid"; exit 1; }
echo "done $(date -u +%FT%TZ)"
