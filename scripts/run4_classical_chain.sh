#!/usr/bin/env bash
# Test run 4 (PLAN.md change 17): classical hedgers validated on 2021 -> freeze -> one forced test run -> haircut check
# -> report. 2 CPU threads, one process at a time. Run 3 must already be archived in results/run3_official/.
cd "$(dirname "$0")/.." || exit 1  # project root
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "start $(date -u +%FT%TZ)"
[ -f results/run3_official/performance_test.csv ] || { echo "FAIL run 3 not archived"; exit 1; }
echo "validate_classical $(date -u +%FT%TZ)"
uv run python -m dh.cli validate_classical > logs/validate_classical.out 2>&1 || { echo "FAIL validate_classical"; exit 1; }
echo "freeze $(date -u +%FT%TZ)"
uv run python -m dh.cli freeze > logs/freeze_run4.out 2>&1 || { echo "FAIL freeze"; exit 1; }
echo "test --force $(date -u +%FT%TZ)"
uv run python -m dh.cli test --force > logs/test_run4.out 2>&1 || { echo "FAIL test"; exit 1; }
echo "haircut $(date -u +%FT%TZ)"
uv run python -m dh.cli haircut > logs/haircut.out 2>&1 || { echo "FAIL haircut"; exit 1; }
echo "report $(date -u +%FT%TZ)"
uv run python -m dh.cli report > logs/report_run4.out 2>&1 || { echo "FAIL report"; exit 1; }
echo "done $(date -u +%FT%TZ)"
