# Repository Guidelines

Primary objective: optimize the solution so `python3 tests/submission_tests.py` reports fewer than `1200` cycles.

## Project Structure & Module Organization
This repository is a Python performance take-home focused on optimizing one kernel.
- `perf_takehome.py`: main submission file; optimize `KernelBuilder.build_kernel`.
- `problem.py`: simulator, ISA primitives, and reference logic used by the starter flow.
- `tests/submission_tests.py`: official correctness/performance thresholds (run before submitting).
- `tests/frozen_problem.py`: frozen simulator/reference used by submission tests.
- `watch_trace.py` and `watch_trace.html`: local trace viewer for `trace.json`.

Keep changes scoped to solution code unless a task explicitly requires broader refactoring.

## Build, Test, and Development Commands
Use Python directly; no build system is required.
- `python perf_takehome.py`: run local `unittest` suite in `perf_takehome.py`.
- `python perf_takehome.py Tests.test_kernel_cycles`: run the main cycle benchmark quickly.
- `python tests/submission_tests.py`: run official submission checks and threshold tests.
- `python perf_takehome.py Tests.test_kernel_trace`: generate a fresh `trace.json`.
- `python watch_trace.py`: open the local hot-reloading trace viewer.
- `git diff origin/main tests/`: verify `tests/` is unchanged before submission.

## Coding Style & Naming Conventions
- Follow Python conventions: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- Keep instruction scheduling code readable: prefer small helpers, clear temporary names, and short comments for non-obvious dependency/packing decisions.
- Preserve deterministic behavior in test/debug paths (for example, seeded random flows when needed).

## Testing Guidelines
- Test framework is `unittest` (invoked through the commands above).
- Treat `tests/submission_tests.py` as the source of truth for correctness and performance.
- Do not modify files under `tests/`; validation expects that directory to match `origin/main`.
- When reporting improvements, include observed cycle count from `python tests/submission_tests.py`.

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects consistent with history (examples: `Vectorize kernel...`, `Pipeline dual-group rounds...`).
- Keep each commit focused on one optimization idea or refactor.
- PRs should include:
  - What changed and why
  - Before/after cycle counts
  - Confirmation that `tests/` is unchanged
  - Any trace evidence (`trace.json`/viewer observations) if relevant
