# Poly-Oracle v1 Skeleton

Phase 1 provides a clean baseline with a minimal CLI boot path and package layout for future migration work.

## Current root layout

- `cli.py`
- `bot/`
- `tests/`
- `pyproject.toml`
- `requirements.txt`
- `.env.example`
- `README.md`
- `legacy/`

## Run

```bash
python cli.py start --help
python cli.py start
```

## Smoke test

```bash
python -m pytest tests/test_cli_smoke.py -q
```

## Migration note

Legacy code was moved instead of deleted to keep migration safe and auditable:

- `legacy/phase1_snapshot/bot_pre_cleanup/`: previous `bot/` implementation before skeleton reset.
- `legacy/phase1_snapshot/tests_pre_cleanup/`: previous `tests/` suite before smoke-only reset.
- `legacy/phase1_snapshot/tools/test`: previous root `test` helper script.
- `legacy/phase1_snapshot/root_deleted_snapshot/`: archived content from tracked files that were removed from root paths (for example old `src/`, `config/`, and historical tests).
- Existing prior archives under `legacy/src/` and `legacy/config/` remain unchanged.
- `legacy/phase1_snapshot/MOVED_FILES.txt`: exact file-level manifest for this Phase 1 move.

Reason: isolate obsolete paths under `legacy/` so v1 can evolve from a small, stable surface without irreversible deletions.
