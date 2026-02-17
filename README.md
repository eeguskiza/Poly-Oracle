# Poly-Oracle v1 Startup Baseline

Phase 2 implements a deterministic, fail-fast startup orchestration for `python cli.py start`.

## Config Format Choice

Startup config uses JSON (`bot/config/startup.json`).

Justification:
- JSON parsing is available in Python standard library (`json`), so startup has no extra parser dependency risk.
- JSON types are strict and deterministic for schema validation.
- Error locations (line/column) are clear when parsing fails.

## Startup Pipeline (`start`)

Order executed:
1. Load `.env` and JSON config.
2. Validate strict settings schema.
3. Initialize structured logger.
4. Initialize SQLite state store.
5. Resolve BTC target market (single-market mode).
6. Initialize primary + fallback data feed placeholders.
7. Warm up feed buffers.
8. Initialize trading loop coordinator scaffold.
9. Register graceful shutdown handlers.
10. Exit with explicit code/reason on fatal conditions, or continue loop / return preflight success.

## Commands

```bash
python3 cli.py start --help
python3 cli.py start --check
python3 cli.py start
```

## Tests

```bash
./venv/bin/python -m pytest tests -q
```

## Migration Note

Legacy code remains archived under `legacy/` and was not deleted to preserve migration safety:
- `legacy/phase1_snapshot/bot_pre_cleanup/`
- `legacy/phase1_snapshot/tests_pre_cleanup/`
- `legacy/phase1_snapshot/root_deleted_snapshot/`
- `legacy/phase1_snapshot/MOVED_FILES.txt`
- Existing previous archives under `legacy/src/` and `legacy/config/`
