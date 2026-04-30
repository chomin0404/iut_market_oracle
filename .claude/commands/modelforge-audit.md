---
description: ModelForge — tail the audit log and summarise recent events
argument-hint: [tail_n]
---
Display recent ModelForge audit entries.

Tail lines: $ARGUMENTS (default: 20)

Steps:
1. Run `uv run python -m src.models audit`.
2. Display each entry: timestamp | event | model_id | verification_overall.
3. Summarise:
   - Total entries in the log.
   - Models forged today.
   - Any FAIL-status verification events in the tail.
4. If the audit log does not exist, report that no forge runs have been executed yet.

Do not edit any file.
