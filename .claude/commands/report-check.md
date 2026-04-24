---
description: Regenerate the markdown report and verify artifacts
---
Verify reporting end to end.

Required workflow:
1. Read `src/huh_twin/reporting.py`, `src/huh_twin/report_skill_portfolio.py`, and `configs/dependency_edges.yaml` only if needed.
2. Run `make report`.
3. Confirm that `output/skill_portfolio_report.md` exists.
4. If the report content changed, summarize what changed in Huh score, basis distribution, or dependency notes.
5. If generation fails, diagnose the root cause and fix it.
