---
description: Review packaging and CLI readiness for release
---
Check whether the package is ready for external use.

Required workflow:
1. Review `pyproject.toml`, `README.md`, `Makefile`, `.github/workflows/ci.yml`, and `src/huh_twin/report_skill_portfolio.py`.
2. Verify that `project.scripts`, license metadata, dev dependencies, and CI commands are aligned.
3. Identify missing packaging or documentation issues.
4. If asked to implement fixes, make the smallest possible patch.
5. Summarize release readiness and remaining gaps.
