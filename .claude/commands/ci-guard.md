---
description: Validate local quality gates before commit or PR
---
Run the repository quality gates and report status.

Required workflow:
1. Run `make lint`.
2. Run `make test-cov`.
3. Run `make report`.
4. Summarize pass/fail status, changed artifacts, and any threshold or CI issues.
5. If something fails, fix the root cause when safe and rerun the affected command.
