---
description: Change dependency edges through YAML and verify report effects
argument-hint: <dependency change request>
---
Modify dependency structure using configuration-first workflow.

Task: $ARGUMENTS

Required workflow:
1. Read `configs/dependency_edges.yaml`, `src/huh_twin/dependency_config.py`, `src/huh_twin/skill_graph.py`, and reporting/tests if needed.
2. Prefer editing YAML instead of Python unless schema changes are required.
3. Explain expected impact on dependency concentration and Huh score before editing.
4. Apply the change.
5. Run `make report` and, if behavior changed materially, `make test-cov`.
6. Summarize before/after expectations and generated artifacts.
