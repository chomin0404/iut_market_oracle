---
description: Modify skill classification safely with tests and report impact
argument-hint: <classification change request>
---
Update the skill classification for this repository.

Task: $ARGUMENTS

Required workflow:
1. Read `src/huh_twin/skill_classification.py`, `src/huh_twin/skill_graph.py`, and the relevant tests.
2. Identify whether the change affects duplicates, counts, basis distribution, or Huh score assumptions.
3. Implement the smallest correct diff.
4. Update tests that encode counts, duplicates, or lookup behavior.
5. Run `make test-cov` and `make report`.
6. Summarize the changed counts, score implications, and files touched.
