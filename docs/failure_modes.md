# Failure Modes

## Purpose
This document lists the major failure modes of the Huh Bayesian Valuation Twin.
Its purpose is to prevent false confidence, hidden assumptions, and silent model drift.

## 1. Skill Graph Engine

### Failure mode
Dependency concentration looks low even though skills are conceptually redundant.

### Cause
The graph is only as good as the declared edges.
If related skills are mislabeled as independent, the score becomes misleading.

### Effect
The portfolio appears diversified but is structurally narrow.

### Detection
- Review item taxonomy manually.
- Compare basis diversity to semantic overlap.
- Audit highly similar labels.

### Mitigation
- Add semantic redundancy checks.
- Require evidence for low-dependency claims.
- Periodically refactor the skill map.

## 2. Bayesian Engine

### Failure mode
Posterior estimates look precise but are driven mostly by arbitrary priors.

### Cause
Weak data and strong priors can produce narrow but misleading posteriors.

### Effect
Scenario confidence is overstated.

### Detection
- Compare posterior to prior.
- Run prior sensitivity checks.
- Report observation count alongside posterior summaries.

### Mitigation
- Version priors in config files.
- Run alternative prior sets.
- Flag low-data situations.

## 3. Valuation Engine

### Failure mode
Enterprise value is dominated by terminal value.

### Cause
DCF often depends heavily on terminal assumptions, especially when discount rate and terminal growth are close.

### Effect
Small assumption errors create large valuation swings.

### Detection
- Report terminal value share of EV.
- Run two-way sensitivity on growth and discount rate.
- Compare DCF to reverse DCF outputs.

### Mitigation
- Cap terminal growth conservatively.
- Require sensitivity tables in every report.
- Add scenario-based cross-checks.

## 4. Reverse DCF

### Failure mode
Implied growth is numerically correct but economically implausible.

### Cause
Reverse DCF solves for consistency with price, not realism.

### Effect
The output may be mistaken for a forecast.

### Detection
- Compare implied growth to market size and reinvestment constraints.
- Check moat and margin feasibility.

### Mitigation
- Pair reverse DCF with business-model constraints.
- Add sanity bounds and commentary.

## 5. Sensitivity Analysis

### Failure mode
Sensitivity tables create false comfort.

### Cause
Local parameter sweeps can miss regime shifts, nonlinear breaks, and dependency changes.

### Effect
The model seems robust inside a narrow window but fails outside it.

### Detection
- Compare local sensitivity to scenario analysis.
- Track structural break indicators.

### Mitigation
- Add regime switching.
- Add entropy monitoring.
- Expand stress test ranges.

## 6. Reporting Layer

### Failure mode
Reports look complete but omit key assumptions.

### Cause
Generated markdown can summarize outputs without enough context.

### Effect
Readers trust numbers without understanding the model boundary.

### Detection
- Check every report for assumptions, method, outputs, sensitivity, and next actions.
- Review missing sections as failures, not style issues.

### Mitigation
- Enforce report templates.
- Fail tests if mandatory sections are absent.

## 7. Claude Code Workflow

### Failure mode
Implementation appears valid because it is syntactically clean, but logic is wrong.

### Cause
Agentic coding can produce plausible code that lacks real verification.

### Effect
Silent regressions and false completion.

### Detection
- Require tests before implementation changes when practical.
- Run lint, tests, and type checks after edits.
- Inspect diffs for unintended file changes.

### Mitigation
- Keep tasks small.
- Use acceptance criteria.
- Treat tests as the external oracle.

## 8. Global Failure Mode

### Failure mode
The system becomes too complex to interpret.

### Cause
Too many modules, distributions, and mathematical ideas are added before integration is stable.

### Effect
The model becomes impressive-looking but decision-useless.

### Detection
- Count assumptions.
- Count modules without tests.
- Track the ratio of decision-relevant outputs to total generated outputs.

### Mitigation
- Freeze scope when integration tests are failing.
- Define a concrete decision use case for each new module before building it.
- Retire components that produce no decision-relevant output.
- Prefer integration over expansion.
