"""Utilities for formatting and persisting PosteriorSummary objects."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from schemas import PosteriorSummary, PriorSpec


def posterior_to_dict(
    posterior: PosteriorSummary,
    prior: PriorSpec | None = None,
    label: str = "",
) -> dict:
    """Return a plain-dict representation suitable for YAML/JSON serialisation.

    Parameters
    ----------
    posterior:
        The posterior summary to serialise.
    prior:
        Optional prior spec included for provenance.
    label:
        Human-readable name for this update (e.g. the prior name from configs/).
    """
    out: dict = {
        "label": label,
        "mean": round(posterior.mean, 8),
        "variance": round(posterior.variance, 10),
        "std": round(posterior.variance**0.5, 8),
        "credible_interval_95": {
            "lo": round(posterior.credible_interval_95[0], 8),
            "hi": round(posterior.credible_interval_95[1], 8),
        },
        "n_evidence": posterior.n_evidence,
        "updated_at": posterior.updated_at.isoformat(),
    }
    if prior is not None:
        out["prior"] = {
            "distribution": prior.distribution,
            "params": prior.params,
        }
    return out


def save_summary(
    posterior: PosteriorSummary,
    path: str | Path,
    prior: PriorSpec | None = None,
    label: str = "",
    fmt: str = "yaml",
) -> Path:
    """Write posterior summary to *path* in YAML or JSON format.

    Parameters
    ----------
    posterior:
        Summary to write.
    path:
        Destination file (parent dirs are created automatically).
    prior:
        Optional prior included for provenance.
    label:
        Human-readable identifier for the update.
    fmt:
        ``"yaml"`` (default) or ``"json"``.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    data = posterior_to_dict(posterior, prior=prior, label=label)

    if fmt == "json":
        dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        dest.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    return dest


def load_priors_yaml(path: str | Path) -> dict[str, PriorSpec]:
    """Load all prior specs from a YAML file (e.g. configs/priors.yaml).

    Expected YAML structure::

        bayesian_priors:
          my_prior:
            distribution: beta
            params: {alpha: 2.0, beta: 5.0}
            description: "..."

    Returns
    -------
    dict[str, PriorSpec]
        Mapping from prior name to validated PriorSpec.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    priors_raw: dict = raw.get("bayesian_priors", {})
    return {name: PriorSpec(**spec) for name, spec in priors_raw.items()}
