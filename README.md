# Evidence-Aware News Ranking

**A code sample by Woojin (Jay) Park: source-specific features, candidate scoring, and grouped evaluation.**

Recommendation logs contain different kinds of evidence: stored preferences, retained recommendation traces, and observed clicks. This implementation keeps those inputs identifiable, constructs candidate-level interactions, and evaluates ranked lists within queries.

This standalone engineering sample is curated from an earlier research implementation. It is not the current PEC-Fuse manuscript release, and it makes no claim to reproduce that manuscript's results. Original Python identifiers (`pec_fuse`) are retained for traceability.

## Review in 10 minutes

| Start here | What to inspect |
|---|---|
| [Executable walkthrough](notebooks/01_ranking_walkthrough.ipynb) | Inputs → feature construction → model comparison → ranked outputs → uncertainty |
| [Feature builder](src/pec_fuse/features.py) | Required fields, retained-rank constraints, source-specific interactions |
| [Scoring and ranking](src/pec_fuse/ranking.py) | ExtraTrees pointwise probabilities and within-query ordering |
| [Evaluation](src/pec_fuse/evaluation.py) | NDCG, recall, MRR, paired user-cluster bootstrap |
| [MIND adapter](src/pec_fuse/mind.py) | Text-vector history summaries and explicitly labeled proxy features |

## Run locally

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_synthetic_demo.py
python -m unittest discover -s tests -v
```

For the notebook, install `jupyter` separately and open `notebooks/01_ranking_walkthrough.ipynb` from this repository. The CSV fixture is synthetic. Its tiny sample size illustrates interfaces, not model superiority or meaningful statistical inference.

## Verified locally

The synthetic CLI, all four walkthrough code cells, and four contract tests pass on Python 3.12 with the versions in [requirements-tested.txt](requirements-tested.txt). The full MIND dataset run has not been executed for this release.

## Design choices

- **Explicit evidence schema:** 25 ordered features, with an 11-feature operational baseline.
- **Candidate-level scoring:** an ExtraTrees classifier scores each item; scores are ordered within a query. This is a pointwise scorer, not a listwise training objective.
- **Grouped evaluation:** each query contributes equally to mean metrics. Bootstrap sampling resamples users, keeping each user's queries together.
- **Inspectable assumptions:** retained recommendation is not measured attention; observed clicks are not satisfaction; synthetic results are not research findings.

## Optional MIND example

Obtain MIND-small separately under its dataset terms, with `train/` and `dev/` containing `news.tsv` and `behaviors.tsv`. Data and pretrained weights are not bundled.

```bash
pip install -e '.[mind]'
python scripts/run_mind_proxy.py --mind-root data/MINDsmall --encoder sentence-transformers/all-MiniLM-L6-v2
```

This run downloads encoder weights and may require substantial memory/time. It is an illustrative proxy experiment, not a standard MIND benchmark: histories substitute for explicit preferences; impression-list order is used as a positional proxy, not verified display rank; sampled outside-slate items are weak negatives, not observed dislikes. Validation uses the last 15% of the supplied training rows, not a guaranteed chronological or user-disjoint split. Aggregate features for validation come only from the fitting partition; fitting-row statistics are still in-sample, not out-of-fold. A production study should use timestamp-aware or cross-fitted aggregates and explicit temporal splits. Full MIND execution is not included in the local smoke-test claim.

## Scope and provenance

Research implementation: Woojin (Jay) Park. Public sample curation adds documentation, a walkthrough, focused tests, finite-input validation, stable tie sorting, and validation-statistic isolation. See [provenance](docs/PROVENANCE.md). No manuscripts, submission archives, private logs, real identifiers, fitted private models, or paper-result tables are included.

[Portfolio](https://woojinpark-jay.github.io/portfolio/)
