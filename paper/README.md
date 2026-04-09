# Paper: Urban Form and Energy

## Canonical Document

The current working narrative is **[case_v2.md](case_v2.md)** — the OA-level
NEPI analysis covering 6,687 English Built-Up Areas at Output Area resolution
(198,779 OAs).

The headline result: the median flat-dominant OA scores Band A
(15,982 kWh/hh/yr); the median detached-dominant OA scores Band F
(26,897 kWh/hh/yr). The 10,915 kWh/hh/yr gap decomposes as Form 45% /
Mobility 43% / Access penalty 14%.

Regenerate all figures with:

```bash
uv run python stats/build_case_oa.py             # three surfaces + basket
uv run python stats/nepi.py                      # NEPI scorecard
uv run python stats/access_penalty_model.py      # empirical penalty
```

## Documents

| File | Description | Status |
| ---- | ----------- | ------ |
| [case_v2.md](case_v2.md) | OA-level NEPI case narrative (198,779 OAs, 6,687 BUAs) — full IMRaD draft with robustness section | **Current** |
| [data.md](data.md) | Data sources, OA-level methodology, and limitations | Current |
| [literature_review.md](literature_review.md) | Thematic literature review | Current |
| [references.bib](references.bib) | BibTeX bibliography | Partial |

## Archived

The following are preserved in `archive/` for reference:

| File | Description |
| ---- | ----------- |
| [archive/case_v1.md](archive/case_v1.md) | LSOA-level case narrative (18 cities, 3.7k LSOAs) |
| [archive/main.tex](archive/main.tex) | Academic paper (LaTeX) — stale, not reconciled with OA results |

## Related

- [Project overview](../README.md)
- [Statistical analysis](../stats/README.md) — NEPI scorecard, planning tool, figures
- [Processing pipeline](../processing/README.md) — national OA pipeline
- [Data acquisition](../data/README.md) — download/prep scripts
- [TODO.md](../TODO.md) — current status and forward work
- [notes/](../notes/) — archived v0 working notes (LSOA-era)
