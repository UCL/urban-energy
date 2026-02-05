# Paper: The Structural Energy Penalties of Urban Sprawl

Sprawling development locks in higher energy demand through three structural mechanisms — floor area, building envelope, and transport distance — that technology cannot eliminate. This paper quantifies each mechanism using EPC and Census data from England, showing that the proportional penalty persists across technology scenarios.

## Documents

| File                                         | Description                                 |
| -------------------------------------------- | ------------------------------------------- |
| [main.tex](main.tex)                         | Full academic paper (LaTeX)                 |
| [references.bib](references.bib)             | BibTeX bibliography                         |
| [literature_review.md](literature_review.md) | Detailed literature review                  |
| [methodology_notes.md](methodology_notes.md) | Working notes on metrics and data decisions |

## Building

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Related

- [Project overview](../README.md) — Key findings and project structure
- [Statistical methodology](../stats/README.md) — Research design and analysis approach
- [Analysis report](../stats/analysis_report_v3.md) — Detailed empirical findings
