# Paper: The Structural Energy Penalties of Urban Sprawl

Academic paper quantifying lock-in effects in building and transport energy.

**Scope:** Domestic (residential) buildings only.

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
