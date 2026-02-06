# Urban Energy

Investigating the structural energy penalties of urban sprawl in England.

**Dataset:** 173,907 domestic EPC certificates with UPRN coordinates (Greater Manchester). SAP-modelled potential energy demand, not metered consumption.

## Thesis

Sprawling development locks in energy penalties that persist regardless of technology:

| Lock-In        | Mechanism                       | Evidence                                   | Magnitude      |
| -------------- | ------------------------------- | ------------------------------------------ | -------------- |
| **Floor area** | Detached houses are larger      | All types compared                         | +59%           |
| **Envelope**   | More exposed walls per m2       | Matched by construction era and floor area | +53% kWh/m2    |
| **Transport**  | Low density = car dependence    | Census density quartiles                   | +22% cars/hh   |
| **Combined**   | Floor area + transport together | High-density flats vs low-density houses   | **+50% total** |

Each shared wall eliminates ~17% of heat loss — a geometric relationship that no amount of insulation can change. Technology reduces absolute demand but the proportional penalty persists.

See [analysis report](stats/analysis_report_v3.md) for detailed findings.

## Project Structure

| Folder                              | Purpose                               |
| ----------------------------------- | ------------------------------------- |
| [data/](data/README.md)             | Data acquisition (Census, EPC, LiDAR) |
| [processing/](processing/README.md) | Building morphology extraction        |
| [stats/](stats/README.md)           | Statistical analysis and methodology  |
| [paper/](paper/README.md)           | Academic paper and literature review  |

## Quick Start

```bash
# Run complete analysis pipeline
uv run python stats/run_all.py
```

Output: [stats/analysis_report_v3.md](stats/analysis_report_v3.md)

## License

GPL-3.0-only
