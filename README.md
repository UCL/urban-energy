# Urban Energy

Investigating the structural energy penalties of urban sprawl in England.

**Scope:** Domestic (residential) buildings only. Non-domestic EPCs are a separate dataset.

**Data:** EPC certificates with UPRN coordinates (available since November 2021). Earlier certificates lack spatial linkage and are excluded.

## Key Finding

Sprawling development locks in energy penalties that technology cannot eliminate:

| Lock-In Type   | Mechanism                              | Magnitude              |
| -------------- | -------------------------------------- | ---------------------- |
| **Floor area** | Detached houses are larger             | +59% floor area        |
| **Envelope**   | More exposed walls = more heat loss/m² | +53% kWh/m² (matched)  |
| **Transport**  | Car dependence = more vehicle-km       | +22% car ownership     |
| **Combined**   | All factors together                   | **+50% total penalty** |

**Policy message:** You cannot insulate and electrify your way out of sprawl.

See [analysis report](stats/analysis_report_v3.md) for detailed findings.

## Project Structure

| Folder                              | Purpose                                  |
| ----------------------------------- | ---------------------------------------- |
| [data/](data/README.md)             | Data acquisition (Census, EPC, LiDAR)    |
| [processing/](processing/README.md) | Building morphology extraction           |
| [stats/](stats/README.md)           | Statistical analysis and methodology     |
| [paper/](paper/README.md)           | Academic paper and literature review     |

## Quick Start

```bash
# Run complete analysis pipeline
uv run python stats/run_all.py
```

Output: [stats/analysis_report_v3.md](stats/analysis_report_v3.md)

## License

GPL-3.0-only
