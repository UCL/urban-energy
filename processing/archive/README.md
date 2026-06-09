# processing/archive

Frozen processing code. **Not imported by the live pipeline** — see
[ROADMAP.md](../../ROADMAP.md).

| File | Status |
| ---- | ------ |
| `pipeline_lsoa.py` | The v0 LSOA-level pipeline, superseded by [pipeline_oa.py](../pipeline_oa.py). Its shared constants and `run_stage1_morphology` (which the OA pipeline used to import from here) were extracted to the live module [processing/common.py](../common.py) on 2026-06-09, so `pipeline_oa.py` no longer imports from `archive/`. This file keeps its own copies and is otherwise inert. |
| `reprocess_collisions.py` | One-off migration utility (sanitised-name → `BUA22CD`-prefixed output dirs). Obsolete for fresh rebuilds. Archived 2026-06-09. |
