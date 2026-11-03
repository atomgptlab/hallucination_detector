# Analysis notebook and figures

The notebook and raw tool exports behind the figures and tables of the paper
(originally released by H. Neralla). Run from this directory:

```
pip install pandas numpy matplotlib jupyter
jupyter notebook DetectorTool_Report_main.ipynb
```

| File | Description |
|------|-------------|
| `DetectorTool_Report_main.ipynb` | Analysis notebook: loads the NeurIPS export and produces all NeurIPS figures |
| `hallucination_report_2.csv` | Raw per-citation export of the NeurIPS 2025 run from the web tool |
| `ui_all_cols_results.csv` | Cleaned NeurIPS results table written by the notebook |
| `citeaudit_all300_combined.csv` | Raw export of the 300-citation CiteAudit run (one API error) |
| `materials_test_results.csv` | Raw export of the 26 materials-reference-type run |
| `ui_status_distribution.png` | Status distribution (66 / 28 / 6) |
| `ui_component_by_status.png` | Mean component score by status |
| `ui_title_match_dist.png` | Title score distribution |
| `ui_author_match_dist.png` | Author score distribution |
| `ui_journal_match_dist.png` | Journal score distribution |

The same runs, in the runner's JSON format with per-field scores and matched
candidates, are in [`../results/`](../results); the input citations are in
[`../data/`](../data).
