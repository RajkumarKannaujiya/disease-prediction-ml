# Dataset acquisition and provenance

The project includes the acquired raw CSV snapshots in `data/raw/` so training and evaluation can be reproduced without relying on a remote dataset changing. Run `python scripts/download_data.py` from the project root only when intentionally refreshing sources; it overwrites the snapshot files and updates their SHA-256 manifest in `data/raw/source_manifest.json`.

| Module | Source | Record scope | License / attribution |
|---|---|---:|---|
| Diabetes | CRAN `mlbench` 2.1-6 `PimaIndiansDiabetes2`; its documentation identifies NIDDK as original owner and UCI as the historic repository source | 768 | The CRAN package is GPL-2. Retain the package/source attribution and license notice with redistributed dataset copies. |
| Heart | UCI dataset 45, Cleveland processed subset | 303 | CC BY 4.0; cite Janosi, Steinbrunn, Pfisterer, and Detrano, DOI 10.24432/C52P4X. |
| Liver | UCI dataset 225, ILPD | 583 | CC BY 4.0; cite Ramana and Venkateswarlu, DOI 10.24432/C5D02C. |
| Kidney | UCI dataset 336, Chronic Kidney Disease | 400 | CC BY 4.0; cite Rubini, Soundarapandian, and Eswaran, DOI 10.24432/C5G020. |

Raw source snapshots are preserved in `data/raw/`; derived field-level profiles are written to `data/processed/`. Each source retains its own license and attribution; the root MIT license applies to project code only. Do not treat class labels, clinical variables, or source populations as equivalent across modules. See `reports/DATASET_SOURCES.md` for measured missingness and target counts after acquisition.
