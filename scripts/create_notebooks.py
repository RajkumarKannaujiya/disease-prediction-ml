"""Generate clean, runnable Jupyter notebooks for the documented workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def build_notebook(title: str, introduction: str, cells: list[str], destination: str) -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11+"},
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{introduction}"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json, sys\nROOT = Path.cwd()\nif not (ROOT / 'src').exists(): ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT / 'src'))"
        ),
        *[nbf.v4.new_code_cell(cell) for cell in cells],
    ]
    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOKS / destination)


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    build_notebook(
        "Data exploration and quality audit",
        "Load the acquired source files, inspect real dimensions, dtypes, missingness, duplicates, target balance, and feature distributions. The notebook does not invent or resample medical records.",
        [
            "from disease_prediction.data import DATASET_KEYS, load_dataset\n\nquality = {}\nfor key in DATASET_KEYS:\n    X, y, report = load_dataset(key, ROOT / 'data' / 'raw')\n    quality[key] = report\n    print(key, X.shape, y.value_counts().sort_index().to_dict())",
            "import pandas as pd\n\nmissing = pd.DataFrame({key: load_dataset(key, ROOT / 'data' / 'raw')[0].isna().sum() for key in DATASET_KEYS}).fillna(0).astype(int)\nmissing",
            "from disease_prediction.training import _save_eda\n\nfor key in DATASET_KEYS:\n    X, y, _ = load_dataset(key, ROOT / 'data' / 'raw')\n    summary = _save_eda(key, X, y, ROOT / 'reports')\n    print(key, summary['shape'], summary['class_imbalance_ratio'])",
        ],
        "01_data_exploration.ipynb",
    )
    build_notebook(
        "Preprocessing without leakage",
        "Demonstrate grouped splitting and pipeline-contained imputers, encoders, and scalers. The held-out fold is never used to fit a transformer.",
        [
            "import pandas as pd\nfrom sklearn.model_selection import StratifiedGroupKFold\nfrom disease_prediction.data import load_dataset\nfrom disease_prediction.preprocessing import make_pipeline\nfrom disease_prediction.training import _candidate_models, SEED\n\nX, y, report = load_dataset('kidney', ROOT / 'data' / 'raw')\ngroups = pd.util.hash_pandas_object(X.astype('string'), index=False).astype(str)\ntrain_idx, test_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups=groups))\nX_train, X_test = X.iloc[train_idx], X.iloc[test_idx]\ny_train, y_test = y.iloc[train_idx], y.iloc[test_idx]\nprint('train', X_train.shape, 'test', X_test.shape, 'training-only fitting follows')",
            "pipeline = make_pipeline(X_train, _candidate_models(y_train)['Logistic Regression'][0])\npipeline.fit(X_train, y_train)\nprint('transformed feature count:', len(pipeline.named_steps['preprocess'].get_feature_names_out()))\nprint('holdout rows transformed:', pipeline.named_steps['preprocess'].transform(X_test).shape[0])",
        ],
        "02_preprocessing.ipynb",
    )
    for key, title, destination in [
        ("diabetes", "Diabetes model comparison", "03_diabetes_model.ipynb"),
        ("heart", "Heart disease model comparison", "04_heart_model.ipynb"),
        ("liver", "Liver disease model comparison", "05_liver_model.ipynb"),
        ("kidney", "Kidney disease model comparison", "06_kidney_model.ipynb"),
    ]:
        build_notebook(
            title,
            "Run the same grouped five-fold hyperparameter search and fixed held-out evaluation used by the command-line trainer. This cell may take several minutes; all printed metrics come from actual execution.",
            [
                "from disease_prediction.training import train_one\nimport json\n\nmanifest_path = ROOT / 'data' / 'raw' / 'source_manifest.json'\nmanifest = json.loads(manifest_path.read_text(encoding='utf-8'))\nsource_hashes = {item['dataset_key']: item['sha256'] for item in manifest}\nresult = train_one('" + key + "', ROOT / 'data' / 'raw', ROOT / 'models', ROOT / 'reports' / 'model_results', ROOT / 'reports', source_hashes)\nprint(result['selected_model'], result['holdout_metrics'])",
            ],
            destination,
        )
    print(f"Created six notebooks in {NOTEBOOKS}")


if __name__ == "__main__":
    main()
