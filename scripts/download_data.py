"""Download cited public benchmark datasets to data/raw.

The script does not redistribute data with the project. It records provenance
and hashes so a later run can be audited against the acquired files.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import pyreadr
from ucimlrepo import fetch_ucirepo


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MLBENCH_VERSION = "2.1-6"
PIMA_URL = f"https://cran.r-project.org/src/contrib/Archive/mlbench/mlbench_{MLBENCH_VERSION}.tar.gz"
UCI_IDS = {"heart": 45, "liver": 225, "kidney": 336}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_frame(frame: pd.DataFrame, key: str, source: str, license_text: str) -> dict:
    path = RAW / f"{key}.csv"
    frame.to_csv(path, index=False)
    return {
        "dataset_key": key,
        "source": source,
        "license": license_text,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": sha256(path),
        "file": path.name,
    }


def download_pima() -> dict:
    request = Request(PIMA_URL, headers={"User-Agent": "academic-ml-project/1.0"})
    with urlopen(request, timeout=45) as response:
        payload = response.read()
    if not payload or len(payload) < 1000:
        raise RuntimeError("The CRAN mlbench package returned an unexpectedly small archive.")
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "mlbench.tar.gz"
        archive_path.write_bytes(payload)
        with tarfile.open(archive_path, "r:gz") as archive:
            matches = [member for member in archive.getmembers() if member.name.endswith("/data/PimaIndiansDiabetes2.rda")]
            if len(matches) != 1:
                raise RuntimeError("PimaIndiansDiabetes2.rda was not present in the pinned mlbench archive.")
            rda_file = Path(temporary) / "PimaIndiansDiabetes2.rda"
            extracted = archive.extractfile(matches[0])
            if extracted is None:
                raise RuntimeError("The Pima R data file could not be read from the package archive.")
            rda_file.write_bytes(extracted.read())
        frame = pyreadr.read_r(str(rda_file))["PimaIndiansDiabetes2"].copy()
    frame.columns = [str(column) for column in frame.columns]
    rename = {
        "pregnant": "Pregnancies", "glucose": "Glucose", "pressure": "BloodPressure",
        "triceps": "SkinThickness", "insulin": "Insulin", "mass": "BMI",
        "pedigree": "DiabetesPedigreeFunction", "age": "Age", "diabetes": "Outcome",
    }
    frame = frame.rename(columns=rename)
    if len(frame) != 768 or frame["Outcome"].nunique() != 2:
        raise RuntimeError(f"Unexpected Pima data shape/target: {frame.shape}; refusing to continue.")
    return save_frame(
        frame,
        "diabetes",
        f"CRAN mlbench {MLBENCH_VERSION} PimaIndiansDiabetes2; original source NIDDK via UCI legacy archive.",
        "mlbench package is GPL-2; retain the package attribution and GPL-2 notice when redistributing this copy.",
    )


def download_uci(key: str, dataset_id: int) -> dict:
    dataset = fetch_ucirepo(id=dataset_id)
    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()
    if features is None or target is None or features.empty or target.empty:
        raise RuntimeError(f"UCI dataset {dataset_id} returned empty data.")
    frame = pd.concat([features, target], axis=1)
    meta = dataset.metadata or {}
    doi = meta.get("doi", "")
    source = f"UCI dataset {dataset_id}; DOI {doi}".strip()
    license_text = "CC BY 4.0 per UCI dataset metadata."
    return save_frame(frame, key, source, license_text)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifests = [download_pima()]
    for key, dataset_id in UCI_IDS.items():
        manifests.append(download_uci(key, dataset_id))
    manifest_path = RAW / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    for item in manifests:
        print(f"{item['dataset_key']}: {item['records']} rows; sha256={item['sha256']}")
    print(f"Source manifest written to {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"Dataset acquisition failed: {exc}", file=sys.stderr)
        raise
