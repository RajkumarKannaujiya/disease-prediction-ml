"""Dataset-backed input schemas and model metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    kind: str = "number"
    minimum: float | None = None
    maximum: float | None = None
    step: str = "any"
    options: tuple[str, ...] = ()
    help_text: str = ""
    option_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Disease:
    key: str
    title: str
    short_description: str
    dataset_name: str
    dataset_source: str
    dataset_license: str
    target_description: str
    population_limit: str
    features: tuple[Feature, ...]
    class_0_label: str = "Class 0 label in the source dataset"
    class_1_label: str = "Class 1 label in the source dataset"


DISEASES: dict[str, Disease] = {
    "diabetes": Disease(
        key="diabetes",
        title="Diabetes",
        short_description="A dataset-based classification using demographic and measurement fields.",
        dataset_name="Pima Indians Diabetes Database",
        dataset_source="https://search.r-project.org/CRAN/refmans/mlbench/html/PimaIndiansDiabetes.html",
        dataset_license="Acquired from the CRAN mlbench package (GPL-2); retain attribution and license when redistributing the packaged copy.",
        target_description="Published binary Outcome label (0/1).",
        population_limit="The source cohort is women aged 21 or older of Pima heritage; results do not establish performance for other populations.",
        features=(
            Feature("Pregnancies", "Pregnancies", minimum=0, maximum=30, step="1", help_text="Number recorded in the source dataset."),
            Feature("Glucose", "Plasma glucose (2-hour test)", minimum=1, maximum=400, help_text="Recorded in mg/dL. Choose ‘I don't know’ if this value is unavailable; do not enter zero as a substitute."),
            Feature("BloodPressure", "Diastolic blood pressure", minimum=1, maximum=250, help_text="Recorded in mm Hg."),
            Feature("SkinThickness", "Triceps skinfold thickness", minimum=1, maximum=150, help_text="Recorded in millimeters."),
            Feature("Insulin", "2-hour serum insulin", minimum=1, maximum=2000, help_text="Use the source dataset's recorded value."),
            Feature("BMI", "Body mass index", minimum=1, maximum=100, help_text="Recorded as kilograms per square meter."),
            Feature("DiabetesPedigreeFunction", "Diabetes pedigree function", minimum=0, maximum=20, help_text="Dataset-specific score; it is not a percentage."),
            Feature("Age", "Age in years", minimum=21, maximum=120, step="1", help_text="The source cohort starts at age 21."),
        ),
        class_0_label="No diabetes label in the source dataset",
        class_1_label="Diabetes label in the source dataset",
    ),
    "heart": Disease(
        key="heart",
        title="Heart disease",
        short_description="An estimate based on the Cleveland heart disease benchmark fields.",
        dataset_name="UCI Heart Disease (Cleveland processed subset)",
        dataset_source="https://archive.ics.uci.edu/dataset/45/heart+disease",
        dataset_license="CC BY 4.0; cite the UCI dataset and its contributors.",
        target_description="Binary target: source num=0 versus source num>0.",
        population_limit="The benchmark is small and historical. Its target reflects the source study definition and is not a current clinical diagnosis.",
        features=(
            Feature("age", "Age in years", minimum=18, maximum=120, step="1", help_text="Recorded in years."),
            Feature("sex", "Sex in source dataset", kind="select", options=("0", "1"), help_text="Choose the category as coded in this historical dataset.", option_labels=("0 · Female", "1 · Male")),
            Feature("cp", "Chest pain type", kind="select", options=("1", "2", "3", "4"), help_text="These are the Cleveland dataset categories.", option_labels=("1 · Typical angina", "2 · Atypical angina", "3 · Non-anginal pain", "4 · Asymptomatic")),
            Feature("trestbps", "Resting blood pressure", minimum=20, maximum=300, help_text="Recorded in mm Hg."),
            Feature("chol", "Serum cholesterol", minimum=1, maximum=1500, help_text="Recorded in mg/dL."),
            Feature("fbs", "Fasting blood sugar above 120 mg/dL", kind="select", options=("0", "1"), help_text="Choose the category in the source record.", option_labels=("0 · No", "1 · Yes")),
            Feature("restecg", "Resting ECG category", kind="select", options=("0", "1", "2"), help_text="Choose the recorded category.", option_labels=("0 · Normal", "1 · ST-T abnormality", "2 · Left ventricular hypertrophy")),
            Feature("thalach", "Maximum heart rate achieved", minimum=1, maximum=300, help_text="Use the value recorded in the source report."),
            Feature("exang", "Exercise-induced angina", kind="select", options=("0", "1"), help_text="Choose the category in the source record.", option_labels=("0 · No", "1 · Yes")),
            Feature("oldpeak", "ST depression (oldpeak)", minimum=0, maximum=20, help_text="Use the value recorded in the source report."),
            Feature("slope", "Peak exercise ST slope", kind="select", options=("1", "2", "3"), help_text="Choose the recorded category.", option_labels=("1 · Upsloping", "2 · Flat", "3 · Downsloping")),
            Feature("ca", "Major vessels colored by fluoroscopy", kind="select", options=("0", "1", "2", "3"), help_text="Choose the recorded count.", option_labels=("0 vessels", "1 vessel", "2 vessels", "3 vessels")),
            Feature("thal", "Thal category", kind="select", options=("3", "6", "7"), help_text="Choose the historical dataset category.", option_labels=("3 · Normal", "6 · Fixed defect", "7 · Reversible defect")),
        ),
        class_0_label="No heart disease label in the source dataset",
        class_1_label="Heart disease label in the source dataset",
    ),
    "liver": Disease(
        key="liver",
        title="Liver disease",
        short_description="An estimate using the Indian Liver Patient Dataset fields.",
        dataset_name="UCI Indian Liver Patient Dataset (ILPD)",
        dataset_source="https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+data",
        dataset_license="CC BY 4.0; cite the UCI dataset and its contributors.",
        target_description="Published Selector class (disease versus non-disease; mapping verified during download).",
        population_limit="The 583-record cohort is regional and has unequal representation by sex; external validity is unknown.",
        features=(
            Feature("age", "Age in years", minimum=0, maximum=120, step="1", help_text="Recorded in years."),
            Feature("gender", "Gender", kind="select", options=("Female", "Male")),
            Feature("total_bilirubin", "Total bilirubin", minimum=0, maximum=1000, help_text="Use the value and unit shown in the source report."),
            Feature("direct_bilirubin", "Direct bilirubin", minimum=0, maximum=1000, help_text="Use the value and unit shown in the source report."),
            Feature("alkphos", "Alkaline phosphatase", minimum=0, maximum=10000, help_text="Also called ALP; use the recorded lab value."),
            Feature("sgpt", "SGPT / ALT", minimum=0, maximum=10000, help_text="Use the recorded lab value."),
            Feature("sgot", "SGOT / AST", minimum=0, maximum=10000, help_text="Use the recorded lab value."),
            Feature("total_proteins", "Total proteins", minimum=0, maximum=100, help_text="Use the value and unit shown in the source report."),
            Feature("albumin", "Albumin", minimum=0, maximum=100, help_text="Use the value and unit shown in the source report."),
            Feature("ag_ratio", "Albumin / globulin ratio", minimum=0, maximum=100, help_text="Use the recorded ratio."),
        ),
        class_0_label="No liver disease label in the source dataset",
        class_1_label="Liver disease label in the source dataset",
    ),
    "kidney": Disease(
        key="kidney",
        title="Chronic kidney disease",
        short_description="An estimate using the UCI chronic kidney disease benchmark fields.",
        dataset_name="UCI Chronic Kidney Disease",
        dataset_source="https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease",
        dataset_license="CC BY 4.0; cite the UCI dataset and its contributors.",
        target_description="Published class: ckd versus notckd.",
        population_limit="The 400-record cohort was collected over a short period at a single source; missingness and population shift limit generalization.",
        features=(
            Feature("age", "Age in years", minimum=0, maximum=120, step="1", help_text="Recorded in years."),
            Feature("bp", "Blood pressure", minimum=1, maximum=350, help_text="Recorded in mm Hg."),
            Feature("sg", "Specific gravity", kind="select", options=("1.005", "1.010", "1.015", "1.020", "1.025"), help_text="Choose the exact category recorded in the source data."),
            Feature("al", "Albumin category", kind="select", options=("0", "1", "2", "3", "4", "5"), help_text="Choose the category from the source report; do not guess.", option_labels=("Category 0", "Category 1", "Category 2", "Category 3", "Category 4", "Category 5")),
            Feature("su", "Sugar category", kind="select", options=("0", "1", "2", "3", "4", "5"), help_text="Choose the category from the source report; do not guess.", option_labels=("Category 0", "Category 1", "Category 2", "Category 3", "Category 4", "Category 5")),
            Feature("rbc", "Red blood cells", kind="select", options=("normal", "abnormal"), help_text="Choose the recorded result."),
            Feature("pc", "Pus cells", kind="select", options=("normal", "abnormal"), help_text="Choose the recorded result."),
            Feature("pcc", "Pus cell clumps", kind="select", options=("present", "notpresent"), help_text="Choose the recorded result."),
            Feature("ba", "Bacteria", kind="select", options=("present", "notpresent"), help_text="Choose the recorded result."),
            Feature("bgr", "Random blood glucose", minimum=0, maximum=2000, help_text="The dataset records this value in mg/dL."),
            Feature("bu", "Blood urea", minimum=0, maximum=2000, help_text="Use the value and unit shown in the source report."),
            Feature("sc", "Serum creatinine", minimum=0, maximum=100, help_text="Use the value and unit shown in the source report."),
            Feature("sod", "Sodium", minimum=0, maximum=1000, help_text="Use the value and unit shown in the source report."),
            Feature("pot", "Potassium", minimum=0, maximum=100, help_text="Use the value and unit shown in the source report."),
            Feature("hemo", "Hemoglobin", minimum=0, maximum=100, help_text="Use the value and unit shown in the source report."),
            Feature("pcv", "Packed cell volume", minimum=0, maximum=100, help_text="Use the value and unit shown in the source report."),
            Feature("wc", "White blood cell count", minimum=0, maximum=1000000, help_text="Use the count shown in the source report."),
            Feature("rc", "Red blood cell count", minimum=0, maximum=100, help_text="Use the count shown in the source report."),
            Feature("htn", "Hypertension", kind="select", options=("yes", "no")),
            Feature("dm", "Diabetes mellitus", kind="select", options=("yes", "no")),
            Feature("cad", "Coronary artery disease", kind="select", options=("yes", "no")),
            Feature("appet", "Appetite", kind="select", options=("good", "poor")),
            Feature("pe", "Pedal edema", kind="select", options=("yes", "no")),
            Feature("ane", "Anemia", kind="select", options=("yes", "no")),
        ),
        class_0_label="No CKD label in the source dataset",
        class_1_label="CKD label in the source dataset",
    ),
}


def get_feature_columns(disease_key: str) -> list[str]:
    """Return ordered feature names for a disease module."""
    return [feature.name for feature in DISEASES[disease_key].features]


def schema_to_dict(disease: Disease) -> dict[str, Any]:
    """Convert an immutable disease schema into JSON-serializable metadata."""
    return {
        "key": disease.key,
        "title": disease.title,
        "features": [feature.__dict__ for feature in disease.features],
        "dataset": disease.dataset_name,
        "source": disease.dataset_source,
        "license": disease.dataset_license,
        "target": disease.target_description,
        "population_limit": disease.population_limit,
        "class_0_label": disease.class_0_label,
        "class_1_label": disease.class_1_label,
    }
