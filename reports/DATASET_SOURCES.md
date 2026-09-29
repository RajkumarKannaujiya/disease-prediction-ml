# Dataset sources, fields, and measured quality

Generated: 2026-09-29

## Diabetes

- **Dataset:** Pima Indians Diabetes Database
- **Source:** https://search.r-project.org/CRAN/refmans/mlbench/html/PimaIndiansDiabetes.html
- **Source citation/license:** Acquired from the CRAN mlbench package (GPL-2); retain attribution and license when redistributing the packaged copy.
- **Target:** Published binary Outcome label (0/1).
- **Source records:** 768; **modeling rows:** 768; **exact duplicate feature/target rows:** 0 (kept and grouped for splitting).
- **Observed target counts (0/1):** `{'0': 500, '1': 268}`.
- **Source file SHA-256:** `e920964eb717a3749833979532bb31110d9ae9788ea0768bc889f91a0b756f39`.
- **Population limitation:** The source cohort is women aged 21 or older of Pima heritage; results do not establish performance for other populations.

### Features

| Feature | Meaning | Type | Missing values | IQR screening flags |
|---|---|---|---:|---:|
| `Pregnancies` | Number of recorded pregnancies. | float64 | 0 | 4 |
| `Glucose` | Plasma glucose concentration from the 2-hour oral glucose tolerance test. | float64 | 5 | 0 |
| `BloodPressure` | Diastolic blood pressure in mm Hg. | float64 | 35 | 14 |
| `SkinThickness` | Triceps skinfold thickness in millimetres. | float64 | 227 | 3 |
| `Insulin` | 2-hour serum insulin measurement. | float64 | 374 | 24 |
| `BMI` | Body mass index. | float64 | 11 | 8 |
| `DiabetesPedigreeFunction` | Dataset pedigree-function score related to family history. | float64 | 0 | 29 |
| `Age` | Age in years. | float64 | 0 | 9 |

### Preprocessing plan actually applied

- Source-specific target mapping; raw input schema checked against the selected dataset columns.
- Exact duplicate detection; rows are retained and identical feature vectors are grouped into outer and cross-validation partitions.
- Pima zero-coded Glucose, BloodPressure, SkinThickness, Insulin, and BMI values are treated as missing; Pregnancies=0 remains valid.
- Numeric missing values use training-fold median imputation and scaling; categorical missing values use training-fold most-frequent imputation and one-hot encoding.
- IQR flags are descriptive. No automatic outlier removal, target-informed selection, or test-set preprocessing is applied.
- Train/test is grouped and stratified with a fixed seed; all tuning is restricted to grouped CV within training data.

## Heart disease

- **Dataset:** UCI Heart Disease (Cleveland processed subset)
- **Source:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Source citation/license:** CC BY 4.0; cite the UCI dataset and its contributors.
- **Target:** Binary target: source num=0 versus source num>0.
- **Source records:** 303; **modeling rows:** 303; **exact duplicate feature/target rows:** 0 (kept and grouped for splitting).
- **Observed target counts (0/1):** `{'0': 164, '1': 139}`.
- **Source file SHA-256:** `548eaf3b9d47152c9e0d25c167dc629740e8a4a13366a9be8fbd789ac3d2ebae`.
- **Population limitation:** The benchmark is small and historical. Its target reflects the source study definition and is not a current clinical diagnosis.

### Features

| Feature | Meaning | Type | Missing values | IQR screening flags |
|---|---|---|---:|---:|
| `age` | Age in years. | int64 | 0 | 0 |
| `sex` | Dataset sex code (0/1). | object | 0 | 0 |
| `cp` | Chest-pain category code. | object | 0 | 0 |
| `trestbps` | Resting blood pressure in mm Hg. | int64 | 0 | 9 |
| `chol` | Serum cholesterol in mg/dL. | int64 | 0 | 5 |
| `fbs` | Indicator for fasting blood sugar above the source threshold. | object | 0 | 0 |
| `restecg` | Resting electrocardiographic category code. | object | 0 | 0 |
| `thalach` | Maximum heart rate achieved. | int64 | 0 | 1 |
| `exang` | Exercise-induced angina indicator. | object | 0 | 0 |
| `oldpeak` | ST depression induced by exercise relative to rest. | float64 | 0 | 5 |
| `slope` | Slope category of the peak exercise ST segment. | object | 0 | 0 |
| `ca` | Number of major vessels colored by fluoroscopy. | object | 4 | 0 |
| `thal` | Thalassemia category code. | object | 2 | 0 |

### Preprocessing plan actually applied

- Source-specific target mapping; raw input schema checked against the selected dataset columns.
- Exact duplicate detection; rows are retained and identical feature vectors are grouped into outer and cross-validation partitions.
- Pima zero-coded Glucose, BloodPressure, SkinThickness, Insulin, and BMI values are treated as missing; Pregnancies=0 remains valid.
- Numeric missing values use training-fold median imputation and scaling; categorical missing values use training-fold most-frequent imputation and one-hot encoding.
- IQR flags are descriptive. No automatic outlier removal, target-informed selection, or test-set preprocessing is applied.
- Train/test is grouped and stratified with a fixed seed; all tuning is restricted to grouped CV within training data.

## Liver disease

- **Dataset:** UCI Indian Liver Patient Dataset (ILPD)
- **Source:** https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+data
- **Source citation/license:** CC BY 4.0; cite the UCI dataset and its contributors.
- **Target:** Published Selector class (disease versus non-disease; mapping verified during download).
- **Source records:** 583; **modeling rows:** 583; **exact duplicate feature/target rows:** 13 (kept and grouped for splitting).
- **Observed target counts (0/1):** `{'0': 167, '1': 416}`.
- **Source file SHA-256:** `192dfdf52c2e310713f6fc9d8402ca5355ffd2a56a30bc66621ddf50a9b46798`.
- **Population limitation:** The 583-record cohort is regional and has unequal representation by sex; external validity is unknown.

### Features

| Feature | Meaning | Type | Missing values | IQR screening flags |
|---|---|---|---:|---:|
| `age` | Age in years. | int64 | 0 | 0 |
| `gender` | Source gender category. | object | 0 | 0 |
| `total_bilirubin` | Total bilirubin measurement. | float64 | 0 | 84 |
| `direct_bilirubin` | Direct bilirubin measurement. | float64 | 0 | 81 |
| `alkphos` | Alkaline phosphatase measurement. | int64 | 0 | 69 |
| `sgpt` | SGPT / alanine aminotransferase measurement. | int64 | 0 | 73 |
| `sgot` | SGOT / aspartate aminotransferase measurement. | int64 | 0 | 66 |
| `total_proteins` | Total protein measurement. | float64 | 0 | 8 |
| `albumin` | Albumin measurement. | float64 | 0 | 0 |
| `ag_ratio` | Albumin-to-globulin ratio. | float64 | 4 | 10 |

### Preprocessing plan actually applied

- Source-specific target mapping; raw input schema checked against the selected dataset columns.
- Exact duplicate detection; rows are retained and identical feature vectors are grouped into outer and cross-validation partitions.
- Pima zero-coded Glucose, BloodPressure, SkinThickness, Insulin, and BMI values are treated as missing; Pregnancies=0 remains valid.
- Numeric missing values use training-fold median imputation and scaling; categorical missing values use training-fold most-frequent imputation and one-hot encoding.
- IQR flags are descriptive. No automatic outlier removal, target-informed selection, or test-set preprocessing is applied.
- Train/test is grouped and stratified with a fixed seed; all tuning is restricted to grouped CV within training data.

## Chronic kidney disease

- **Dataset:** UCI Chronic Kidney Disease
- **Source:** https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- **Source citation/license:** CC BY 4.0; cite the UCI dataset and its contributors.
- **Target:** Published class: ckd versus notckd.
- **Source records:** 400; **modeling rows:** 400; **exact duplicate feature/target rows:** 0 (kept and grouped for splitting).
- **Observed target counts (0/1):** `{'0': 150, '1': 250}`.
- **Source file SHA-256:** `35b9acfe7a600abc32b0cba139351c5a34d203cf8f0dbc20c275a8cd9e53a594`.
- **Population limitation:** The 400-record cohort was collected over a short period at a single source; missingness and population shift limit generalization.

### Features

| Feature | Meaning | Type | Missing values | IQR screening flags |
|---|---|---|---:|---:|
| `age` | Age in years. | float64 | 9 | 10 |
| `bp` | Blood pressure in mm Hg. | float64 | 12 | 36 |
| `sg` | Urine specific-gravity category. | object | 47 | 0 |
| `al` | Urine albumin category. | object | 46 | 0 |
| `su` | Urine sugar category. | object | 49 | 0 |
| `rbc` | Red-blood-cell test category. | object | 152 | 0 |
| `pc` | Pus-cell test category. | object | 65 | 0 |
| `pcc` | Pus-cell-clumps category. | object | 4 | 0 |
| `ba` | Bacteria category. | object | 4 | 0 |
| `bgr` | Random blood glucose measurement. | float64 | 44 | 34 |
| `bu` | Blood urea measurement. | float64 | 19 | 38 |
| `sc` | Serum creatinine measurement. | float64 | 17 | 51 |
| `sod` | Sodium measurement. | float64 | 87 | 16 |
| `pot` | Potassium measurement. | float64 | 88 | 4 |
| `hemo` | Hemoglobin measurement. | float64 | 52 | 1 |
| `pcv` | Packed cell volume. | float64 | 71 | 1 |
| `wc` | White blood cell count. | float64 | 106 | 10 |
| `rc` | Red blood cell count. | float64 | 131 | 1 |
| `htn` | Hypertension history category. | object | 2 | 0 |
| `dm` | Diabetes mellitus history category. | object | 2 | 0 |
| `cad` | Coronary artery disease history category. | object | 2 | 0 |
| `appet` | Appetite category. | object | 1 | 0 |
| `pe` | Pedal-edema category. | object | 1 | 0 |
| `ane` | Anemia category. | object | 1 | 0 |

### Preprocessing plan actually applied

- Source-specific target mapping; raw input schema checked against the selected dataset columns.
- Exact duplicate detection; rows are retained and identical feature vectors are grouped into outer and cross-validation partitions.
- Pima zero-coded Glucose, BloodPressure, SkinThickness, Insulin, and BMI values are treated as missing; Pregnancies=0 remains valid.
- Numeric missing values use training-fold median imputation and scaling; categorical missing values use training-fold most-frequent imputation and one-hot encoding.
- IQR flags are descriptive. No automatic outlier removal, target-informed selection, or test-set preprocessing is applied.
- Train/test is grouped and stratified with a fixed seed; all tuning is restricted to grouped CV within training data.
