# Viva Preparation

Answers are intentionally concise; adapt them to the measured outputs in `PROJECT_REPORT.md`.

## Core questions

### 1. Why did you choose this project?
It combines data provenance, preprocessing, comparative classification, evaluation, explainability, a web interface, and privacy-aware history. The goal is a reproducible academic demonstration, not a clinical product.

### 2. What is machine learning?
Machine learning is a set of methods that estimate patterns from examples and apply those learned patterns to new inputs.

### 3. Why supervised learning?
Each selected dataset contains input variables and a defined target label, so the task is supervised classification.

### 4. Why classification?
The source targets represent two classes after documented binary mapping, such as the CKD versus non-CKD label.

### 5. What is a feature?
A feature is an input column used by a model. In this project each form is limited to fields present in that module's training data.

### 6. What is a target variable?
The target is the label the model learns to classify. Its definition comes from the source dataset and is not interchangeable across modules.

### 7. What is data leakage?
Leakage occurs when information unavailable at prediction time, or information from evaluation data, influences model fitting or selection.

### 8. How did you reduce leakage?
Imputation, encoding, and scaling are inside an sklearn Pipeline. Hyperparameter selection uses training folds, and exact duplicate feature rows are grouped into one split.

### 9. Why detect duplicate records?
Duplicates can inflate measured performance if copies cross train and test partitions. Identical feature rows are grouped instead of automatically removed because they may represent distinct records.

### 10. Why use grouped splitting?
It prevents identical feature vectors from appearing in both training and evaluation partitions, reducing an obvious source of optimistic leakage.

### 11. What is a Pipeline?
An sklearn Pipeline chains preprocessing and an estimator so each cross-validation fold fits transformations only on its training portion.

### 12. Why impute missing values?
Many estimators do not accept missing values. The numeric median and categorical most-frequent value are learned from training data only.

### 13. Why are some Pima zeros treated as missing?
The source documentation identifies zero-coded values for several measurements as physically implausible missing placeholders. Pregnancies=0 remains a valid value.

### 14. Why encode categorical fields?
Most estimators need numeric arrays. One-hot encoding represents nominal categories without inventing an ordinal distance.

### 15. Why scale numeric features?
Scaling helps distance- and margin-based models such as KNN and SVM, and can improve optimization for Logistic Regression.

### 16. What is Logistic Regression?
It models the log-odds of a class as a linear function of inputs and can provide a useful, comparatively interpretable baseline.

### 17. How does a Decision Tree work?
It recursively partitions the feature space using rules chosen to reduce impurity, producing an interpretable but potentially high-variance model.

### 18. Why use Random Forest?
It averages many bootstrapped trees with randomized feature selection, often reducing variance compared with one tree.

### 19. What is an SVM?
A Support Vector Machine seeks a separating boundary with a large margin; kernels allow nonlinear boundaries.

### 20. How does KNN classify?
KNN assigns a class using nearby training points. It depends on feature scaling and can be sensitive to distance and sample representation.

### 21. Why evaluate XGBoost?
It is a gradient-boosted tree candidate for tabular classification. It is included as a comparison, not assumed to be best.

### 22. What is cross-validation?
Cross-validation rotates held-out folds within training data to estimate how choices vary across different training subsets.

### 23. What is hyperparameter tuning?
It searches model settings such as tree depth or regularization strength. Here it uses GridSearchCV inside grouped training folds.

### 24. Why use a fixed random seed?
A fixed seed makes data splitting and stochastic model setup repeatable for the same data and software environment.

### 25. What is stratification?
Stratification aims to preserve class proportions across splits, which is useful when positive and negative counts differ.

### 26. What is accuracy?
Accuracy is the fraction of all records classified correctly. It can look high while an important minority class is missed.

### 27. What is precision?
Precision is the share of predicted positives that are true positives; low precision means more false alarms.

### 28. What is recall?
Recall or sensitivity is the share of actual positives detected; low recall means more false negatives.

### 29. Why is recall considered for medical-risk projects?
Missing a positive label can be consequential, so sensitivity must be examined. Increasing sensitivity may also increase false positives.

### 30. What is the F1 score?
F1 is the harmonic mean of precision and recall, balancing the two when both matter.

### 31. What is ROC-AUC?
ROC-AUC summarizes ranking across thresholds using true-positive and false-positive rates. It does not select a safe operating threshold.

### 32. What is average precision?
Average precision summarizes precision-recall performance and is useful when class prevalence is uneven.

### 33. What does a confusion matrix show?
It counts true negatives, false positives, false negatives, and true positives for the selected threshold.

### 34. What is a threshold?
It converts a score or probability estimate into a class decision. A threshold reflects trade-offs and should be chosen using training/validation data, not the final test set.

### 35. What is probability calibration?
A calibrated probability has empirical outcome frequencies close to the stated probability over comparable cases. This project does not claim clinical calibration.

### 36. What is a Brier score?
It is mean squared error between predicted probabilities and binary outcomes; lower is better, but it is affected by prevalence and calibration.

### 37. How did you address class imbalance?
Candidate classifiers use class weights where supported, and performance reports include recall and precision-recall metrics. No synthetic rows enter the test set.

### 38. What is overfitting?
Overfitting occurs when a model learns noise or sample-specific patterns that do not generalize to new records.

### 39. What is underfitting?
Underfitting occurs when a model is too simple or constrained to capture useful patterns in the training data.

### 40. What is feature selection?
Feature selection chooses a subset of input variables. It must be fitted inside training folds to avoid using validation/test outcomes.

### 41. What is SHAP?
SHAP assigns additive feature contributions relative to a reference distribution for a specific model output.

### 42. Does SHAP prove a feature causes disease?
No. SHAP explains model behavior under a chosen background; it is not causal inference or a medical conclusion.

### 43. Why use model-agnostic SHAP here?
The same permutation approach can explain different selected estimators. One-hot contributions are grouped back to the original dataset variables.

### 44. How does Flask serve predictions?
Flask routes accept requests, validate form values, call a prediction service that loads a saved pipeline, and render a result page.

### 45. Why use SQLite?
SQLite is a small local relational database with no separate server process, suitable for a single-user academic demonstration.

### 46. How are SQL injections reduced?
Database writes and reads use parameterized SQL values instead of concatenating user input into statements.

### 47. Why add CSRF protection?
CSRF tokens help ensure state-changing form submissions originated from the application session.

### 48. How are models saved?
Joblib serializes each fitted preprocessing-and-estimator pipeline; JSON metadata records its version, features, source hash, and evaluation summary.

### 49. What is the risk of Joblib files?
Joblib uses pickle-based serialization, which can execute code when loading malicious artifacts. The app must load only trusted project-generated files.

### 50. What happens during a prediction?
The server validates one input row, applies the saved transformations, predicts a dataset class and score, computes local contributions when possible, and stores the minimal history entry.

### 51. Why does the application store no name?
The demo only needs module, fields, output, timestamp, and model version for its history feature; names and contact details are unnecessary.

### 52. Why can the system not replace a doctor?
It has no clinical validation, uses narrow historical datasets, may be biased or miscalibrated, and cannot interpret an individual's full clinical context.

### 53. Why isn't there a general symptom checker?
The selected modules use different clinical or survey variables. A single symptom model would need a suitable multi-disease dataset and independent validation.

### 54. What is external validation?
It evaluates a frozen model on a separate population or site that was not used in training or model selection.

### 55. What is population shift?
Population shift occurs when feature distributions, prevalence, measurement practices, or relationships differ between training data and deployment users.

### 56. Why are the four module probabilities not comparable?
The modules have different cohorts, target labels, class prevalence, and modeling procedures, so the numeric outputs have different meanings.

### 57. What does the model version identify?
It identifies the selected trained artifact version and makes history records traceable to a reproducible pipeline manifest.

### 58. What is the difference between a global and local explanation?
A global summary describes contributions across a sample; a local explanation describes contributions for one submitted record relative to a reference.

## Difficult technical questions

### D1. Why does the holdout score remain valid if the saved artifact is later refit on all rows?
The reported holdout estimate was computed before the refit using a model selected only on training data. The full-data artifact is for deployment/demo after evaluation; its performance is not independently measured on data it has seen. External validation is still required.

### D2. How would you select a probability threshold without contaminating the test set?
Generate out-of-fold probabilities within development data, select a threshold using a predeclared objective or cost trade-off, lock it, then evaluate once on the untouched test set.

### D3. Why can very high CKD test metrics still be untrustworthy?
The test set is small and drawn from the same benchmark. Strong features may reflect the source diagnosis process, while population shift, label leakage, and sampling variation remain possible.

### D4. How does grouped CV reduce duplicate leakage when no patient ID exists?
Hash identical feature rows into groups and ensure each group stays in one fold. It prevents exact-vector copies crossing partitions but cannot identify different records from the same person.

### D5. What if missingness is not at random?
Median or mode imputation can be biased if missingness depends on the unobserved value or disease state. Analyze missingness patterns, report them, and validate alternative approaches without fitting on held-out data.

### D6. Why is recall-based model selection not automatically clinically safe?
Recall ignores false-positive burden, calibration, operational consequences, and clinical utility. A threshold and model require stakeholder-defined costs and prospective validation.

### D7. How do you aggregate one-hot SHAP values?
Explain the transformed estimator inputs, then sum encoded-category contributions belonging to each original variable. This preserves the additive contribution across the encodings.

### D8. Why keep synthetic oversampling out of the full pipeline?
If oversampling happens before splitting, near-copies can leak into validation/test data. If justified, it must be fitted only inside each training fold and evaluated on the original class distribution.

### D9. What is the difference between ranking quality and calibration?
ROC-AUC measures ranking across thresholds; calibration measures agreement between predicted probability and observed frequency. A model can rank well and be poorly calibrated.

### D10. What evidence would be needed before real clinical deployment?
Clear intended-use definition, representative external and prospective validation, calibration and threshold evaluation, subgroup/fairness analysis, clinical workflow review, security/privacy controls, and applicable regulatory review.
