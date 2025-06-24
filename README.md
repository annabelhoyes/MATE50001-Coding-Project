# MATE50001: Glass Classification Using Machine Learning

**Group Members:**  
Annabel Hoyes, Clotilde Offner, James Ormsby, Sophia Reid, Romain Santeiu  

---

## Project Overview

This project applies machine learning to classify different types of glass based on their elemental composition. Data is sourced from forensic databases across the UK, US, and EU, and processed through a full ML pipeline including:

- Data merging from heterogeneous formats (`.csv`, `.json`)
- Pre-processing (outlier handling, scaling, imputing)
- Classification using SVM with grid search hyperparameter tuning
- Visualisation of data distributions and classifier performance

---

## Key Technologies

- Python 3.x
- `scikit-learn` (SVM, KNN, GridSearchCV)
- `pandas` / `numpy`
- `matplotlib` (interactive buttons and plots)

---

## Dataset Description

Glass samples were obtained from:
- UK (`glass_uk.csv`)
- EU (`glass_eu.json`)
- US (`glass_us.json`) — includes grouped data by department (`FBI`, `CIA`, etc.)

Each entry includes:
- Elemental composition (`Na`, `Mg`, `Al`, etc.)
- Refractive Index (`RI`)
- Glass Type (classification target)

---

## Preprocessing Steps

1. **Outlier Removal**: Based on IQR thresholds
2. **Scaling**: Min-Max (preferred) or Standard scaling
3. **Imputing**: Using `KNNImputer` with `n_neighbors = 3` for missing values

---

## Classifier Design

- Primary model: SVM (`rbf` & `linear` kernel)  
- Evaluation metric: F1 Score (weighted)  
- Optimization: GridSearchCV to tune `C` and `gamma`  
- Validation: StratifiedShuffleSplit for consistent folds

---

## Visualisation

Two interactive plots using `matplotlib`:
1. **Elemental Composition vs. N**:
   - Toggle elements and glass types
   - Switch between raw/processed/evidence data views

2. **Classifier Performance**:
   - Bar chart of F1 scores for each SVM configuration
   - Dynamically updates when elements are toggled

---

## Evidence Classification

An additional file `samples.evidence` is used to test the classifier against unknown samples. Predictions are displayed in the terminal.

---

## How to Run

1. **Install requirements**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
