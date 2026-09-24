# MAGIC Gamma Telescope Particle Classification using Logistic Regression

A machine learning project implementing a binary classification pipeline using Logistic Regression to differentiate between gamma telescope signals (`g`) and hadron background noise (`h`) using the MAGIC Gamma Telescope dataset.

---

## 📌 Project Overview

The objective of this project is to classify high-energy gamma-ray events against hadronic background showers observed by an atmospheric Cherenkov telescope. 

* **Algorithm:** Logistic Regression (Scikit-Learn)
* **Dataset:** MAGIC Gamma Telescope Dataset (`magic04.data`)
* **Task Type:** Binary Classification
* **Evaluation Metrics:** Precision, Recall, F1-Score, Accuracy

---

## 📊 Dataset & Feature Description

The dataset consists of 19,020 instances collected to simulate registration of high-energy gamma particles in an atmospheric Cherenkov telescope.

### Features

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `fLength` | Continuous | Major axis of ellipse [mm] |
| `fWidth` | Continuous | Minor axis of ellipse [mm] |
| `fSize` | Continuous | 10-log of sum of content of all pixels [in #phot] |
| `conc` | Continuous | Ratio of sum of two highest pixels to size |
| `fConcl` | Continuous | Ratio of highest pixel to size |
| `fAsym` | Continuous | Distance from highest pixel to center, projected onto major axis [mm] |
| `fM3Long` | Continuous | 3rd root of 3rd moment along major axis [mm] |
| `fM3Trans` | Continuous | 3rd root of 3rd moment along minor axis [mm] |
| `fAlpha` | Continuous | Angle of major axis with vector to origin [deg] |
| `fDist` | Continuous | Distance from origin to center of ellipse [mm] |
| **`class`** | Categorical | Target variable: `g` (Gamma signal -> mapped to `1`), `h` (Hadron background -> mapped to `0`) |

---

## 🛠️ Pipeline Architecture & Methodology

1. **Data Preprocessing & Encoding:**
   * Parsed raw headerless dataset and assigned standard feature names.
   * Binary encoded target classes (`g` = `1`, `h` = `0`).

2. **Data Splitting:**
   * Split dataset into **Train (60%)**, **Validation (20%)**, and **Test (20%)** sets.

3. **Feature Scaling & Oversampling:**
   * Applied `StandardScaler` to normalize feature representations across distributions.
   * Addressed class imbalance on the training dataset using `RandomOverSampler` (`imblearn`).

4. **Model Training:**
   * Trained Scikit-Learn `LogisticRegression(max_iter=1000)` on the resampled/scaled training set.

---

## 📈 Experimental Results

Model performance on the unseen test dataset:

```text
              precision    recall  f1-score   support

           0       0.76      0.60      0.67      1344
           1       0.80      0.90      0.85      2460

    accuracy                           0.79      3804
   macro avg       0.78      0.75      0.76      3804
weighted avg       0.79      0.79      0.79      3804
