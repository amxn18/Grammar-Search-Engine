# SHL Grammar Scoring Engine

A machine learning system for predicting the **grammar score (0–5)** of spoken English recordings using acoustic speech features.

> **SHL Hiring Assessment 2026 — Research Intern**

## Overview

The task is to predict a continuous grammar score for spoken English audio recordings. The dataset contains **769 labelled training recordings** and **216 test recordings**.

The project follows an experiment-driven approach: simple baselines were established first, acoustic and linguistic approaches were evaluated separately, hybrid approaches were tested, and the final model was selected using both validation performance and Kaggle hidden-test performance.

The final selected solution is an **ExtraTrees acoustic regression model** using 37 engineered audio features.

### Final Results

| Metric | Result |
|---|---:|
| Training RMSE | **0.2472** |
| Validation RMSE | **0.7380** |
| Validation Pearson | **0.8010** |
| Kaggle Public Score | **0.7418** |
| Training samples | 769 |
| Test samples | 216 |

---

## Problem Statement

Given a spoken English audio recording, predict a continuous grammar score between **0 and 5**.

The competition evaluates predictions using:

- **RMSE (Root Mean Squared Error)** — lower is better.
- **Pearson Correlation** — higher is better.

The goal is therefore to learn a regression function that produces accurate grammar scores while maintaining a strong correlation with the true scores.

---

## Dataset

The dataset contains:

```text
Dataset_Final/
├── train/
│   └── *.wav
├── test/
│   └── *.wav
├── train.csv
├── test.csv
└── sample_submission.csv
```

### Dataset size

- Training recordings: **769**
- Test recordings: **216**
- Audio sampling rate: **16 kHz**
- Audio channels: **Mono**

The observed training labels were:

```text
0.0
1.0
1.5
2.0
2.5
3.0
3.5
4.0
4.5
5.0
```

### Target statistics

| Statistic | Value |
|---|---:|
| Minimum | 0.0 |
| Maximum | 5.0 |
| Mean | 3.3134 |
| Median | 3.0 |
| Standard deviation | 1.2390 |

---

## Project Pipeline

The final pipeline is:

```text
                         AUDIO
                           │
                           ▼
                 Acoustic Feature
                    Extraction
                           │
                           ▼
                  37 Acoustic
                     Features
                           │
                           ▼
                  ExtraTrees
                   Regressor
                           │
                           ▼
                   Grammar Score
                       0–5
```

The final model is intentionally acoustic-only. Linguistic approaches were investigated during experimentation but did not improve the hidden Kaggle score.

---

# Exploratory Data Analysis

## Audio Characteristics

The training audio recordings were inspected for duration, sampling rate, channels and general data quality.

Duration statistics:

| Statistic | Duration |
|---|---:|
| Mean | 55.78 sec |
| Standard deviation | 7.91 sec |
| Minimum | 20.06 sec |
| Median | 60.07 sec |
| Maximum | 61.04 sec |

All inspected recordings were mono 16 kHz audio.

Rather than blindly cropping all recordings to a fixed length, the available audio was preserved and duration was included as a feature.

## Data Quality Observation

A group of recordings with score `0.0` contained severe distortion/noise when manually inspected. These recordings were retained rather than removed because they are part of the original labelled distribution and similar cases could potentially occur in the test set.

---

# Validation Strategy

A fixed train/validation split was used throughout the experiments to ensure fair comparison between models.

```text
Training samples:   615
Validation samples: 154
Validation size:    20%
Random state:       42
```

Because some target values were extremely rare, a stratification group was created so that sufficiently frequent labels were stratified while extremely rare labels were grouped into a `rare` category.

The same split was used for model comparison.

---

# Baseline

A constant mean predictor was used as the initial benchmark.

The baseline predicts the mean training target for every validation recording.

```text
Validation RMSE = 1.2269
Pearson = Undefined
```

Pearson correlation is undefined for constant predictions because the predictions have zero variance.

---

# Acoustic Feature Engineering

The final model uses **37 engineered acoustic features** extracted with `librosa`.

## Features

### Signal / Temporal Features

- Audio duration
- RMS energy mean
- RMS energy standard deviation
- Zero-crossing rate mean
- Zero-crossing rate standard deviation

### Spectral Features

- Spectral centroid mean and standard deviation
- Spectral bandwidth mean and standard deviation
- Spectral rolloff mean and standard deviation

### MFCC Features

13 MFCC coefficients were calculated.

For each MFCC:

- Mean
- Standard deviation

were extracted.

This produces:

```text
37 acoustic features
```

---

# Models and Experiments

The project evaluated multiple approaches rather than assuming that a more complex architecture would necessarily perform better.

## 1. Acoustic Ridge Regression

Standardized acoustic features were used with Ridge Regression.

| Metric | Score |
|---|---:|
| Training RMSE | 0.8146 |
| Validation RMSE | 0.8407 |
| Validation Pearson | 0.7295 |

---

## 2. Acoustic Random Forest

A Random Forest Regressor was trained on the acoustic features.

```python
RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
```

Results:

| Metric | Score |
|---|---:|
| Training RMSE | 0.3733 |
| Validation RMSE | 0.7658 |
| Validation Pearson | 0.7823 |

---

# Linguistic Experiments

Because grammar is fundamentally related to language, a separate linguistic branch was investigated.

The linguistic pipeline was:

```text
Audio
  ↓
Whisper Base
  ↓
Transcript
  ↓
all-MiniLM-L6-v2
  ↓
384-dimensional embedding
  ↓
Ridge Regression
```

## Whisper

Whisper Base was used for speech-to-text transcription.

Transcriptions were checkpointed during processing so that expensive transcription did not need to be repeated.

## TF-IDF

TF-IDF was evaluated using word and character n-grams with Ridge Regression.

| Metric | Score |
|---|---:|
| Training RMSE | 0.9363 |
| Validation RMSE | 1.1032 |
| Validation Pearson | 0.5936 |

## Sentence Embeddings

`all-MiniLM-L6-v2` produced 384-dimensional transcript embeddings.

| Metric | Score |
|---|---:|
| Training RMSE | 1.0103 |
| Validation RMSE | 1.0567 |
| Validation Pearson | 0.6000 |

The linguistic models were weaker than the acoustic models when used independently.

---

# Hybrid Experiments

## Feature-Level Fusion

The 37 acoustic features were concatenated with the 384-dimensional MiniLM embeddings:

```text
37 acoustic features
+
384 embedding features
=
421 features
```

A Ridge model was trained on the combined representation.

| Metric | Score |
|---|---:|
| Training RMSE | 0.4158 |
| Validation RMSE | 0.8895 |
| Validation Pearson | 0.7150 |

This did not outperform the acoustic Random Forest.

## Prediction-Level Fusion

A prediction-level ensemble was also tested:

```text
90% Acoustic Random Forest
+
10% Sentence-Embedding Ridge
```

Validation result:

| Metric | Score |
|---|---:|
| Validation RMSE | **0.7618** |
| Validation Pearson | **0.7905** |

This improved both validation metrics over the Random Forest alone, but the hidden Kaggle evaluation later showed that this hybrid did not generalize as well as the final ExtraTrees acoustic model.

---

# ExtraTrees Optimization

ExtraTrees was evaluated as an alternative tree-based acoustic model.

Configuration:

```python
ExtraTreesRegressor(
    n_estimators=400,
    min_samples_leaf=3,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
```

Validation results:

```text
Validation RMSE    = 0.737952
Validation Pearson = 0.801046
```

Rounded:

```text
Validation RMSE    = 0.7380
Validation Pearson = 0.8010
```

This improved over the Random Forest on both validation metrics.

---

# Kaggle Submission Experiments

Three submissions were evaluated on the hidden Kaggle test set.

| Version | Model | Public Score |
|---|---|---:|
| V1 | Random Forest + MiniLM 90/10 | 0.6848 |
| V2 | ExtraTrees + MiniLM 90/10 | 0.6762 |
| **V3** | **ExtraTrees Acoustic Only** | **0.7418** |

### Key finding

The local validation results alone suggested that the hybrid approach could be useful. However, the Kaggle hidden evaluation showed that the **ExtraTrees acoustic-only model generalized substantially better**.

This was an important model-selection result: increasing model complexity does not necessarily improve real-world generalization.

The final model was therefore selected based on the strongest hidden-test performance.

---

# Final Model

## ExtraTrees Regressor

The final model uses the 37 acoustic features described above.

```python
ExtraTreesRegressor(
    n_estimators=400,
    min_samples_leaf=3,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
```

The model was retrained using all **769 labelled training recordings** before generating test predictions.

---

# Final Training Performance

The final ExtraTrees model was trained on the complete training dataset.

```text
Training RMSE = 0.2472
```

Exact value:

```text
0.24723352832483975
```

The training RMSE is reported separately from validation and Kaggle performance because it is measured on the data used to fit the final model.

---

# Final Validation Performance

The final ExtraTrees model achieved:

```text
Validation RMSE    = 0.7380
Validation Pearson = 0.8010
```

These values were obtained using the fixed 20% validation split before final retraining on all training data.

---

# Final Test Predictions

The final model generated predictions for all **216 test recordings**.

Prediction statistics:

| Statistic | Value |
|---|---:|
| Number of predictions | 216 |
| Minimum | 2.4846 |
| Maximum | 4.3049 |
| Mean | 3.2441 |
| Missing predictions | 0 |
| Unique filenames | 216 |

All predictions were constrained to the valid score range:

```text
0 ≤ prediction ≤ 5
```

---

# Submission Validation

The final submission was validated with:

```python
assert len(submission_v3) == len(test_df)
assert submission_v3["filename"].is_unique
assert submission_v3["label"].notna().all()
assert submission_v3["label"].between(0, 5).all()
```

All checks passed.

Final submission:

```text
Rows:              216
Unique filenames:  216
Missing values:     0
```

Submission file:

```text
submission_v3_extratrees_only.csv
```

---

# Kaggle Result

The final selected submission was:

```text
Submission: submission_v3_extratrees_only.csv
Model: ExtraTrees Acoustic Only
Public Score: 0.7418
Status: COMPLETE
```

Kaggle Submission ID:

```text
56885398
```

---

# Final Results

| Metric | Final Result |
|---|---:|
| Training RMSE | **0.2472** |
| Validation RMSE | **0.7380** |
| Validation Pearson | **0.8010** |
| Kaggle Public Score | **0.7418** |
| Training recordings | **769** |
| Test recordings | **216** |
| Acoustic features | **37** |

---

# Technologies Used

- Python
- NumPy
- Pandas
- Librosa
- Scikit-learn
- PyTorch
- OpenAI Whisper
- Sentence Transformers
- Matplotlib
- Kaggle API

---

# Reproducibility

The main experiments use:

```text
Random state = 42
```

Intermediate artifacts were checkpointed during expensive processing, including:

- Acoustic feature files
- Whisper transcripts
- Sentence embeddings

This reduces the need to repeat expensive audio and transcription processing.

---

# Project Structure

The repository currently contains the main competition notebook:

```text
Grammar-Search-Engine/
└── Grammar_Search_Engine_SHL.ipynb
```

The notebook contains the complete experiment history, feature engineering, model evaluation, final model selection and submission workflow.

---

# Key Learnings

### 1. Acoustic information was highly predictive

MFCC, energy, zero-crossing and spectral statistics provided a strong signal for this dataset.

### 2. More complex does not always mean better

The linguistic branch and hybrid models were useful experiments, but the final hidden-test performance favored the simpler acoustic-only ExtraTrees model.

### 3. Local validation is not enough

The hybrid model performed well on the local validation set, but the Kaggle hidden evaluation preferred ExtraTrees alone.

This demonstrates why model selection should consider robust validation and, when available, genuine held-out evaluation.

### 4. Experimentation should be empirical

Instead of assuming that a multimodal architecture would win, multiple approaches were evaluated and the final model was selected from observed results.

---

# Final Architecture

```text
                         ┌─────────────┐
                         │    Audio    │
                         └──────┬──────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Acoustic Extraction │
                    └──────────┬──────────┘
                               │
                               ▼
                       37 Acoustic
                          Features
                               │
                               ▼
                    ┌─────────────────────┐
                    │ ExtraTrees Regressor│
                    └──────────┬──────────┘
                               │
                               ▼
                       Grammar Score
                            0 – 5
```

---

# Conclusion

The final system is an acoustic speech regression pipeline built with engineered audio features and an ExtraTrees Regressor.

After evaluating multiple acoustic, linguistic and hybrid approaches, the ExtraTrees acoustic-only model achieved the strongest hidden-test performance.

The final Kaggle public score was:

# **0.7418**

The project demonstrates a complete machine-learning workflow from data exploration and feature engineering through model comparison, validation, final training and hidden-test evaluation.
'''
