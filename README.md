# amazon-hackathon-ytechno

# Business Entity Resolution

ML-based solution for resolving business entities across multiple noisy and independent data sources.

## 📌 Problem Statement

This project addresses a Business Entity Resolution (ER) problem.

We are given business records from three independent sources:

- **Source 1 (S1):** Deduplicated reference source
- **Source 2 (S2):** Noisy business records
- **Source 3 (S3):** Noisy business records

For every Source 1 entity, the goal is to identify **all matching records from Source 2 and Source 3** that refer to the same real-world business.

A Source 1 entity may have:

- Zero matches
- One match
- Multiple matches

The data contains noisy and inconsistent business names, addresses, and other fields.

## 🎯 Objective

Build a machine learning pipeline that can:

1. Clean and normalize business records
2. Generate plausible candidate matches using blocking
3. Calculate similarity and structured features
4. Predict whether candidate pairs represent the same business
5. Optimize predictions for the challenge's **F₀.₅ score**
6. Generate valid submission files

## 🧠 Planned Pipeline

```text
Raw Data
   ↓
Data Exploration
   ↓
Normalization / Preprocessing
   ↓
Candidate Generation / Blocking
   ↓
Similarity & Structured Features
   ↓
ML Matching Model
   ↓
Probability / Match Score
   ↓
F₀.₅ Threshold Optimization
   ↓
Final Entity Matches
   ↓
Submission Files
```

## 🔍 Key Challenges

The dataset may contain:

- Name abbreviations
- Legal suffix variations
- Typos
- Punctuation differences
- Word-order variations
- Transliteration differences
- Missing address components
- Address formatting variations
- Landmark-based addresses
- Businesses with similar or identical names
- Entities with no matching records

## 🤖 Machine Learning Approach

We will begin with a simple baseline and iteratively improve it.

### Baseline

- Fuzzy string similarity
- TF-IDF / cosine similarity
- Structured matching features
- Logistic Regression

### Main Model

We plan to evaluate tree-based models such as:

- CatBoost
- XGBoost

### Additional Experiments

Depending on validation performance, we may investigate:

- Character n-gram similarity
- Advanced blocking strategies
- Semantic embeddings
- Multilingual sentence-transformer features
- Additional domain-specific features
- Threshold optimization

Model selection will be based on validation performance rather than model complexity.

## 🚧 Candidate Generation / Blocking

Blocking is used to reduce the number of unnecessary pair comparisons while preserving true matches.

Potential blocking strategies include:

- Country-based blocking
- Name-token blocking
- Address-token blocking
- PIN/city-based blocking
- Character similarity based blocking
- Multiple blocking rules combined through candidate-set union

The candidate-generation stage is treated as a critical component because a true match cannot be recovered by the final model if it was excluded during blocking.

## 📊 Evaluation

The challenge evaluates submissions using **F₀.₅**, which gives greater importance to precision than recall.

We will evaluate:

- Precision
- Recall
- F₀.₅
- Blocking recall
- False positives
- False negatives
- Singleton prediction performance

Threshold selection will be performed using validation data.

## 📁 Project Structure

```text
business-entity-resolution/
│
├── data/
│   ├── train/
│   └── test/
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── blocking.py
│   ├── candidate_generation.py
│   ├── similarity.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_blocking.ipynb
│   ├── 04_features.ipynb
│   └── 05_model_comparison.ipynb
│
├── tests/
│
├── models/
│
├── experiments/
│
├── output/
│
├── requirements.txt
├── README.md
└── Documentation_template.md
```

## 👥 Team

| Member | Role | Responsibilities |
| **Jashwanth** | Preprocessing / Blocking Lead | Data cleaning, normalization, blocking, candidate generation |
| **Nishant** | ML / Feature Lead | Similarity features, TF-IDF, ML models, feature experiments |
| **Vikass** | Support | Testing, error analysis, supplementary features, documentation |
| **Sharvesh** | Integration | EDA, evaluation, F₀.₅, threshold tuning, architecture, integration |

## 🔀 Git Workflow

We use a shared GitHub repository with the following branch structure:

```text
main
  │
  └── develop
        │
        ├── feature/sharvesh-eda
        ├── feature/nishant-ml
        ├── feature/jashwanth-blocking
        └── feature/vikass-qa
```

### Rules

- `main` contains stable versions only.
- `develop` is the integration branch.
- Each team member works primarily on their own feature branch.
- Changes are pushed through Pull Requests.
- Code should be reviewed before merging into `develop`.
- `main` should only contain tested and stable versions.

## 🧪 Experiments

Experiments will be documented in `experiments/`.

Each experiment should record:

```text
Experiment ID
Blocking strategy
Features used
Model
Threshold
Validation F₀.₅
Precision
Recall
Observations
```

This allows us to compare approaches objectively and reproduce successful configurations.

## 📦 Submission Outputs

The final pipeline must generate:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

### matching_results.tsv

Contains the final matches for every Source 1 test entity.

### candidate_pairs.tsv

Contains the candidate set generated by the blocking/candidate-generation stage.

Final predicted matches must be a subset of the generated candidate set.

## 🧪 Validation

Before submission, validate the generated files using the challenge-provided validator:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## 🚫 Data Usage Restrictions

This project uses only the data provided for the challenge.

External business/entity lookup or data augmentation will not be used, including:

- Commercial entity-resolution APIs
- Government business databases
- Geocoding APIs
- External business datasets
- Internet-based business identity lookup

## 📌 Current Status

### Phase 1 — Reconnaissance

- [ ] Dataset exploration
- [ ] Ground-truth analysis
- [ ] Noise-pattern analysis
- [ ] Normalization design
- [ ] Blocking design
- [ ] Feature design

### Phase 2 — Baseline

- [ ] Candidate generation
- [ ] Similarity features
- [ ] Logistic Regression baseline
- [ ] Initial F₀.₅ evaluation

### Phase 3 — Optimization

- [ ] Improved blocking
- [ ] CatBoost/XGBoost
- [ ] Hard-negative analysis
- [ ] Threshold optimization
- [ ] Error analysis

### Phase 4 — Advanced

- [ ] Semantic similarity
- [ ] Additional feature engineering
- [ ] Model comparison
- [ ] Final pipeline optimization

### Phase 5 — Submission

- [ ] Generate final predictions
- [ ] Validate submission
- [ ] Complete methodology document
- [ ] Package final submission
