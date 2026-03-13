# Cross-Domain Fake News Detection using TF-IDF, BERT and SBERT

## Overview

Fake news detection models often perform well when trained and tested on the same dataset but struggle when applied to different datasets. This project investigates the challenge of **cross-domain fake news detection**.

Traditional text representations such as **TF-IDF** are compared with **semantic embeddings from BERT and Sentence-BERT (SBERT)** to evaluate how well they generalize across datasets.

Two datasets were used in this study:

* **ISOT Fake News Dataset** – contains full news articles labeled as real or fake
* **LIAR Dataset** – contains short political claims labeled by truthfulness

The experiments analyze performance within datasets and across domains.

---

## Objective

The main goals of this project are:

1. Evaluate fake news detection performance within a dataset.
2. Analyze the performance drop when models are tested on a different dataset.
3. Compare TF-IDF with transformer-based embeddings.
4. Study whether semantic embeddings improve cross-domain generalization.

---

## Datasets

### 1. ISOT Fake News Dataset

Contains long-form news articles labeled as **real** or **fake**.

### 2. LIAR Dataset

Contains short political claims with truthfulness labels.
For this project, labels were simplified to binary classes:

* **0 → Fake**
* **1 → Real**

---

## Methodology

The project follows a typical NLP pipeline:

Text → Preprocessing → Feature Representation → Classification

### Text Preprocessing

Two approaches were used:

**Regex Cleaning**

* Convert text to lowercase
* Remove URLs
* Remove special characters
* Remove extra spaces

**spaCy Processing**

* Tokenization
* Stopword removal
* Lemmatization
* Punctuation removal

---

## Feature Representations

Three types of text representations were tested:

### TF-IDF

Captures word importance using frequency statistics.

### BERT Embeddings

Contextual embeddings generated using pretrained transformer models.

### Sentence-BERT (SBERT)

Produces dense semantic sentence embeddings suitable for similarity and classification tasks.

Model used:

```
all-MiniLM-L6-v2
```

---

## Classification Models

The following machine learning classifiers were used:

* **Logistic Regression**
* **Support Vector Machine (SVM)**

For some experiments:

```
class_weight = 'balanced'
```

was used to reduce class bias.

---

## Experiments

The following experiments were conducted:

1. TF-IDF models trained and tested on ISOT dataset
2. TF-IDF models trained and tested on LIAR dataset
3. Cross-domain TF-IDF (Train: ISOT → Test: LIAR)
4. BERT embeddings with Logistic Regression on ISOT
5. SBERT embeddings with Logistic Regression on LIAR
6. Cross-domain SBERT (Train: ISOT → Test: LIAR)
7. Cross-domain SBERT with class balancing

---

## Results

| Experiment                    | Model               | Accuracy |
| ----------------------------- | ------------------- | -------- |
| ISOT (TF-IDF)                 | Logistic Regression | 0.99     |
| ISOT (TF-IDF)                 | SVM                 | 1.00     |
| LIAR (TF-IDF)                 | Logistic Regression | 0.63     |
| LIAR (TF-IDF)                 | SVM                 | 0.64     |
| Cross-Domain TF-IDF           | Logistic Regression | 0.51     |
| Cross-Domain TF-IDF           | SVM                 | 0.48     |
| ISOT (BERT)                   | Logistic Regression | 0.94     |
| LIAR (SBERT)                  | Logistic Regression | 0.65     |
| Cross-Domain SBERT            | Logistic Regression | 0.55     |
| Cross-Domain SBERT + Balanced | Logistic Regression | 0.55     |

---

## Key Findings

* Models achieve extremely high accuracy when trained and tested on the same dataset.
* Performance drops significantly when models are tested on a different dataset.
* This drop occurs due to **domain shift** between datasets.
* Semantic embeddings from SBERT slightly improve cross-domain performance compared to TF-IDF.
* Class balancing does not significantly solve the domain shift problem.

These findings highlight the difficulty of building fake news detection systems that generalize across domains.

---

## Project Structure

```
Fake-News-Cross-Domain-Detection

datasets/
    isot/
    liar/

notebooks/
    01_isot_tfidf_logistic_svm.ipynb
    02_liar_tfidf_logistic_svm.ipynb
    03_cross_domain_tfidf.ipynb
    04_isot_bert_logistic.ipynb
    05_liar_sbert_logistic.ipynb
    06_cross_domain_sbert.ipynb
    07_cross_domain_sbert_class_balanced.ipynb

README.md
requirements.txt
```

---

## Installation

Clone the repository:

```
git clone [git clone https://github.com/yourusername/fake-news-cross-domain.git](https://github.com/TanujaMore27/Cross-domain-fake-news-detection-project.git)
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Requirements

```
pandas
numpy
scikit-learn
spacy
sentence-transformers
transformers
matplotlib
seaborn
jupyter
```

---

## Future Work

Possible improvements for this project include:

* Fine-tuning transformer models on combined datasets
* Domain adaptation techniques
* Data augmentation for cross-domain learning
* Testing larger transformer models

---

## Conclusion

This project demonstrates that fake news detection models trained on one dataset struggle to generalize to another dataset due to domain differences.

While semantic embeddings such as SBERT improve performance slightly, cross-domain fake news detection remains a challenging problem that requires more advanced domain adaptation techniques.
