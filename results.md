# Experimental Results

This project evaluates fake news detection performance using different feature representations and machine learning models across two datasets: **ISOT** and **LIAR**.

---

# Within-Dataset Experiments

## ISOT Dataset (TF-IDF)

### Logistic Regression

Accuracy: **0.99**

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Fake  | 0.99      | 0.99   | 0.99     |
| Real  | 0.99      | 0.99   | 0.99     |

---

### Support Vector Machine

Accuracy: **1.00**

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Fake  | 1.00      | 1.00   | 1.00     |
| Real  | 1.00      | 1.00   | 1.00     |

---

## LIAR Dataset (TF-IDF)

### Logistic Regression

Accuracy: **0.63**

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Fake  | 0.59      | 0.45   | 0.51     |
| Real  | 0.65      | 0.76   | 0.70     |

---

### Support Vector Machine

Accuracy: **0.64**

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Fake  | 0.62      | 0.42   | 0.50     |
| Real  | 0.64      | 0.80   | 0.72     |

---

# Cross-Domain Experiments

Train: **ISOT**
Test: **LIAR**

## TF-IDF Models

### Logistic Regression

Accuracy: **0.51**

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Fake  | 0.44      | 0.50   | 0.47     |
| Real  | 0.57      | 0.51   | 0.54     |

---

### Support Vector Machine

Accuracy: **0.48**

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Fake  | 0.41      | 0.51   | 0.46     |
| Real  | 0.55      | 0.46   | 0.50     |

---

# Transformer-Based Experiments

## BERT Embeddings (ISOT)

Classifier: Logistic Regression

Accuracy: **0.945**

Confusion Matrix:

```
[[947 53]
 [55 925]]
```

---

## SBERT Embeddings (LIAR)

Classifier: Logistic Regression

Accuracy: **0.65**

Confusion Matrix:

```
[[164 177]
 [96 353]]
```

---

# Cross-Domain SBERT

Train: **ISOT**
Test: **LIAR**

Accuracy: **0.548**

Confusion Matrix:

```
[[90 251]
 [106 343]]
```

---

# Cross-Domain SBERT (Class Balanced)

Classifier: Logistic Regression
Parameter:

```
class_weight='balanced'
```

Accuracy: **0.554**

Confusion Matrix:

```
[[88 253]
 [99 350]]
```

---

# Summary

| Experiment          | Accuracy  |
| ------------------- | --------- |
| TF-IDF (ISOT)       | 0.99      |
| TF-IDF (LIAR)       | 0.63      |
| Cross-Domain TF-IDF | 0.48–0.51 |
| BERT (ISOT)         | 0.94      |
| SBERT (LIAR)        | 0.65      |
| Cross-Domain SBERT  | 0.55      |

---

# Key Insight

Models trained and tested on the same dataset achieve high accuracy.
However, when evaluated across datasets, performance drops significantly due to **domain shift**.

Semantic embeddings (SBERT) improve cross-domain performance slightly compared to TF-IDF but do not completely solve the problem.

