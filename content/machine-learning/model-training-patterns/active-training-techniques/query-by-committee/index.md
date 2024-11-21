---
linkTitle: "Query-by-Committee"
title: "Query-by-Committee: Consensus on Uncertain Samples for Labeling"
description: "A machine learning design pattern involving multiple models to identify and label the most uncertain samples through consensus."
categories:
- Model Training Patterns
- Active Training Techniques
tags:
- machine learning
- active learning
- ensemble learning
- model training
- uncertainty sampling
date: 2024-10-06
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-training-patterns/active-training-techniques/query-by-committee"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Query-by-Committee: Consensus on Uncertain Samples for Labeling

The **Query-by-Committee** (QBC) design pattern is an active training technique in machine learning that employs a committee of diverse models to identify the most uncertain samples in the data. These models work together to reach a consensus on which data points should be selected for manual labeling, thereby improving the efficiency of training datasets where labeled data is a scarce or expensive resource.

### Detailed Description

In many machine learning tasks, labeled data is often limited or costly to obtain. The Query-by-Committee pattern addresses this issue by using multiple models, termed as a "committee," to determine which samples are most in need of labeling. The key principle behind QBC is that the models in the committee will disagree the most about the data points that lie near the decision boundaries. 

Here's the high-level process:

1. **Initial Training:** Train a committee of diverse models on a small, initial labeled dataset.
2. **Query Selection:** Present the unlabeled data to the committee and have each model predict the labels.
3. **Disagreement Measurement:** Measure the uncertainty or disagreement among the models' predictions, often utilizing metrics such as entropy, variance, or majority vote.
4. **Sample Selection:** Select the samples with the highest levels of predicted disagreement for labeling.
5. **Labeling:** Acquire the true labels for these samples through human annotation or other means.
6. **Model Retraining:** Retrain the committee with the newly labeled data and repeat the process until satisfactory performance is achieved.

### Advantages

- **Efficiency:** Reduces the amount of labeled data required by focusing on the most informative examples.
- **Improved Performance:** Enhances model generalization by resolving uncertainty around the decision boundaries.

### Disadvantages

- **Computationally Intensive:** Requires maintaining and evaluating multiple models.
- **Complexity:** Involves sophisticated strategies for quantifying disagreement and managing multiple models.

### Mathematical Formulation

The essence of Query-by-Committee can be quantified by disagreement measures. Suppose we have a committee of \\( M \\) models \\( H_1, H_2, \ldots, H_M \\). Let each model \\( H_i \\) predict the probabilities for class \\( c \\) for a particular instance \\( x \\):

{{< katex >}} P_{c}^i(x) {{< /katex >}}

The average disagreement can be calculated using:

{{< katex >}} D(x) = \frac{1}{M} \sum_{i=1}^{M} \left[ \sum_{c=1}^{C} P_{c}^i(x) \log \left( P_{c}^i(x) \right) \right] {{< /katex >}}

### Examples

#### Example 1: Python with Scikit-Learn

In Python, using scikit-learn, we can implement a basic version of QBC:

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = make_multilabel_classification(n_samples=1000, n_classes=3, n_labels=1, random_state=42)
X_train, X_unlabeled, y_train, y_unlabeled = train_test_split(X, y, test_size=0.9, random_state=42)

committee_size = 3
committee = [RandomForestClassifier(n_estimators=10) for _ in range(committee_size)]
for model in committee:
    model.fit(X_train, y_train)

def query_by_committee(committee, X_unlabeled, n_instances):
    # Collect predictions from each model
    predictions = np.array([model.predict(X_unlabeled) for model in committee])
    disagreements = np.var(predictions, axis=0).sum(axis=1)  # Measure of disagreement

    # Select the most uncertain samples
    query_idx = np.argsort(disagreements)[-n_instances:]
    return query_idx

query_idx = query_by_committee(committee, X_unlabeled, n_instances=10)
X_query = X_unlabeled[query_idx]
y_query = y_unlabeled[query_idx]

X_train = np.concatenate((X_train, X_query), axis=0)
y_train = np.concatenate((y_train, y_query), axis=0)

for model in committee:
    model.fit(X_train, y_train)

accuracy = np.mean([accuracy_score(y_unlabeled, model.predict(X_unlabeled)) for model in committee])
print("Ensemble Accuracy after query-by-committee: ", accuracy)
```

### Related Design Patterns

1. **Bootstrap Aggregating (Bagging):**
   - Involved in training multiple models independently and combining their results, which can inspire the committee setup in QBC.
   - **Description:** Aggregates multiple bootstrapped replicated model results to improve generalization and robustness.
   
2. **Uncertainty Sampling:**
   - Directly related to the measure of uncertainty seen in QBC.
   - **Description:** Seeks out samples where the model is uncertain about the output, prioritizing them for labeling.

3. **Ensemble Learning:**
   - The general concept of combining multiple models.
   - **Description:** Utilizing several models to improve predictive performance.

### Additional Resources

- [Scikit-Learn User Guide on Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Active Learning Literature Survey](https://arxiv.org/abs/0906.1458)
- [Active Learning Book by Burr Settles](https://www.morganclaypool.com/doi/abs/10.2200/S00258ED1V01Y201207AIM018)

### Summary

The Query-by-Committee pattern leverages multiple models to focus labeling efforts on the most uncertain samples, thereby increasing the efficiency and effectiveness of training data usage in machine learning projects. By iteratively selecting and labeling the most contentious samples, QBC ensures that the models' decision boundaries are reinforced where necessary, leading to better overall model performance with fewer labeled data points. Applications of QBC are particularly useful in scenarios with limited labeled data availability, where optimizing training data utility is crucial.

