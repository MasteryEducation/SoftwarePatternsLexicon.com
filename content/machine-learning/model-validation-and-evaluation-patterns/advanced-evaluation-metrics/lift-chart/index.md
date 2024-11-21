---
linkTitle: "Lift Chart"
title: "Lift Chart: Chart That Shows the Improvement a Model Offers Over Random Guessing"
description: "A detailed description of Lift Chart, a crucial pattern in model validation and evaluation, demonstrating the model's performance enhancement over random guessing, with examples, related patterns, and additional resources."
categories:
- Model Validation and Evaluation Patterns
tags:
- evaluation metrics
- model validation
- performance assessment
- data visualization
- advanced metrics
date: 2023-10-05
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/advanced-evaluation-metrics/lift-chart"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Lift Chart: Chart That Shows the Improvement a Model Offers Over Random Guessing

The **Lift Chart** is an essential tool in the domain of model validation and evaluation, especially when dealing with classification models. It visually represents how much better a predictive model is at identifying positive instances compared to a random guess. The chart is particularly useful in marketing, fraud detection, and other domains where distinguishing between positive and negative cases holds significant value.

### Introduction to Lift Chart

A Lift Chart evaluates a model by comparing its performance to a baseline model that makes predictions randomly. It's a plot where the x-axis represents the percentage of the sample, and the y-axis indicates the lift, which is the ratio of the results obtained with the model to the results obtained with random selection.

Mathematically, the lift for each decile (or percentile) can be written as:

{{< katex >}}\text{Lift (for decile k)} = \frac{ \text{target rate in decile k}}{ \text{overall target rate in the population}}{{< /katex >}}

### How to Construct a Lift Chart

1. **Sort the Data**: Sort the dataset based on the predicted probabilities (or scores) in descending order.
2. **Divide into Deciles**: Divide the sorted dataset into 10 equal parts (deciles).
3. **Compute the Target Rate**: For each decile, compute the actual rate of positive cases.
4. **Calculate Lift**: Calculate lift by dividing the target rate in each decile by the overall target rate in the dataset.
5. **Plot the Lift**: Plot the percentage of data on the x-axis and the lift on the y-axis.

### Example

#### Python Example using Scikit-learn and Matplotlib

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

np.random.seed(42)
N = 1000
X = np.random.rand(N, 5)
y = (np.random.rand(N) > 0.8).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]

results_df = pd.DataFrame({'true_label': y_test, 'predicted_prob': y_pred_proba})
results_df = results_df.sort_values(by='predicted_prob', ascending=False)

results_df['decile'] = pd.qcut(results_df['predicted_prob'], 10, labels=False)
lift = results_df.groupby('decile')['true_label'].mean() / results_df['true_label'].mean()

lift.plot(kind='bar', figsize=(10, 6))
plt.ylabel('Lift')
plt.xlabel('Decile')
plt.title('Lift Chart')
plt.show()
```

### Related Design Patterns

1. **Confusion Matrix**: Evaluates a classification model's performance by summarizing true positives, false positives, true negatives, and false negatives. This pattern helps understand different types of prediction errors.

2. **ROC Curve (Receiver Operating Characteristic Curve)**: Plots the true positive rate against the false positive rate across different threshold values. It helps evaluate the trade-off between sensitivity and specificity.

3. **Precision-Recall Curve**: Plots precision against recall for different threshold settings. It is particularly useful for imbalanced datasets where the positive class is rare.

4. **Gain Chart**: Similar to the Lift Chart but focuses on the cumulative gain, showing how much of the positive class is captured by different percentages of the data.

### Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html): Comprehensive guide on evaluation metrics and model validation techniques.
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html): Detailed instructions on creating various plots and visualizations.
- Research Paper: "Lift Charts: A method for visualizing function learning" by Gary M. Weiss – A thorough academic discussion on lift charts.

### Summary

The **Lift Chart** is a powerful visualization technique used to assess the performance of classification models by showing how much better the model is at predicting positive cases compared to random guessing. By separating the dataset into deciles based on predicted probabilities, the lift chart highlights the model's effectiveness in identifying true positives. It is often employed alongside other metrics such as ROC curves and confusion matrices to provide comprehensive model evaluation.

With practical examples in Python and a detailed build-up, this article illustrates the construction and application of Lift Charts effectively, ensuring that developers and data scientists can leverage this tool for better model validation and performance assessment.
