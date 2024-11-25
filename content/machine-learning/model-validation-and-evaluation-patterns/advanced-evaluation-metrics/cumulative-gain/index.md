---
linkTitle: "Cumulative Gain"
title: "Cumulative Gain: Measuring Classifier Gain Over Random Prediction"
description: "An advanced evaluation metric that measures the cumulative gain of using a classifier over random prediction."
categories:
- Model Validation and Evaluation Patterns
tags:
- evaluation
- metrics
- classifier
- validation
- overview
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/advanced-evaluation-metrics/cumulative-gain"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


The Cumulative Gain design pattern is an advanced evaluation metric used to measure the effectiveness of a classification model. It quantifies the advantage of using the classifier in comparison to random prediction.

## What is Cumulative Gain?

Cumulative Gain, often visualized via a Gain Chart, provides insights into how well a classifier separates the positive instances from the negative ones. The curve compares the performance of the classifier against a hypothetically perfect model and a random model to showcase the cumulative benefits of using the classifier.

### The Core Idea

Mathematically, Cumulative Gain (CG) can be described as:

{{< katex >}}
CG(n) = \sum_{i=1}^{n}y_i
{{< /katex >}}

where:
- \\(CG(n)\\) is the cumulative gain at the \\(n\\)-th instance.
- \\(y_i\\) is the true binary outcome for the \\(i\\)-th instance, which is 1 for a positive class and 0 for a negative class.

The Cumulative Gain is typically plotted with the cumulative number of instances sampled (on the x-axis) against the total number of positive instances found (on the y-axis).

## How Cumulative Gain Works

1. **Ordering Predictions**: Sort the predicted probabilities (or scores) from highest to lowest.
2. **Calculating True Positives**: For each instance at the \\(n\\)-th position in the sorted list, calculate whether it is a true positive.
3. **Cumulative Sum**: Sum the number of true positives up to the \\(n\\)-th instance.
4. **Random Model Comparison**: The curve for a random model, which selects instances without bias, serves as a baseline.
5. **No-gain Model Comparison**: The no-gain (or luck) model line is usually represented diagonally to indicate no improvement over random guessing.

### Example in Python

Here’s an example of how to generate a cumulative gain chart using Python and Scikit-learn:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_probs = np.array([0.9, 0.1, 0.8, 0.4, 0.05, 0.55, 0.2, 0.6, 0.75, 0.65])

sorted_indices = np.argsort(y_probs)[::-1]
y_true_sorted = y_true[sorted_indices]

cum_gain = np.cumsum(y_true_sorted)

n_positives = np.sum(y_true)
random_model = np.linspace(0, n_positives, len(y_true))

plt.figure(figsize=(8, 6))
plt.plot(cum_gain, label='Cumulative Gain (Model)')
plt.plot(random_model, label='Cumulative Gain (Random)', linestyle='--')
plt.xlabel('Number of Instances')
plt.ylabel('Cumulative Positive Instances')
plt.title('Cumulative Gain Chart')
plt.legend()
plt.show()
```

## Related Design Patterns

### Precision-Recall Curve
The Precision-Recall Curve is closely related to Cumulative Gain as it also evaluates classifier performance, especially for imbalanced datasets. It plots Precision (\\(\frac{TP}{TP + FP}\\)) against Recall (\\(\frac{TP}{TP + FN}\\)).

### ROC and AUC
The ROC Curve (Receiver Operating Characteristic) and its area under the curve (AUC) measure the trade-off between True Positive Rate (Recall) and False Positive Rate (FPR). The cumulative gain chart can be considered a transformation of the ROC curve focused on cumulative benefits.

## Additional Resources

- **Scikit-learn Documentation**: [Generating Evaluation Metrics in Scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
- **Kaggle**: Explore datasets with powerful visual learning tools.
- **Books**: "Pattern Recognition and Machine Learning" by Christopher M. Bishop for deeper insights into classification metrics.

## Summary

Cumulative Gain charts provide a visual and quantitative method to evaluate the performance of a classifier relative to random guessing. By following the step-by-step process to generate and interpret the gain chart, one can easily compared the classifier's efficacy in identifying relevant positive cases against an ideal or random classifier.

Understanding and applying the Cumulative Gain pattern facilitates better model verification and validation processes, allowing data scientists and ML practitioners to make informed decisions about their models' utility and performance.

