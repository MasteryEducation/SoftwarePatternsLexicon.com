---
linkTitle: "Accuracy"
title: "Accuracy: Proportion of Correctly Classified Instances"
description: "A detailed guide to understanding and applying the Accuracy metric in Machine Learning, including examples, related patterns, and additional resources."
categories:
- Model Validation and Evaluation Patterns
tags:
- accuracy
- evaluation
- classification
- metrics
- performance
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/evaluation-metrics/accuracy"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

**Accuracy**, a key evaluation metric in machine learning, measures the proportion of correctly classified instances among the total instances. It's widely used for evaluating classification algorithms, where the goal is to maximize the number of correct predictions.

### Mathematical Definition

Mathematically, accuracy is defined as:

{{< katex >}} \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} {{< /katex >}}

Where:
- \\( TP \\) (True Positive): Correctly predicted positive instances.
- \\( TN \\) (True Negative): Correctly predicted negative instances.
- \\( FP \\) (False Positive): Incorrectly predicted positive instances.
- \\( FN \\) (False Negative): Incorrectly predicted negative instances.

## Example Calculation

Consider a binary classification problem with the following confusion matrix:

|              | Predicted Positive | Predicted Negative |
|--------------|--------------------|--------------------|
| Actual Positive | 50 (TP)         | 10 (FN)           |
| Actual Negative | 5 (FP)          | 35 (TN)           |

By substituting these values in the accuracy formula:

{{< katex >}} \text{Accuracy} = \frac{50 + 35}{50 + 35 + 5 + 10} = \frac{85}{100} = 0.85 {{< /katex >}}

The accuracy is 85%, meaning 85% of the instances are correctly classified.

## Implementation Examples

### Python with scikit-learn

Here's a Python example using the `scikit-learn` library:

```python
from sklearn.metrics import accuracy_score

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

### R with caret

For R users, the `caret` package provides similar functionality:

```r
library(caret)

y_true <- factor(c(1, 0, 1, 1, 0, 1, 0, 0, 1, 0))
y_pred <- factor(c(1, 0, 1, 0, 0, 1, 0, 1, 1, 0))

confusionMatrix(y_pred, y_true)$overall['Accuracy']
```

## Related Design Patterns

### Precision

**Precision** is another important metric, especially in scenarios where the cost of false positives is high. It's calculated as:

{{< katex >}} \text{Precision} = \frac{TP}{TP + FP} {{< /katex >}}

Precision helps to understand the fraction of positive predictions that are actually correct.

### Recall

**Recall** (or Sensitivity) is crucial in contexts where the cost of false negatives is significant. It's defined as:

{{< katex >}} \text{Recall} = \frac{TP}{TP + FN} {{< /katex >}}

Recall measures the ability of the model to capture all relevant instances.

### F1 Score

The **F1 Score** combines precision and recall into a single metric, providing a balanced measure. It is the harmonic mean of precision and recall:

{{< katex >}} \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} {{< /katex >}}

## Additional Resources

- [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [R caret package documentation](https://topepo.github.io/caret/index.html)
- [Machine Learning Course by Andrew Ng](https://www.coursera.org/learn/machine-learning)

## Summary

Accuracy is a fundamental and straightforward evaluation metric in classification problems. While it provides a quick snapshot of model performance, it should be used in conjunction with other metrics like precision, recall, and F1 score, especially in imbalanced datasets. Through multiple examples and related design patterns, this article provides a comprehensive understanding of accuracy and its application.


