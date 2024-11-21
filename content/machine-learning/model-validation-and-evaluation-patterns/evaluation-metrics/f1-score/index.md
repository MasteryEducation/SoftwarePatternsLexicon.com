---
linkTitle: "F1 Score"
title: "F1 Score: Harmonic mean of precision and recall"
description: "A comprehensive look at the F1 Score, an essential evaluation metric that balances precision and recall for binary and multiclass classification problems."
categories:
- Model Validation and Evaluation Patterns
tags:
- machine learning
- evaluation metrics
- precision
- recall
- F1 Score
- classification
date: 2023-10-23
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/evaluation-metrics/f1-score"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

In machine learning, evaluating model performance is crucial. Among various evaluation metrics, the F1 Score stands out for its effectiveness in balancing precision and recall for tasks involving binary and multiclass classification. This article delves into the F1 Score, its importance, and how to compute it. Additionally, examples in Python and other languages are provided, followed by related design patterns and further resources.

## Theoretical Foundation

The F1 Score is the harmonic mean of precision and recall, offering a balanced measure of a model's performance, especially when dealing with imbalanced datasets. It can be defined as:

{{< katex >}}
F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
{{< /katex >}}

### Precision

Precision is the ratio of correctly predicted positive observations to the total predicted positives:

{{< katex >}}
\text{Precision} = \frac{TP}{TP + FP}
{{< /katex >}}

### Recall

Recall, or sensitivity, is the ratio of correctly predicted positive observations to all observations in the actual class:

{{< katex >}}
\text{Recall} = \frac{TP}{TP + FN}
{{< /katex >}}

### F1 Score

Given the above definitions, the F1 Score is computed as:

{{< katex >}}
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
{{< /katex >}}

where:
- \\(TP\\) (True Positives)
- \\(FP\\) (False Positives)
- \\(FN\\) (False Negatives)

## Practical Implementation

### Python Example

Here is an example of calculating the F1 Score in Python using `sklearn`:

```python
from sklearn.metrics import f1_score

y_true = [0, 1, 1, 1, 0, 1, 0, 0]
y_pred = [0, 1, 0, 1, 0, 1, 1, 0]

f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)
```

### R Example

In R, using the `caret` package:

```R
library(caret)

y_true <- factor(c(0, 1, 1, 1, 0, 1, 0, 0))
y_pred <- factor(c(0, 1, 0, 1, 0, 1, 1, 0))

conf_matrix <- confusionMatrix(y_pred, y_true)
precision <- conf_matrix$byClass['Pos Pred Value']
recall <- conf_matrix$byClass['Sensitivity']

f1 <- 2 * ((precision * recall) / (precision + recall))
print(f1)
```

### Other Languages

Similarly, implementations are straightforward in other languages like Java and MATLAB. Refer to the respective libraries for syntax and method calls.

## Related Design Patterns

- **Precision-Recall Curve**: This visual method provides deep insights into the trade-off between precision and recall, particularly useful for imbalanced datasets.
- **ROC-AUC**: While focusing on true positive and false positive rates, the ROC-AUC serves a slightly different but related purpose, often used together with F1 score for comprehensive evaluation.
- **Confusion Matrix**: A fundamental tool providing the essential groundwork to calculate precision, recall, and subsequently the F1 Score.

## Additional Resources

- [Scikit-learn Documentation - F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Machine Learning Mastery - F1 Score](https://machinelearningmastery.com/f1-score-machine-learning/)
- [Coursera - Data Science Specialization](https://www.coursera.org/specializations/jhu-data-science)

## Summary

The F1 Score is a powerful evaluation metric harmonizing precision and recall. It's particularly valuable in scenarios with imbalanced classes, ensuring that both false positives and false negatives are minimized proportionately. By understanding and implementing the F1 Score, machine learning practitioners can better evaluate and refine their models for robust performance.

---

In conclusion, the F1 Score stands as a cornerstone evaluation metric that balances the trade-off between precision and recall, providing a single metric to understand the model's effectiveness comprehensively.
