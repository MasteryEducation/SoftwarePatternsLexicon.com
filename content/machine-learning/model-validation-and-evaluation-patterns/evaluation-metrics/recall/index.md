---
linkTitle: "Recall"
title: "Recall: Proportion of Actual Positives that Were Identified Correctly"
description: "Recall measures the proportion of actual positives that were identified correctly. It is an essential metric in the context of imbalanced datasets or where the cost of false negatives is high."
categories:
- Model Validation and Evaluation Patterns
tags:
- Evaluation Metrics
- Model validation
- Precision
- Imbalanced data
- Classification tasks
date: 2023-10-20
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/evaluation-metrics/recall"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Recall is a crucial evaluation metric in machine learning, which quantifies the ability of a model to find all the relevant cases (true positives) within a dataset. It is especially important in situations where missing actual positive cases (false negatives) can be costly or detrimental.

## Definition

Recall, also known as Sensitivity, is defined as the number of true positive predictions divided by the sum of true positive and false negative predictions. Mathematically, it's expressed as:

{{< katex >}}
\text{Recall} = \frac{TP}{TP + FN}
{{< /katex >}}

Where:
- \\( TP \\) (True Positives) is the number of actual positive cases correctly identified.
- \\( FN \\) (False Negatives) is the number of actual positive cases incorrectly identified as negative.

## Importance

Recall becomes extremely important in scenarios such as:
- **Medical Diagnosis**: Missing a disease (false negative) can have severe consequences.
- **Fraud Detection**: Missing a fraud case in transactions can lead to financial losses.
- **Spam Detection**: Missing a spam email can potentially miss phishing attempts.

In such cases, recall ensures that the model catches as many actual positive cases as possible.

## Example Calculation

Consider a binary classification problem with the following confusion matrix:

|                | Predicted Positive | Predicted Negative |
|----------------|---------------------|---------------------|
| **Actual Positive** | 60                   | 10                   |
| **Actual Negative** | 5                    | 25                   |

From the confusion matrix:
- \\( TP = 60 \\)
- \\( FN = 10 \\)

Using the recall formula:

{{< katex >}}
\text{Recall} = \frac{TP}{TP + FN} = \frac{60}{60 + 10} = \frac{60}{70} \approx 0.857
{{< /katex >}}

This means the model has a recall of approximately 85.7%, indicating it successfully identified 85.7% of the actual positive cases.

## Implementation Examples

### Python (Scikit-learn)

```python
from sklearn.metrics import recall_score

y_true = [1, 1, 1, 1, 0, 0, 0, 0]
y_pred = [1, 1, 0, 1, 0, 1, 0, 0]

recall = recall_score(y_true, y_pred)
print(f'Recall: {recall:.2f}')
```

### R (Caret Package)

```r
install.packages("caret")
library(caret)

y_true <- factor(c(1, 1, 1, 1, 0, 0, 0, 0))
y_pred <- factor(c(1, 1, 0, 1, 0, 1, 0, 0))

recall <- posPredValue(y_pred, y_true, positive = "1")
print(paste("Recall:", round(recall, 2)))
```

## Related Design Patterns

1. **Precision**: Measures the proportion of positive predictions that are actually correct.
   - Formal Definition: \\(\text{Precision} = \frac{TP}{TP + FP}\\)
   - In scenarios where the cost of false positives is high, precision becomes critical.

2. **F1 Score**: The harmonic mean of precision and recall, offering a balance between the two metrics.
   - Formal Definition: \\(\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}\\)
   - Useful when both false positives and false negatives are significant.

3. **ROC-AUC**: Provides an aggregate measure of performance across different classification thresholds.
   - It plots the true positive rate (recall) against the false positive rate (1-precision) at various threshold settings.

## Additional Resources

- [Scikit-learn Documentation on Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
- [Understanding Evaluation Metrics (Blog)](https://towardsdatascience.com/understanding-evaluation-metrics-a20dcbdc6882)
- [Machine Learning Mastery on Precision-Recall](https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/)

## Summary

Recall is a critical evaluation metric for understanding a model's performance, particularly in identifying all relevant positive cases. By focusing on recall, practitioners ensure that the model effectively minimizes false negatives. Combining recall with other metrics such as precision and the F1 score gives a more comprehensive view of a model's performance, crucial for robust machine learning systems.

By understanding and implementing recall, you can ensure your model aligns well with the real-world cost and relevance of various types of errors, thus enhancing its effectiveness and reliability.
