---
linkTitle: "Hamming Loss"
title: "Hamming Loss: Fraction of Labels that are Incorrectly Predicted"
description: "The Hamming Loss is an advanced evaluation metric used to measure the fraction of labels that are incorrectly predicted by a multi-label classifier."
categories:
- Model Validation and Evaluation Patterns
tags:
- machine learning
- evaluation
- metrics
- hamming loss
- multi-label classification
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/advanced-evaluation-metrics/hamming-loss"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

In multi-label classification problems, each instance can be assigned multiple labels simultaneously. One of the critical challenges in evaluating such models is to measure their accuracy effectively. The **Hamming Loss** is an advanced evaluation metric designed precisely for this purpose by accounting for the set of predictions that are incorrect.

## Definition and Formula

The Hamming Loss quantifies the fraction of labels that are incorrectly predicted. It is calculated as the average number of misclassified labels per instance and is formally defined as:

{{< katex >}}
\textrm{Hamming Loss} = \frac{1}{N \times L} \sum_{i=1}^{N} \sum_{j=1}^{L} \mathbf{1}(y_{ij} \neq \hat{y}_{ij})
{{< /katex >}}

where:
- \\( N \\) is the number of instances.
- \\( L \\) is the number of labels.
- \\( y_{ij} \\) is the true label of the \\( j^{th} \\) label for the \\( i^{th} \\) instance.
- \\( \hat{y}_{ij} \\) is the predicted label for the \\( j^{th} \\) label of the \\( i^{th} \\) instance.
- \\( \mathbf{1}( \cdot ) \\) is the indicator function that returns 1 when the argument is true and 0 when false.

## Example

Let's consider a simple example where we have 3 instances and 3 labels for a multi-label classification task.

### True Labels
| Instance | Label 1 | Label 2 | Label 3 |
|----------|---------|---------|---------|
| 1        | 1       | 0       | 1       |
| 2        | 0       | 1       | 0       |
| 3        | 1       | 1       | 1       |

### Predicted Labels
| Instance | Label 1 | Label 2 | Label 3 |
|----------|---------|---------|---------|
| 1        | 0       | 0       | 1       |
| 2        | 0       | 1       | 1       |
| 3        | 1       | 0       | 1       |

### Calculation
For instance 1: 2 labels are incorrect.
For instance 2: 1 label is incorrect.
For instance 3: 1 label is incorrect.

Hamming Loss can be calculated as follows:
{{< katex >}}
\textrm{Hamming Loss} = \frac{1}{3 \times 3} (2 + 1 + 1) = \frac{4}{9} = 0.444
{{< /katex >}}

Thus, our model's Hamming Loss is approximately 0.444.

## Implementation Examples

### Python (Scikit-Learn)

```python
from sklearn.metrics import hamming_loss

y_true = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
y_pred = [[0, 0, 1], [0, 1, 1], [1, 0, 1]]

h_loss = hamming_loss(y_true, y_pred)
print("Hamming Loss: {:.3f}".format(h_loss))
```

### R

```r
library(Metrics)

y_true <- matrix(c(1, 0, 1, 0, 1, 0, 1, 1, 1), nrow=3, byrow=TRUE)
y_pred <- matrix(c(0, 0, 1, 0, 1, 1, 1, 0, 1), nrow=3, byrow=TRUE)

h_loss <- hamming_loss(y_true, y_pred)
print(sprintf("Hamming Loss: %.3f", h_loss))
```

### Related Design Patterns

- **Precision and Recall**: These metrics provide insights into the correctness of the predictions specifically for binary and multi-class scenarios.
- **F1 Score**: Combines the precision and recall of a classifier into a single metric.
- **Subset Accuracy**: Measures the fraction of instances which have all their labels (in a multi-label problem) correctly predicted.
- **Jaccard Index**: Used for comparing the similarity and diversity of sample sets.

## Additional Resources

- [Multi-label Classification on Wikipedia](https://en.wikipedia.org/wiki/Multi-label_classification)
- [Scikit-Learn Multioutput and Multilabel module](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-classification)
- [Machine Learning Mastery on Multi-label Techniques](https://machinelearningmastery.com/multi-label-classification-with-deep-learning/)
  
## Summary

Hamming Loss serves a crucial role in evaluating multi-label classification models by measuring the proportion of labels that are incorrectly predicted. Its significance is particularly elevated in complex scenarios where multiple labels are assigned to instances, and the model's performance must be scrutinized meticulously.

With a sound understanding and proper implementation of Hamming Loss, data scientists and machine learning engineers can fine-tune their multi-label models more effectively to achieve higher accuracy and better predictive performance.
