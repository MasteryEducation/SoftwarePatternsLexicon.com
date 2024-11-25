---
linkTitle: "Cumulative Accuracy Profile (CAP)"
title: "Cumulative Accuracy Profile: Evaluating Binary Classification Models"
description: "Understanding and implementing the Cumulative Accuracy Profile (CAP), a technique for evaluating binary classification models with practical examples."
categories:
- Model Validation and Evaluation Patterns
- Advanced Evaluation Techniques
tags:
- machine-learning
- model-validation
- binary-classification
- accuracy-evaluation
- CAP-technique
date: 2023-10-15
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/advanced-evaluation-techniques/cumulative-accuracy-profile-(cap)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

The **Cumulative Accuracy Profile (CAP)** is a visual and analytical technique used for evaluating the performance of binary classification models. It provides insight into a model's ability to discriminate between positive and negative classes, thus guiding model refinement and evaluation.

## Principles of CAP

The CAP curve is constructed by plotting the cumulative number of predicted positives (i.e., the number of true positives plus false positives) versus the ordered instances from the most to the least probable to be positive. 

### Steps to Construct CAP:

1. **Sort Instances:** Sort the test dataset by the predicted probability, from the highest to the lowest.
2. **Calculate Cumulative Values:** Calculate the cumulative number of actual positive instances.
3. **Plotting:**
   - The x-axis represents the cumulative number of instances evaluated.
   - The y-axis represents the cumulative number of actual positive instances identified.

### Example

Suppose we have the following predictions from a binary classifier:

| Instance | True Label | Predicted Probability |
|----------|------------|-----------------------|
| A        | 1          | 0.95                  |
| B        | 0          | 0.85                  |
| C        | 1          | 0.75                  |
| D        | 0          | 0.55                  |
| E        | 1          | 0.45                  |

Sort instances by the predicted probability:

| Instance | True Label | Predicted Probability | Cumulative Positives |
|----------|------------|-----------------------|----------------------|
| A        | 1          | 0.95                  | 1                    |
| B        | 0          | 0.85                  | 1                    |
| C        | 1          | 0.75                  | 2                    |
| D        | 0          | 0.55                  | 2                    |
| E        | 1          | 0.45                  | 3                    |

Using the table above, you can plot the CAP curve.

{{< katex >}}
\begin{array}{ccc}
\text{Cumulative Instances} & \text{Cumulative Positives} \\
0 & 0 \\ 
1 & 1 \\ 
2 & 1 \\ 
3 & 2 \\ 
4 & 2 \\ 
5 & 3 \\ 
\end{array}
{{< /katex >}}

## Interpreting the CAP Curve

- **Diagonal Line:** Random classifier performance.
- **Perfect Model Line:** Model that predicts all positives before any negatives.
- **Actual Model Line:** The performance of the actual model.

**Area Under CAP (AUC-CAP):** The area between the CAP curve and the diagonal line can indicate model quality; higher is better.

### Implementation Example

#### Python Example with Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_cap(y_true, y_probs):
    data = list(zip(y_true, y_probs))
    data.sort(key=lambda x: x[1], reverse=True)
    cum_pos = np.cumsum([x[0] for x in data])
    total_pos = np.sum(y_true)
    total_count = len(y_true)
    
    plt.plot([0, total_count], [0, total_pos], linestyle='--', label='Random Model')
    plt.plot(range(1, total_count+1), cum_pos, label='Actual Model')
    plt.xlabel('Number of Instances')
    plt.ylabel('Number of Positives Identified')
    plt.title('Cumulative Accuracy Profile')
    plt.legend()
    plt.show()

true_labels = [1, 0, 1, 0, 1]
pred_prob = [0.95, 0.85, 0.75, 0.55, 0.45]

plot_cap(true_labels, pred_prob)
```

## Related Design Patterns

### Receiver Operating Characteristic (ROC) Curve
The ROC curve is another technique to evaluate the performance of binary classifiers, plotting the true positive rate against the false positive rate. 

#### Usage:
- Used in situations where class distribution is imbalanced.
  
### Precision-Recall Curve
The precision-recall curve is particularly valuable for imbalanced datasets, plotting precision against recall for different thresholds.

## Additional Resources

- **Scikit-learn Documentation:** [ROC and AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
- **Matplotlib Documentation:** [Creating visualizations with Matplotlib](https://matplotlib.org/stable/index.html)

## Summary

The Cumulative Accuracy Profile (CAP) is an insightful method for the evaluation of binary classification models. By visualizing the cumulative distribution of positives predicted by the model, CAP aids in understanding model performance, particularly when handling imbalanced datasets. The use of CAP curves, along with other evaluation methods like ROC and precision-recall curves, can provide a comprehensive view of your model's effectiveness.

By implementing CAP in your analysis pipeline, you ensure robust evaluation, better performance metrics interpretation, and an actionable approach to model improvement.


