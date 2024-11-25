---
linkTitle: "Precision"
title: "Precision: Proportion of Correctly Predicted Positive Observations"
description: "Detailed exploration of the Precision metric in machine learning, including descriptions, programming examples, related patterns, additional resources, and a summary."
categories:
- Model Validation and Evaluation Patterns
tags:
- Machine Learning
- Model Evaluation
- Precision
- Metrics
- Evaluation Metrics
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/evaluation-metrics/precision"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Precision is a fundamental evaluation metric in binary classification tasks that assess the accuracy of positive predictions. It is particularly important in scenarios where the cost of false positives is high. It is defined as the ratio of correctly predicted positive observations to the total predicted positives.

## Formula
Mathematically, precision (\\(P\\)) is expressed as:

{{< katex >}} P = \frac{TP}{TP + FP} {{< /katex >}}

Where:
- \\(TP\\) (True Positives): Correctly predicted positive cases.
- \\(FP\\) (False Positives): Incorrectly predicted positive cases.

## Importance of Precision
Precision focuses on the quality of true positive predictions. When false positives are particularly problematic (e.g., in medical diagnosis or fraud detection), high precision is vital.

## Detailed Example

### Python Implementation with scikit-learn

```python
from sklearn.metrics import precision_score

y_true = [0, 1, 1, 1, 0, 1, 0]

y_pred = [0, 1, 0, 1, 0, 1, 1]

precision = precision_score(y_true, y_pred)
print("Precision:", precision)
```

In this example, `precision_score` from scikit-learn calculates the precision of the predictions.

### R Implementation with caret

```R
library(caret)

y_true <- factor(c(0, 1, 1, 1, 0, 1, 0))
y_pred <- factor(c(0, 1, 0, 1, 0, 1, 1))

conf_matrix <- confusionMatrix(y_pred, y_true)

precision <- conf_matrix$byClass['Pos Pred Value']
print(paste("Precision:", precision))
```

In R, using the `caret` package, the `confusionMatrix` function helps calculate the precision.

### TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras import backend as K

y_true = tf.constant([0, 1, 1, 1, 0, 1, 0], dtype=tf.float32)
y_pred = tf.constant([0, 1, 0, 1, 0, 1, 1], dtype=tf.float32)

precision = tf.keras.metrics.Precision()
precision.update_state(y_true, y_pred)
print("Precision:", K.eval(precision.result()))
```

Using TensorFlow/Keras, the `Precision` class computes precision directly on TensorFlow tensors.

## Related Design Patterns

### **Recall**

Recall, also known as sensitivity, is the proportion of true positives identified out of all actual positives:

{{< katex >}} R = \frac{TP}{TP + FN} {{< /katex >}}

Where False Negatives (\\(FN\\)) are the positive cases missed by the model. There's often a trade-off between precision and recall that can be managed using the Precision-Recall curve.

### **F1 Score**

The F1 Score is the harmonic mean of precision and recall, providing a single metric that balances both:

{{< katex >}} F1 = 2 \cdot \frac{P \cdot R}{P + R} {{< /katex >}}

It's useful when you need a balance between Precision and Recall.

## Additional Resources

To deepen your understanding of precision and its importance in various applications, consider exploring these resources:

1. [Scikit-learn Documentation: Precision and Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
2. [Keras Documentation: Precision](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Precision)
3. [Understanding Precision, Recall, and F1 Score](https://towardsdatascience.com/precision-vs-recall-386cf9f89488)
4. [Evaluation Metrics for Machine Learning](https://developers.google.com/machine-learning/crash-course/classification/accuracy)

## Summary

Precision is a crucial evaluation metric for assessing the correctness of positive predictions in binary classification tasks. By focusing on reducing false positives, it becomes particularly important in applications such as medical diagnosis, information retrieval, and fraud detection. While using Precision, it is also helpful to consider other related metrics like Recall and F1 Score to get a comprehensive understanding of the model’s performance.

Make sure to iterate on your model assessments to fine-tune based on the trade-offs between precision, recall, and other relevant metrics specific to your use case. By leveraging these concepts appropriately, you can significantly improve the effectiveness of your classification models.
