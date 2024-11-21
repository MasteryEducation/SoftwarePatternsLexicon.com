---
linkTitle: "Matthews Correlation Coefficient (MCC)"
title: "Matthews Correlation Coefficient (MCC): Measure of the quality of binary classifications"
description: "An advanced evaluation metric used to measure the quality of binary classifications."
categories:
- Model Validation and Evaluation Patterns
tags:
- MCC
- Binary Classification
- Evaluation Metrics
- Model Validation
- Machine Learning
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/advanced-evaluation-metrics/matthews-correlation-coefficient-(mcc)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


The **Matthews Correlation Coefficient (MCC)** is a metric used in machine learning to measure the quality of binary classifications. It provides a balanced measure even when the classes are of very different sizes, making it particularly useful for datasets with class imbalance.

## What is MCC?

The MCC is a correlation coefficient between the observed and predicted binary classifications. It returns a value between -1 and +1:
- **+1** indicates a perfect prediction,
- **0** indicates no better than random prediction,
- **-1** indicates total disagreement between prediction and observation.

It’s considered one of the best metrics for assessing binary classification performance as it considers all elements of the confusion matrix.

The formula for MCC is given by:

{{< katex >}}
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
{{< /katex >}}

where:
- \\( TP \\): True Positives
- \\( TN \\): True Negatives
- \\( FP \\): False Positives
- \\( FN \\): False Negatives

## Example Calculation

Consider the following confusion matrix for a binary classifier:

|             | Predicted Positive | Predicted Negative |
|-------------|--------------------|--------------------|
| **Actual Positive** | 50                 | 10                 |
| **Actual Negative** | 5                  | 35                 |

Here:
- \\( TP = 50 \\)
- \\( TN = 35 \\)
- \\( FP = 5 \\)
- \\( FN = 10 \\)

Using the MCC formula:

{{< katex >}}
\text{MCC} = \frac{50 \cdot 35 - 5 \cdot 10}{\sqrt{(50+5)(50+10)(35+5)(35+10)}} = \frac{1750 - 50}{\sqrt{55 \cdot 60 \cdot 40 \cdot 45}} \approx \frac{1700}{5400.69} \approx 0.315
{{< /katex >}}

## MCC in Different Programming Languages and Frameworks

### Python (Using Scikit-Learn)
```python
from sklearn.metrics import matthews_corrcoef

y_true = [1, 1, 1, 0, 0, 0, 1, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

mcc = matthews_corrcoef(y_true, y_pred)
print(f'MCC: {mcc}')
```

### R
```r
install.packages("mltools")
library(mltools)
library(data.table)

y_true <- c(1, 1, 1, 0, 0, 0, 1, 0, 1, 1)
y_pred <- c(1, 0, 1, 0, 0, 1, 1, 0, 1, 0)

mcc <- mcc(y_true, y_pred)
print(paste('MCC:', mcc))
```

### JavaScript (Using TensorFlow.js)
```javascript
const tf = require('@tensorflow/tfjs');

const yTrue = tf.tensor1d([1, 1, 1, 0, 0, 0, 1, 0, 1, 1]);
const yPred = tf.tensor1d([1, 0, 1, 0, 0, 1, 1, 0, 1, 0]);

function matthewsCorrelationCoefficient(yTrue, yPred) {
  const tp = yTrue.mul(yPred).sum().dataSync();
  const tn = yTrue.neg().add(1).mul(yPred.neg().add(1)).sum().dataSync();
  const fp = yPred.sum().sub(tp).dataSync();
  const fn = yTrue.sum().sub(tp).dataSync();

  const numerator = (tp * tn) - (fp * fn);
  const denominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));

  return numerator / denominator;
}

const mcc = matthewsCorrelationCoefficient(yTrue, yPred);
console.log(`MCC: ${mcc}`);
```

## Related Design Patterns

### Confusion Matrix
A confusion matrix is a table used to evaluate the performance of a classification algorithm. It provides a more detailed breakdown of the performance compared to a single metric.

### Precision-Recall Curve
The precision-recall curve is useful for evaluating binary classifiers where specific class imbalances exist. It plots precision against recall for different threshold settings.

### ROC Curve and AUC
The Receiver Operating Characteristic (ROC) curve is another graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. In addition, the Area Under the Curve (AUC) provides a single metric to summarize the performance.

## Additional Resources

- [Scikit-Learn MCC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
- [Wikipedia: Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
- [Metrics to Evaluate your Machine Learning Algorithm](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234)

## Summary
The Matthews Correlation Coefficient (MCC) is a powerful evaluation metric for binary classification tasks, offering a comprehensive single-figure assessment that considers true and false positives and negatives. By encompassing all aspects of the confusion matrix, MCC provides a balanced evaluation even with imbalanced class distributions. Utilizing MCC alongside other metrics and visualizations, such as the confusion matrix and ROC curves, allows practitioners to gain deeper insights into model performance and make more informed decisions.

Using various tools and libraries available in different programming languages, MCC can be effortlessly integrated into the model evaluation pipeline, helping achieve more accurate and reliable machine learning models.


