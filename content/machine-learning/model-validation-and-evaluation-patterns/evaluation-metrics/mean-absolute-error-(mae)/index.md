---
linkTitle: "Mean Absolute Error (MAE)"
title: "Mean Absolute Error (MAE): Evaluation Metric for Regression Models"
description: "Mean Absolute Error (MAE) is a commonly used evaluation metric in regression problems that measures the average magnitude of errors in a set of predictions, without considering their direction."
categories:
- Model Validation and Evaluation Patterns
tags:
- regression
- evaluation
- metrics
- MAE
- model-validation
date: 2023-10-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/evaluation-metrics/mean-absolute-error-(mae)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Mean Absolute Error (MAE): Evaluation Metric for Regression Models

Mean Absolute Error (MAE) is a commonly used evaluation metric in regression problems. It measures the average magnitude of errors in a set of predictions, without considering their direction. The formula for MAE is given by:

{{< katex >}}
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
{{< /katex >}}

where \\\(y_i\\\) represents the actual values, \\\(\hat{y}_i\\\) represents the predicted values, and \\\(n\\\) is the number of observations.

MAE is very intuitive to understand as it represents the average absolute deviation of predictions from actual values, making it easily interpretable.

### Formula Breakdown

- **Absolute error**: The absolute value of the difference between the predicted value and the actual value for each observation: \\\(|y_i - \hat{y}_i|\\\).
- **Average**: The sum of these absolute errors divided by the number of observations, \\\(n\\\).

### Advantages and Disadvantages of MAE

#### Advantages

1. **Interpretability**: MAE is easy to understand and interpret as it provides an average of the absolute errors.
2. **Robustness to Outliers**: Unlike Mean Squared Error (MSE), MAE does not disproportionately penalize larger errors.

#### Disadvantages

1. **Non-differentiable**: MAE is not differentiable at zero, making it less suitable for some gradient-based optimization techniques.
2. **Equidistant Impact**: All errors are treated equally. It does not consider the magnitude of errors as seriously as MSE, which can be a drawback in some applications.

### Examples

#### Example in Python using Scikit-Learn

```python
import numpy as np
from sklearn.metrics import mean_absolute_error

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

mae = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error:", mae)
```

#### Example in R

```r
y_true <- c(3.0, -0.5, 2.0, 7.0)
y_pred <- c(2.5, 0.0, 2.0, 8.0)

mae <- mean(abs(y_true - y_pred))
print(paste("Mean Absolute Error:", mae))
```

### Related Design Patterns

1. **Mean Squared Error (MSE)**: Measures the average of the squares of the errors. Unlike MAE, MSE gives a higher penalty for larger errors.
2. **Root Mean Squared Error (RMSE)**: The square root of the mean squared error. It’s more interpretable in terms of the original units of output variable.
3. **R-squared (R²)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
4. **Cross-Validation**: Technique for assessing the performance of a predictive model by partitioning the data and evaluating the model on each partition in turn.

### Additional Resources

1. **Scikit-learn Documentation on MAE**: [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
2. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**: Chapter covering model evaluation techniques, including MAE.
3. **The Elements of Statistical Learning**: A comprehensive text with broader context on model assessment and selection.

### Summary

Mean Absolute Error (MAE) is a straightforward and intuitive metric for evaluating regression models, representing the average of the absolute errors. It is advantageous for its simplicity and robustness to outliers but has limitations in terms of differentiability and sensitivity to error magnitude. MAE holds a critical place within the broader ecosystem of evaluation metrics, complementing others like MSE and RMSE to give a complete picture of model performance.

By incorporating MAE in your model evaluation pipeline, you can gain a deeper insight into the average performance of your regression models relative to true values, helping you to fine-tune and improve them effectively.
