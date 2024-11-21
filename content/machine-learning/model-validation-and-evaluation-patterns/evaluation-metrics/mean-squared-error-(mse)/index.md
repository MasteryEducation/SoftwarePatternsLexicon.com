---
linkTitle: "Mean Squared Error (MSE)"
title: "Mean Squared Error (MSE): Average of Squared Errors Between Predicted and Actual Values"
description: "Detailed explanation of the Mean Squared Error (MSE), its applications, examples, related design patterns, and additional resources"
categories:
- Model Validation and Evaluation Patterns
tags:
- evaluation metrics
- loss functions
- regression
- model evaluation
- squared error
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/evaluation-metrics/mean-squared-error-(mse)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Mean Squared Error (MSE) is a widely used evaluation metric in machine learning, particularly in regression analysis. It measures the average of the squared differences between the predicted values and the actual values. The primary goal of using MSE is to quantify the quality of a model by calculating how well it predicts the actual outcome.

## Mathematical Definition

The Mean Squared Error is defined mathematically as:

{{< katex >}}
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
{{< /katex >}}

where:
- \\( n \\) is the number of observations
- \\( y_i \\) is the actual value
- \\( \hat{y}_i \\) is the predicted value

## Key Characteristics
- **Sensitivity to Outliers**: Since error terms are squared, larger errors are more penalized than smaller ones, making MSE sensitive to outliers.
- **Differentiability**: The function is differentiable, which makes it suitable for optimization algorithms like gradient descent.

## Programming Examples

Let’s explore some examples of calculating Mean Squared Error in different programming languages and libraries.

### Python (with NumPy)

```python
import numpy as np

y_true = np.array([2.5, 0.0, 2.1, 1.3])

y_pred = np.array([3.0, -0.5, 2.0, 1.5])

mse = np.mean((y_true - y_pred)**2)
print(f"Mean Squared Error: {mse}")
```

### R

```r
y_true <- c(2.5, 0.0, 2.1, 1.3)

y_pred <- c(3.0, -0.5, 2.0, 1.5)

mse <- mean((y_true - y_pred)^2)
print(paste("Mean Squared Error:", mse))
```

### TensorFlow (Keras)

```python
import tensorflow as tf

y_true = tf.constant([2.5, 0.0, 2.1, 1.3], dtype=tf.float32)
y_pred = tf.constant([3.0, -0.5, 2.0, 1.5], dtype=tf.float32)

mse = tf.keras.losses.MeanSquaredError()
result = mse(y_true, y_pred).numpy()
print(f"Mean Squared Error: {result}")
```

### Example in JavaScript

```javascript
const y_true = [2.5, 0.0, 2.1, 1.3];
const y_pred = [3.0, -0.5, 2.0, 1.5];

const mse = y_true.reduce((sum, actual, i) => {
  const error = actual - y_pred[i];
  return sum + error * error;
}, 0) / y_true.length;

console.log(`Mean Squared Error: ${mse}`);
```

## Related Design Patterns

### Mean Absolute Error (MAE)

**Description**: MAE measures the average of the absolute differences between the predicted values and the actual values. Unlike MSE, it is less sensitive to outliers since it does not square the errors.

### Root Mean Squared Error (RMSE)

**Description**: RMSE is the square root of MSE. It brings the error back to the original unit of measurement, making it more interpretable in terms of the original values.

### R-squared (Coefficient of Determination)

**Description**: R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It provides an indication of goodness-of-fit.

## Additional Resources

1. [Wikipedia: Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error)
2. [Scikit-Learn Documentation: Evaluating MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
3. [Deep Learning Book by Ian Goodfellow: Loss Functions](https://www.deeplearningbook.org/)
4. [StatQuest with Josh Starmer YouTube Channel: MSE Explained](https://www.youtube.com/user/joshstarmer)

## Summary

Mean Squared Error (MSE) is an essential evaluation metric that quantifies the difference between the predicted and actual values. Its sensitivity to large errors makes it particularly useful for assessing the performance of regression models. By understanding and implementing MSE, you can better evaluate the accuracy of your models and make informed decisions to improve them.

Understanding MSE in conjunction with other evaluation metrics like MAE and RMSE provides a more comprehensive analysis of a model's performance. For a well-rounded evaluation strategy, consider the problem context and the implications of various error metrics.
