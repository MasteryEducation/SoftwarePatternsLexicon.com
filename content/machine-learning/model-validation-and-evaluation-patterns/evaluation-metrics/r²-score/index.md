---
linkTitle: "R² Score"
title: "R² Score: Proportion of Variance Explained by the Independent Variables"
description: "The R² score, also known as the coefficient of determination, measures the proportion of variance in the dependent variable that can be explained by the independent variables in a regression model."
categories:
- Model Validation and Evaluation Patterns
tags:
- Evaluation Metrics
- Model Evaluation
- Regression Analysis
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/evaluation-metrics/r²-score"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## R² Score: Proportion of Variance Explained by the Independent Variables

The R² score, or the coefficient of determination, is a metric used to evaluate the performance of a regression model. It provides an indication of how well the independent variables explain the variability of the dependent variable. The R² score ranges from 0 to 1, with higher values indicating a better fit of the model.

### Mathematical Definition

Formally, the R² score can be defined as:

{{< katex >}}
R² = 1 - \frac{SS_{res}}{SS_{tot}}
{{< /katex >}}

where:
- \\( SS_{res} \\) is the residual sum of squares (the sum of the squares of the residuals, which are the differences between the observed and predicted values).
- \\( SS_{tot} \\) is the total sum of squares (the sum of the squared differences between the observed values and the mean of the observed values).

Another way to express this is:

{{< katex >}}
SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
{{< /katex >}}
{{< katex >}}
SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2
{{< /katex >}}

### Interpretation

- **R² = 1**: The model explains all the variability of the response data around its mean.
- **R² = 0**: The model explains none of the variability of the response data around its mean.
- **R²** between 0 and 1: Indicates the proportion of variability explained by the model, with higher values signifying a better model.

### Example Implementation

Let's look at an example in Python using scikit-learn and in R.

#### Python (scikit-learn)
```python
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = LinearRegression().fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f'R² score: {r2}')
```

#### R
```R
library(MASS)

X <- matrix(c(1, 1, 1, 2, 2, 2, 2, 3), ncol=2, byrow=TRUE)
y <- X %*% c(1, 2) + 3

df <- data.frame(X1 = X[,1], X2 = X[,2], y = y)

model <- lm(y ~ X1 + X2, data = df)

y_pred <- predict(model, df)

r2 <- summary(model)$r.squared
print(paste('R² score:', r2))
```

### Related Design Patterns

- **Adjusted R² Score**: A modified version of the R² score that adjusts for the number of predictors in the model. It prevents overestimation of the fitness for the models with many predictors.
- **Mean Squared Error (MSE)**: Measures the average of the squared differences between observed and predicted values.
- **Mean Absolute Error (MAE)**: Measures the average of the absolute differences between observed and predicted values.

### Additional Resources

1. [Understanding the R² Score - A Comprehensive Guide](https://example-resource-1.com)
2. [scikit-learn: Metrics and Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
3. [Introduction to Linear Regression - R Documentation](https://example-resource-2.com)

### Summary

The R² score is a crucial metric in regression analysis which helps determine how well the independent variables explain the variance of the dependent variable. Understanding and correctly using the R² score can provide deep insights into the efficacy and accuracy of predictive models. Its simplicity and interpretability make it widely used, but it's essential to consider it alongside other metrics to ensure a comprehensive evaluation of model performance.
