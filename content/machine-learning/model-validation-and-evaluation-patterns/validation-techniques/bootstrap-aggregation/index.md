---
linkTitle: "Bootstrap Aggregation"
title: "Bootstrap Aggregation: Resampling Data with Replacement to Create Multiple Datasets"
description: "A detailed look into the Bootstrap Aggregation (Bagging) pattern, its applications in model validation and evaluation, along with examples and related design patterns."
categories:
- Model Validation and Evaluation Patterns
tags:
- machine learning
- model validation
- resampling
- bagging
- overfitting
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/validation-techniques/bootstrap-aggregation"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Bootstrap Aggregation, commonly known as Bagging, is a powerful ensemble method that helps improve model accuracy and robustness. It involves generating multiple datasets by resampling the original dataset with replacement and training a model on each of these datasets. The predictions of these models are then aggregated (typically by averaging) to produce a final prediction. This technique is effective in reducing variance and minimizing overfitting, especially for high-variance methodologies like decision trees.

## Key Components
1. **Resampling with Replacement**: Generating multiple datasets by randomly selecting samples from the original dataset, allowing duplicate records.
2. **Multiple Models**: Training separate models on each resampled dataset.
3. **Aggregation**: Combining model predictions by averaging (for regression) or voting (for classification).

## Mathematical Foundation
Given a dataset \\( D = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\} \\), we derive \\( B \\) different datasets \\( D_1, D_2, \ldots, D_B \\) by bootstrap sampling:

- For each \\( D_b \\) (where \\( b \\) ranges from 1 to \\( B \\)), randomly sample \\( n \\) instances from \\( D \\) with replacement.
- Train a model \\( f_b \\) on each \\( D_b \\).

The final prediction \\(\hat{y}\\) is the aggregated prediction of the individual models:
- **Regression**: \\(\hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(x)\\)
- **Classification**: \\(\hat{y} = \text{mode}\{f_1(x), f_2(x), \ldots, f_B(x)\}\\)

## Example Implementations

### Python (using scikit-learn)
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

### R (using caret)
```r
library(caret)
data(iris)

set.seed(42)
trainIndex <- createDataPartition(iris$Species, p = .7, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

control <- trainControl(method = "boot", number = 10)
model <- train(Species~., data = trainData, method = "treebag", trControl = control)

predictions <- predict(model, newdata = testData)
confusionMatrix(predictions, testData$Species)
```

## Related Design Patterns
### Boosting
Boosting is another ensemble technique that builds models sequentially, where each model attempts to correct the errors of its predecessor. It typically improves model accuracy but is more prone to overfitting compared to bagging.

### Stacking
Stacking also involves training multiple models but uses a meta-learner to combine their predictions. Unlike bagging and boosting, stacking can combine different model types, potentially capturing various patterns in the dataset.

## Additional Resources
1. [Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123-140.](https://link.springer.com/article/10.1007/BF00058655)
2. [Ensemble Methods: Foundations and Algorithms by Zhi-Hua Zhou](https://www.wiley.com/en-us/Ensemble+Methods%3A+Foundations+and+Algorithms-p-978047052393,978-0-470-90982-3)
3. [Python Machine Learning by Sebastian Raschka](https://www.amazon.com/Python-Machine-Learning-Second-Cookbook/dp/1787125939)

## Summary
Bootstrap Aggregation (Bagging) is a vital technique in the toolbox of machine learning practitioners. It mitigates overfitting and variance in predictive models by leveraging resampling. Through multiple iterations and aggregation of results, it ensures more reliable and stable predictions. Bagging is particularly useful for high variance models like decision trees, providing improved performance and robustness in various scenarios.
{{< katex />}}

