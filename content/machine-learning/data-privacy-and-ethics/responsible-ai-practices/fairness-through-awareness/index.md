---
linkTitle: "Fairness Through Awareness"
title: "Fairness Through Awareness: Explicitly Modeling and Mitigating Biases"
description: "A detailed exploration of the Fairness Through Awareness design pattern, focusing on the explicit modeling and mitigation of biases in machine learning models."
categories:
- Data Privacy and Ethics
tags:
- Fairness
- Bias Mitigation
- Responsible AI
- Ethical AI
- Model Evaluation
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/data-privacy-and-ethics/responsible-ai-practices/fairness-through-awareness"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction
Machine learning models can inadvertently propagate and amplify biases present in the training data. The **Fairness Through Awareness** design pattern emphasizes the importance of explicitly modeling and mitigating these biases to ensure more ethical and responsible AI practices. This pattern falls under the broader category of Responsible AI Practices, specifically within the realm of Data Privacy and Ethics.

## Detailed Description
Fairness Through Awareness involves identifying potential biases in data, understanding how these biases can affect model outcomes, and implementing strategies to mitigate these biases. The primary objectives are to ensure that the model's predictions are fair and unbiased and to foster trustworthiness in AI systems.

## Key Concepts
- **Bias Identification**: Recognizing and quantifying biases in the dataset that could lead to unfair outcomes.
- **Bias Mitigation Techniques**: Applying techniques such as re-sampling, re-weighting, or modifying the training process to alleviate identified biases.
- **Fairness Metrics**: Using metrics like demographic parity, disparate impact, and equalized odds to evaluate the fairness of model predictions.

## Examples

### Example 1: Mitigating Gender Bias in Hiring Algorithms
Suppose you are building a machine learning model to screen resumes for job applicants. If historical hiring data reflects gender bias, the model might also learn and perpetuate this bias.

**Python Implementation Using `sklearn` and `fairlearn`**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

data = pd.read_csv('historical_hiring_data.csv')
X = data.drop(columns=['hired', 'gender'])
y = data['hired']
sensitive_feature = data['gender']

X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive_feature, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

dp = DemographicParity()
dp.fit(X_train, y_train, sensitive_features=sensitive_feature)

fair_model = ExponentiatedGradient(model, constraints=DemographicParity())
fair_model.fit(X_train, y_train, sensitive_features=sensitive_train)

baseline_predictions = model.predict(X_test)
fair_predictions = fair_model.predict(X_test)

# Functions to compute fairness metrics go here

print("Baseline Model Fairness Metrics:")
print(f"Demographic Parity Difference: {dp.score(X_test, baseline_predictions, sensitive_features=sensitive_test)}")

print("Fair Model Fairness Metrics:")
print(f"Demographic Parity Difference: {dp.score(X_test, fair_predictions, sensitive_features=sensitive_test)}")
```

### Example 2: Addressing Racial Bias in Loan Approval Models
Consider a machine learning model used for loan approval. Historical data might reflect racial bias, resulting in biased predictions.

**R Implementation Using `fairml`**

```r
library(fairml)

data <- read.csv('loan_approval_data.csv')
X <- data[, !(names(data) %in% c('approved', 'race'))]
y <- data$approved
sensitive_feature <- data$race

set.seed(42)
trainIndex <- sample(1:nrow(data), 0.8 * nrow(data), replace = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]
sensitive_train <- sensitive_feature[trainIndex]
sensitive_test <- sensitive_feature[-trainIndex]

baseline_model <- glm(approved ~ ., data = data[trainIndex, ], family = binomial)
predicted_probabilities <- predict(baseline_model, data[-trainIndex, ], type = "response")
predicted_labels <- ifelse(predicted_probabilities > 0.5, 1, 0)

fairness_metrics <- disparate_impact_ratio(y_test, predicted_labels, sensitive_test)
print(f"Baseline Model Fairness Metrics: {fairness_metrics}")

fair_model <- mitigate_bias(X_train, y_train, sensitive_train, model_function = "glm")

fair_predictions <- predict(fair_model, data[-trainIndex, ])
fair_fairness_metrics <- disparate_impact_ratio(y_test, fair_predictions, sensitive_test)
print(f"Fair Model Fairness Metrics: {fair_fairness_metrics}")
```

## Related Design Patterns
- **Bias Audit**: Conducting thorough audits to identify biases in training data and model outputs.
- **Algorithmic Transparency**: Ensuring the decision-making processes of models are clear and understandable to identify and mitigate biases.
- **Inclusive Design**: Designing systems to be inclusive from the onset, incorporating diverse perspectives and minimizing bias.
- **Data Provenance**: Keeping detailed records of data sources and transformations to trace and eliminate biases.

## Additional Resources
1. **Fairness and Machine Learning: Limitations and Opportunities** by Solon Barocas, Moritz Hardt, and Arvind Narayanan
2. [Fairness in Machine Learning](https://fairmlbook.org)
3. [AI Fairness 360 Toolkit](https://aif360.mybluemix.net/) by IBM
4. [Fairlearn](https://fairlearn.org) - A Python package for assessing and improving fairness in machine learning

## Summary
Fairness Through Awareness is a crucial design pattern aimed at fostering responsible AI systems by explicitly modeling and mitigating biases. By identifying and addressing biases in the data and model, machine learning practitioners can create more ethical and equitable models. This pattern is complemented by related practices such as Bias Audits and Algorithmic Transparency, contributing to a holistic approach to ethical AI.

Implementing this design pattern might require additional computational resources and effort, but the payoff in ethical, trustworthy, and inclusive AI systems far outweighs these costs.
