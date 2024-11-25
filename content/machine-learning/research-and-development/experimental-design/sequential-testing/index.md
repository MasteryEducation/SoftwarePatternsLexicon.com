---
linkTitle: "Sequential Testing"
title: "Sequential Testing: Continuously Monitoring Experiment Results and Making Decisions"
description: "A detailed overview of the Sequential Testing design pattern in machine learning, where experiment results are continuously monitored, and decisions are made along the way."
categories:
- Research and Development
tags:
- machine learning
- design patterns
- experimental design
- sequential testing
- research
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/research-and-development/experimental-design/sequential-testing"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

Sequential Testing, in the context of machine learning experimental design, refers to a methodology where experiment results are continuously monitored and analyzed as data is collected. This approach allows for early termination of experiments when necessary (such as when a conclusive result is obtained), saving resources and enabling faster iterations.

Sequential Testing contrasts with traditional fixed-sample designs where one collects data up to a predetermined sample size before conducting any analysis. The flexibility in sequential testing allows for dynamic decision-making based on interim results, which can lead to more efficient experiment conduct, faster learning, and better resource management.

## Sequential Testing in Machine Learning

### Conceptual Framework

Sequential testing involves three main components:
1. **Interim Analysis**: Periodically assessing data at multiple points during the experiment.
2. **Early Stopping Rules**: Criteria that allow for termination of the test once conclusive results are observed.
3. **Error Control**: Maintaining control over Type I (false positive) and Type II (false negative) errors throughout the testing process.

### Key Benefits

1. **Resource Efficiency**: By terminating experiments early when conclusive results are obtained, you save on computational and time resources.
2. **Faster Insights**: Enables quicker decision-making and iterative testing, thereby accelerating the learning process.
3. **Continual Improvement**: Allows for ongoing monitoring and adaptability to changing circumstances or new insights.

### Example: A/B Testing with Sequential Analysis

Consider an A/B testing scenario where you are testing two different versions of a feature (A and B) in a web application to determine which one improves user engagement metrics.

#### Python Example Using Sequential Probability Ratio Test (SPRT)

```python
import numpy as np
from scipy.stats import norm

alpha = 0.05
beta = 0.2
mean_0 = 0.5  # Null hypothesis for A
mean_1 = 0.55 # Alternative hypothesis for B
s = 0.1       # Standard deviation (assumed to be known)

a = np.log(beta / (1 - alpha))
b = np.log((1 - beta) / alpha)

data_a = np.random.normal(mean_0, s, 100)  # Simulated data for group A
data_b = np.random.normal(mean_1, s, 100)  # Simulated data for group B

log_likelihood_ratio = 0
decision = "continue"

for i in range(min(len(data_a), len(data_b))):
    log_likelihood_ratio += np.log(norm(mean_1, s).pdf(data_b[i]) / norm(mean_0, s).pdf(data_a[i]))
    if log_likelihood_ratio <= a:
        decision = "reject H1"
        break
    elif log_likelihood_ratio >= b:
        decision = "reject H0"
        break

print(f"Decision after {i+1} samples: {decision}")
```

In this example, we use the Sequential Probability Ratio Test (SPRT), which is a common sequential testing approach. We define hypotheses for interim analysis: the null hypothesis (H0) and the alternative hypothesis (H1). The decision boundaries are derived from alpha and beta error probabilities. As data is collected, log-likelihood ratios are computed to determine if the ongoing experiment can be stopped early or should continue.

### Detailed Explanation of Components

1. **Interim Analysis**:
    - At each step, collected data samples are analyzed to estimate metrics of interest (e.g., mean user engagement scores for groups A and B).
    - The analysis is iterative, occurring after a predefined number of new data points are collected.

2. **Early Stopping Rules**:
    - Based on pre-defined thresholds (a and b), decide whether to accept the alternative hypothesis, reject the null hypothesis, or continue the experiment.
    - These thresholds are determined utilizing the probability distribution of the null and alternative hypotheses.

3. **Error Control**:
    - Traditionally, error rates in sequential analysis are controlled using approaches like the O’Brien-Fleming or Pocock boundaries.
    - These allow for maintaining a balance between Type I and Type II error rates, ensuring robust decision making.

## Related Design Patterns

### 1. **Randomized Experimentation**
   Randomized experimentation involves assigning subjects randomly to either a treatment or control group. It acts as a basis for A/B testing and can be enhanced using sequential designs.

### 2. **Adaptive Experimentation**
   In Adaptive Experimentation, the procedure adapts based on interim results, potentially shifting more subjects to the better performing treatment. Sequential testing can be coupled with adaptive methods to refine resource allocations dynamically.

### 3. **Bayesian Optimization**
   Bayesian Optimization involves a probabilistic model to explore and exploit hyperparameter tuning in machine learning models. It often incorporates sequential decision-making to converge significantly faster on optimal solutions.

## Additional Resources

1. [Sequential Analysis: A Guide for Quality Control and Survey Samples](https://www.amazon.com/Sequential-Analysis-Quality-Control-Survey/dp/0471619506) by David Siegmund.
2. [Sequential Experiment Designs for Machine Learning](https://www.jstor.org/stable/24306088) – Research articles and resources available through academic repositories like JSTOR.
3. [Carnegie Mellon University](https://www.cmu.edu/) provides Online learning resources on design patterns and experimental designs in machine learning.

## Summary

Sequential Testing in machine learning offers a powerful alternative to traditional experiment designs by focusing on continuous data monitoring and early decision-making. It comprises interim analysis, early stopping rules, and error control to improve resource efficiency and accelerate learning cycles. This pattern can be widely applied in A/B testing scenarios, adaptive experimentation, and in conjunction with techniques like Bayesian optimization to achieve more robust and agile experimental outcomes.

Embracing sequential testing not only saves computational and financial resources but also speeds up the discovery of optimal solutions in dynamic environments. Through valid statistical approaches and clear decision rules, sequential testing can be a significant asset in the arsenal of machine learning practitioners and researchers.
