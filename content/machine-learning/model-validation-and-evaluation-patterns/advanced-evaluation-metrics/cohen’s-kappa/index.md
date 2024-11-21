---
linkTitle: "Cohen’s Kappa"
title: "Cohen’s Kappa: Measure of Agreement Between Two Raters"
description: "An advanced evaluation metric used to measure the agreement between two raters for categorical items, going beyond mere percentage agreement by considering the agreement occurring by chance."
categories:
- Model Validation and Evaluation Patterns
tags:
- Validation
- Evaluation Metrics
- Classification
- Cohen's Kappa
- Rater Agreement
date: 2023-10-10
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-validation-and-evaluation-patterns/advanced-evaluation-metrics/cohen’s-kappa"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Cohen’s Kappa is a statistic that measures inter-rater agreement for categorical items. It's used to quantify the degree to which two raters agree beyond what is expected by random chance, making it a robust measure in scenarios where simple percentage agreement may not suffice. This makes Cohen’s Kappa especially useful in the evaluation of qualitative (categorical) data.

## Formula and Interpretation

The Cohen's Kappa coefficient (\\(\kappa\\)) is calculated with the following formula:

{{< katex >}}
\kappa = \frac{P_o - P_e}{1 - P_e}
{{< /katex >}}

Where:
- \\(P_o\\) is the observed agreement proportion.
- \\(P_e\\) is the expected agreement proportion by chance.

- **\\(P_o\\):** This is the proportion of times the raters agree. If out of \\(N\\) items, the raters agree on \\(N_{\text{agree}}\\) items, then \\(P_o = \frac{N_{\text{agree}}}{N}\\).

- **\\(P_e\\):** This is calculated based on the marginal totals of the ratings provided by each rater.

### Calculation Example

Suppose we have a contingency table from two raters (Rater A and Rater B) who classify items into categories Positive and Negative:

|                    | Rater B: Positive | Rater B: Negative | Total    |
|--------------------|-------------------|-------------------|---------|
| **Rater A: Positive** | 50                | 10                | 60       |
| **Rater A: Negative** | 5                 | 35                | 40       |
| **Total**              | 55                | 45                | 100     |

1. **Observed Agreement \\(P_o\\):**
   The observed agreement is the sum of the diagonal elements (raters agree) divided by the total number of items:
   
   {{< katex >}}
   P_o = \frac{50 + 35}{100} = \frac{85}{100} = 0.85
   {{< /katex >}}

2. **Expected Agreement \\(P_e\\):**
   The expected agreement by chance is calculated using the marginal probabilities:
   
   {{< katex >}}
   P_e = \left( \frac{(50 + 10) \times (50 + 5)}{100^2} \right) + \left( \frac{(35 + 5) \times (10 + 35)}{100^2} \right)
   {{< /katex >}}
   
   {{< katex >}}
   P_e = \left( \frac{60 \times 55}{100^2} \right) + \left( \frac{40 \times 45}{100^2} \right)
   {{< /katex >}}

   {{< katex >}}
   P_e = \left( 0.33 \right) + \left( 0.18 \right) = 0.51
   {{< /katex >}}

3. **Cohen's Kappa \\(\kappa\\):**

   {{< katex >}}
   \kappa = \frac{0.85 - 0.51}{1 - 0.51} = \frac{0.34}{0.49} \approx 0.69
   {{< /katex >}}

### Interpretation of Cohen's Kappa

Cohen's Kappa generally ranges from -1 to 1:

- \\( \kappa = 1 \\) indicates perfect agreement.
- \\( \kappa = 0 \\) indicates no agreement beyond what is expected by chance.
- \\( \kappa < 0 \\) indicates less agreement than expected by chance.
- Guidelines for interpretation (often cited but should be used carefully):
  - \\( \kappa < 0.2 \\): Slight agreement.
  - \\( \kappa \in [0.21, 0.40]\\): Fair agreement.
  - \\( \kappa \in [0.41, 0.60]\\): Moderate agreement.
  - \\( \kappa \in [0.61, 0.80]\\): Substantial agreement.
  - \\( \kappa \in [0.81, 1.00]\\): Almost perfect agreement.

## Implementation

### Python

Using the `sklearn` library in Python, Cohen's Kappa can be computed easily:

```python
from sklearn.metrics import cohen_kappa_score

rater1 = [1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
rater2 = [1, 0, 1, 1, 1, 0, 0, 0, 0, 1]

kappa_score = cohen_kappa_score(rater1, rater2)
print(f'Cohen\'s Kappa: {kappa_score}')
```

### R

In R, the `irr` package can be used:

```R
library(irr)

ratings <- matrix(c(1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1), ncol = 2)
kappa <- kappa2(ratings, "unweighted")
print(kappa)
```

## Related Design Patterns

### Confusion Matrix

A confusion matrix provides a more granular insight into the performance of classification models by showing the count of true positive, false positive, true negative, and false negative predictions. It's used as a foundational building block for many evaluation metrics, including Cohen’s Kappa.

### F1 Score

F1 Score is another measure of a model's accuracy that balances precision and recall. While Cohen's Kappa adjusts for chance agreement, F1 Score emphasizes the harmonic mean of precision and recall.

### Matthews Correlation Coefficient (MCC)

MCC balances all four confusion matrix categories (TP, FP, TN, FN) into a single correlation coefficient. It's particularly useful for binary classifications with imbalanced classes, akin to Cohen’s Kappa which accounts for chance agreement.

## Additional Resources

1. **Research Paper:** Cohen, J. (1960). "A coefficient of agreement for nominal scales." Educational and Psychological Measurement, 20(1), 37-46.
2. **Online Tutorials:** Introduction to Cohen's Kappa: [YouTube Video](https://www.youtube.com/watch?v=zT9qsm5zp4Y)
3. **Scikit-learn Documentation:** [cohen_kappa_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)

## Summary

Cohen's Kappa is a robust statistic for evaluating inter-rater reliability on categorical data, going beyond simple percent agreement by accounting for expected agreement by chance. Its values range from -1 to 1, providing intuitive interpretation guidelines. It is widely implemented in various machine learning and statistical libraries, making it accessible for different types of data validation tasks. Understanding and applying Cohen's Kappa can significantly enhance the reliability assessment of classification models and improve decision-making processes in model evaluation.

---

By mastering Cohen's Kappa, practitioners can ensure their models or classification systems are effectively evaluated for consistency and reliability, thus making more informed decisions in real-world applications.
