---
linkTitle: "Counterfactual Explanations"
title: "Counterfactual Explanations: Providing Explanations by Showing What Changes to Input Would Change the Output"
description: "A detailed examination of Counterfactual Explanations in machine learning, focusing on how modifying input variables can lead to different outcomes, thus enhancing the interpretability of ML models."
categories:
- Emerging Fields
- Explainable AI (XAI)
tags:
- Explainable AI
- XAI
- Counterfactual Explanations
- Interpretability
- Transparency
date: 2023-10-15
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/emerging-fields/explainable-ai-(xai)/counterfactual-explanations"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Counterfactual explanations are a robust mechanism within the domain of Explainable AI, aimed at elucidating the decision-making process of machine learning models. The central idea is to reveal how slight modifications to the input variables could alter the machine learning model's output. This technique can significantly increase the transparency and trustworthiness of ML models, especially in critical applications like healthcare, finance, and legal systems.

## Detailed Explanation

Counterfactual explanations are based on presenting an alternative scenario that leads to a different outcome. Given the input data point and its corresponding prediction from an ML model, the goal is to identify and present the minimal changes necessary to achieve a desired output. 

### Defining Counterfactuals

Mathematically, let \\( \mathbf{x} \\) be an input vector and \\( \mathbf{y} = f(\mathbf{x}) \\) be the output from a model \\( f \\). A counterfactual explanation for a different desired output \\( \mathbf{y}' \\) involves finding an input \\( \mathbf{x}' \\) such that:
{{< katex >}} f(\mathbf{x}') = \mathbf{y}' {{< /katex >}}

Ideally, \\( \mathbf{x}' \\) should be as close to \\( \mathbf{x} \\) as possible according to some distance metric \\( d(\mathbf{x}, \mathbf{x}') \\).

## Examples

### Credit Approval Scenario

Consider a credit approval application utilizing a decision tree model. Let \\( \mathbf{x} = (35, 75000, 2) \\) be an input vector representing age, income, and number of existing loans. Suppose the model denies the credit application (\\( y = 0 \\)) but the applicant wishes to know what needs to change to get approval (\\( y' = 1 \\)). A counterfactual explanation might suggest:
{{< katex >}} \mathbf{x}' = (35, 85000, 2) {{< /katex >}}
This indicates that increasing the income to $85,000 might change the result to approval.

### Python Implementation Using Dice Library

Here's an example using the [DiCE library](https://github.com/interpretml/DiCE), a popular library for generating diverse counterfactual explanations.

```python
import dice_ml
from dice_ml import Dice
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('samples/credit.csv')
features = ['age', 'income', 'loan amount']
data[features] = data[features].fillna(0)
model = RandomForestClassifier()
model.fit(data[features], data['approved'])

dice_data = dice_ml.Data(dataframe=data, continuous_features=['age', 'income'], outcome_name='approved')
dice_model = dice_ml.Model(model=model, backend="sklearn")
explainer = Dice(dice_data, dice_model)

query_instance = pd.DataFrame({'age': 35, 'income': 75000, 'loan amount': 2, 'approved': 0}, index=[0])
counterfactuals = explainer.generate_counterfactuals(query_instance, total_CFs=1, desired_class='opposite')
print(counterfactuals.cf_examples_list[0].final_cfs_df)
```

## Related Design Patterns

### 1. **Interpretable Models**
   - **Description:** Utilize simple and transparent models like linear regression or decision trees, which are inherently interpretable, as opposed to complex, black-box models.
   - **Use Case:** When transparency is paramount and the accuracy-performance tradeoff is acceptable.

### 2. **Local Interpretable Model-agnostic Explanations (LIME)**
   - **Description:** LIME approximates black-box models locally with an interpretable model to explain individual predictions.
   - **Use Case:** When specific predictions need explanation without changing the main model.

### 3. **SHapley Additive exPlanations (SHAP)**
   - **Description:** A unified measure based on cooperative game theory to explain the output of machine learning models for individual predictions.
   - **Use Case:** When a consistent and axiomatic explanation of predictions is needed across features.

## Additional Resources

- **Research Papers:**
  - [Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR](https://arxiv.org/abs/1711.00399)
  - [Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations](https://arxiv.org/abs/1905.07697)
  
- **Libraries and Tools:**
  - [DiCE (Diverse Counterfactual Explanations)](https://github.com/interpretml/DiCE)
  - [Alibi-Explain](https://github.com/SeldonIO/alibi)

- **Courses and Tutorials:**
  - [Coursera - Explainable AI (University of Edinburgh)](https://www.coursera.org/learn/explainable-ai)
  - [Kaggle - Interpret Machine Learning Models](https://www.kaggle.com/learn/machine-learning-explainability)

## Summary

Counterfactual explanations play a critical role in enhancing the interpretability of machine learning models. By showcasing how slight alterations to input variables result in differentiated outcomes, they offer valuable insights into the model's decision-making process. Particularly useful in contexts demanding high transparency and accountability, such as credit scoring and healthcare, they contribute to safer and more ethical AI applications. Understanding this pattern encourages the development of more trustworthy and user-friendly ML systems, promoting the integration of AI systems into sensitive real-world domains.
