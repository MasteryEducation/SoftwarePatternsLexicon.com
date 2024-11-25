---
linkTitle: "Rule Extraction"
title: "Rule Extraction: Converting complex model decisions into human-readable rules"
description: "An explainability technique for extracting rules from complex machine learning models to provide human-readable explanations of model decisions."
categories:
- Advanced Techniques
tags:
- Machine Learning
- Explainability
- Rule Extraction
- Interpretability
- Decision Rules
date: 2023-10-05
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/explainability-techniques/rule-extraction"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction to Rule Extraction

In an era where machine learning models are becoming ever more complex, interpretability has become a crucial aspect of their deployment. **Rule Extraction** is an explainability technique aimed at converting complex model decisions into human-readable rules. This ensures that critical decisions made by models, such as those involving healthcare or finance, can be easily understood and justified.

## How Rule Extraction Works

Rule Extraction involves deriving a set of if-then rules from a complex model's decision boundary. These rules aim to approximate the model's behavior, providing insight into how it makes decisions. This process helps in scrutinizing the model for biases and ensuring its alignment with domain knowledge or legal requirements.

### Mathematical Formulation

Given a complex model \\( f \\) that maps an input \\( X \\) to an output \\( y \\):

1. **Objective:**
   {{< katex >}}
   \text{Find } \{ R_i \}_{i=1}^N \text{ such that } f(X) \approx R(X)
   {{< /katex >}}

   where \\( R(X) = \{ R_i \}_{i=1}^N \text{ represents a set of human-readable rules. } \\)


2. **Rule Formulation:**

   {{< katex >}}
   R_i : \bigwedge_{j} \text{Condition}_{ij} \rightarrow \text{Decision}_i
   {{< /katex >}}

   where \\( \text{Condition}_{ij} \\) are boolean expressions involving feature \\( X_j \\).

## Techniques for Rule Extraction

### Example-Based

#### Decision Trees
- **Explanation:** Convert decision paths into if-then rules.
- **Advantage:** Straightforward and intuitive.
- **Example:** 

  ```python
  from sklearn.tree import DecisionTreeClassifier, export_text

  # Train a DecisionTreeClassifier
  dt = DecisionTreeClassifier().fit(X_train, y_train)
  
  # Extract rules
  rules = export_text(dt, feature_names=feature_names)
  print(rules)
  ```

#### RuleFit Algorithm
- **Explanation:** Combines rule-based systems with linear models.
- **Advantage:** More expressive than simple decision trees.

  ```python
  from rulefit import RuleFit
  
  # Train RuleFit
  rf = RuleFit().fit(X_train, y_train)
  
  # Extract rules
  rules = rf.get_rules()
  rules = rules[rules.coef != 0]  # Filter relevant rules
  print(rules)
  ```

### Model-Based

#### Support Vector Machine with Rule Extraction
- **Explanation:** Use rule extraction techniques like S+VSM.
- **Advantage:** Understand reasoning behind SVM classifications.

### Post-Hoc Interpretation

#### LIME (Local Interpretable Model-agnostic Explanations)
- **Explanation:** Generates local surrogate models to interpret predictions.
- **Advantage:** Model-agnostic and works on any black-box model.

  ```python
  import lime
  from lime.lime_tabular import LimeTabularExplainer
  
  # Create LIME explainer
  explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names, discretize_continuous=True)
  
  # Explain a prediction
  exp = explainer.explain_instance(X_test[0], model.predict)
  exp.show_in_notebook()
  ```

#### SHAP (SHapley Additive exPlanations)
- **Explanation:** Uses Shapley values from cooperative game theory.
- **Advantage:** Consistent and local attributions for each feature.

  ```python
  import shap
  
  # Create SHAP explainer
  explainer = shap.TreeExplainer(model)
  
  # Explain a prediction
  shap_values = explainer.shap_values(X_test)
  shap.summary_plot(shap_values, X_test)
  ```

## Examples

### Decision Tree Example in Scikit-Learn

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
X, y = iris.data, iris.target

dt = DecisionTreeClassifier().fit(X, y)

rules = export_text(dt, feature_names=iris.feature_names)
print(rules)
```

### Rule Extraction Using RuleFit

```python
from rulefit import RuleFit

rf = RuleFit()
rf.fit(X_train, y_train)

rules = rf.get_rules()
rules_subset = rules[rules.coef != 0]  # Only display rules with non-zero coefficients
print(rules_subset)
```

## Related Design Patterns

### 1. **Model Monitoring**
   - **Description:** Continuously monitor model performance and behavior in production to detect and mitigate issues.
   - **Relation:** Both enhance trust in ML systems, but Rule Extraction focuses on interpretability while Monitoring aims at observing stability and performance.

### 2. **Feature Importance**
   - **Description:** Identify and quantify the importance of each feature in the model.
   - **Relation:** Rule Extraction often uses feature importance as a basis for forming rules.

### 3. **Ensemble Stacking**
   - **Description:** Combine multiple models to improve performance.
   - **Relation:** Rules can be extracted from each model in an ensemble to understand the composite decision-making process.

## Additional Resources

1. **Books:**
   - "Interpretable Machine Learning: A Guide for Making Black Box Models Explainable" by Christoph Molnar.

2. **Papers:**
   - Pedro Domingos, "Knowledge Discovery with Support Vector Machines," 1998.

3. **Online Tutorials:**
   - Coursera course on "Interpretable Machine Learning" by the University of Tuebingen.

## Summary

Rule Extraction is a vital explainability technique that translates complex model predictions into understandable if-then rules. This process benefits stakeholders by providing transparency and aiding in compliance with ethical and legal standards. Through various methods such as decision trees, the RuleFit algorithm, and interpretability tools like LIME and SHAP, Rule Extraction offers multiple avenues to demystify the often opaque nature of machine learning models.

By adopting Rule Extraction techniques, data scientists and machine learning engineers can create models that are both powerful and interpretable, bridging the gap between technical accuracy and real-world accountability.
