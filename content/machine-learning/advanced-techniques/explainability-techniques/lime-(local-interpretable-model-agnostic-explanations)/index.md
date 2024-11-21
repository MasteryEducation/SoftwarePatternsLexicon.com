---
linkTitle: "LIME"
title: "LIME: Providing Local Explanations for Model Predictions"
description: "Understanding how individual predictions are made in machine learning models using Local Interpretable Model-agnostic Explanations (LIME)."
categories:
- Explainability Techniques
- Advanced Techniques
tags:
- machine learning
- interpretability
- explainability
- LIME
- model-agnostic
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/explainability-techniques/lime-(local-interpretable-model-agnostic-explanations)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Machine learning models often operate as black boxes, especially when using complex algorithms such as deep learning. The LIME (Local Interpretable Model-Agnostic Explanations) design pattern provides local explanations to understand and interpret individual predictions. This design pattern is critical for building trust with users and ensuring compliance with regulations that require explainability.

## Key Concepts

LIME explains individual predictions by approximating the model locally around the prediction. Here are the fundamental concepts:

1. **Model-Agnostic**: LIME can be used with any machine learning model.
2. **Local Explanations**: Focuses on understanding individual predictions rather than the global model structure.
3. **Interpretable Models**: Uses simple, interpretable models (e.g., linear models or decision trees) to approximate the complex model locally.

## Detailed Explanation

### Algorithm Steps

1. **Generate Perturbed Samples**: Create samples by perturbing the input instance and obtain their corresponding predictions from the black-box model.
2. **Weighting by Proximity**: Assign weights to these samples based on the proximity to the original instance. Closer perturbed points are given higher weight.
3. **Training a Simple Model**: Fit an interpretable model on the perturbed samples using the weights. This interpretable model approximates the decision boundary of the complex model locally.
4. **Explainability**: The interpretable model provides an explanation for the original prediction.

### Mathematical Formulation

Given a black-box model \\( f \\) and an instance \\( x \\), the objective of LIME is to find an interpretable model \\( g \\) (such as a linear model) such that \\( g \\) approximates \\( f \\) locally around \\( x \\). This can be formalized as:

{{< katex >}}
\arg\min_{g \in G} \, \mathcal{L}(f, g, \pi_x) + \Omega(g)
{{< /katex >}}

where:
- \\( \mathcal{L} \\) is a loss function measuring the fidelity of \\( g \\) in approximating \\( f \\).
- \\( \pi_x \\) is a proximity measure between \\( x \\) and perturbed samples.
- \\( \Omega \\) is a complexity measure of the interpretable model \\( g \\).

## Example: Explaining an Image Classification Model

Let's apply LIME to a convolutional neural network (CNN) trained to classify images. We'll use Python and the `lime` package.

### Python Implementation

1. **Install Necessary Libraries**

```bash
pip install lime
pip install keras
pip install tensorflow
```

2. **Loading a Pre-trained Model and Dataset**

```python
from keras.applications import inception_v3 as inc_net
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

model = inc_net.InceptionV3()

img_path = 'path_to_image.jpg'
img = inc_net.preprocess_input(plt.imread(img_path))

img = np.array([img])
```

3. **Generate Explanations Using LIME**

```python
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()
```

### Interpretation

The output visualization shows areas of the image that most influenced the model's prediction. This builds trust and understanding of how the model arrives at a particular decision.

## Related Design Patterns

### SHAP (SHapley Additive exPlanations)

- **Description**: Provides both local and global explainability based on cooperative game theory, distributing the prediction output among features fairly.
- **Difference**: While LIME uses local linear approximations, SHAP leverages Shapley values to ensure consistency and local accuracy with an additive feature attribution method.

### PDP (Partial Dependence Plot)

- **Description**: Plots the relationship between target variables and feature(s) to show the effect of changing a feature’s value on the predicted outcome.
- **Difference**: PDP provides global interpretability by showing averaged effects, whereas LIME focuses on local interpretability.

## Additional Resources

- **LIME Repository**: [LIME GitHub](https://github.com/marcotcr/lime)
- **Key Paper**: Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- **Tutorials**: [Hands-On Tutorial](https://medium.com/@odsc/local-interpretable-model-agnostic-explanations-lime-17b6944cc9ec)

## Summary

LIME provides a powerful yet flexible way to interpret individual predictions from any machine learning model. By approximating the model locally with interpretable methods, it ensures that stakeholders can understand and trust the model's decisions. This design pattern serves as an essential tool for enhancing transparency and accountability in machine learning applications.
