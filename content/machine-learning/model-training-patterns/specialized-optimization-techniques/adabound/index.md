---
linkTitle: "AdaBound"
title: "AdaBound: Adaptive Learning Rate Bounds for Gradient-Based Optimization"
description: "AdaBound is an optimization algorithm that combines the best of adaptive gradient methods and stochastic gradient descent (SGD). It adapts the learning rate dynamically and provides a bound for the learning rates for more stable and faster convergence."
categories:
- Model Training Patterns
- Specialized Optimization Techniques
tags:
- AdaBound
- Adaptive Learning Rate
- Optimization
- Gradient Descent
- Regularization
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/model-training-patterns/specialized-optimization-techniques/adabound"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

AdaBound is an optimization algorithm designed to address the limitations of adaptive gradient methods like Adam while maintaining a balance with the robustness of stochastic gradient descent (SGD). It introduces dynamic bounds on the learning rate, which helps to stabilize and accelerate the convergence of the model training process. This article delves deeply into the details of AdaBound, its implementation, and its practical applications.

## Fundamentals of AdaBound

AdaBound stands out because of its adaptive learning rate scheduling, which smoothly transitions from an adaptive method like Adam to SGD, providing both rapid initial learning and stable long-term training. The cornerstone of AdaBound is its bound functions that limit the learning rates within a certain range, which is dynamically adjusted based on the training progression.

### Mathematical Formulation

Central to AdaBound's effectiveness are its hyperparameters and formulas. Let's start with some standard notations and build from there.

For weight updates, AdaBound uses the following formulas.

1. **Compute gradients**:

{{< katex >}}
g_t = \nabla_{\theta}L(\theta_t)
{{< /katex >}}

2. **Update bias-corrected first and second moment estimates**:

{{< katex >}}
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
{{< /katex >}}

{{< katex >}}
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
{{< /katex >}}

3. **Compute bias-corrected moments**:

{{< katex >}}
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
{{< /katex >}}

{{< katex >}}
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
{{< /katex >}}

4. **Calculate the adaptive learning rate bounds** based on lower and upper bound functions, \\( l_t \\) and \\( u_t \\):

{{< katex >}}
l_t = \eta \left(1 - \frac{1}{(1 + \gamma t)}\right)
{{< /katex >}}

{{< katex >}}
u_t = \eta \left(1 + \frac{1}{(1 + \gamma t)}\right)
{{< /katex >}}

where \\( \eta \\) and \\( \gamma \\) are hyperparameters.

5. **Clip the learning rate**:

{{< katex >}}
\eta_{t} = \min(\max(\frac{\eta}{\sqrt{\hat{v}_t}} l_t), u_t)
{{< /katex >}}

6. **Update weights** using the bound learning rate:

{{< katex >}}
\theta_{t+1} = \theta_t - \eta_t \hat{m}_t
{{< /katex >}}

## Practical Implementation

### Python Implementation with PyTorch

AdaBound has been implemented in various deep learning frameworks. Below is an example using PyTorch.

```python
import torch
from torch.optim.optimizer import Optimizer

class AdaBound(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(AdaBound, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBound does not support sparse gradients')
                
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                adapted_lr = (group['lr'] * (1 - 1 / (1 + group['gamma'] * step)))

                lower_bound = adapted_lr * (1 - 1 / (1 + group['gamma'] * step))
                upper_bound = adapted_lr * (1 + 1 / (1 + group['gamma'] * step))

                final_lr = group['final_lr'] * group['lr'] / adapted_lr
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                step_size = step_size / denom

                eta_t = torch.clamp(step_size, lower_bound, upper_bound)
                eta_t = torch.min(eta_t, final_lr/denom)
                
                p.data.addcdiv_(-eta_t * bias_correction1, exp_avg, denom)
        
        return loss

model = torch.nn.Linear(10, 2)  # simple model
optimizer = AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)

for input, target in data_loader:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
```

## Related Design Patterns

1. **Adam (Adaptive Moment Estimation)**:
   - **Description**: Utilizes running averages of both the gradients and the second moments of the gradients. Suitable for problems with sparse gradients or noisy high-dimensional space.
   - **Implementation**: 
   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   ```

2. **SGD (Stochastic Gradient Descent)**:
   - **Description**: Traditionally has a constant learning rate and is known for its robustness but can suffer from slow convergence.
   - **Implementation**:
   ```python
   optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
   ```

3. **RMSProp**:
   - **Description**: RMSProp optimizes with a moving average of squared gradients, ensuring a more balanced approach.
   - **Implementation**:
   ```python
   optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
   ```

## Additional Resources

- [Original AdaBound paper on arXiv](https://arxiv.org/abs/1902.09843)
- [PyTorch implementation of AdaBound](https://github.com/Luolc/AdaBound)
- [Comprehensive article on optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
- [Deep Learning Book by Ian Goodfellow (Chapter on Optimization Techniques)](https://www.deeplearningbook.org/)

## Summary

AdaBound provides an innovative approach to optimization by combining the fast convergence property of adaptive gradient methods with the stable convergence of stochastic gradient descent. By dynamically bounding the learning rates, AdaBound ensures that training proceeds smoothly and efficiently. If you're looking to strike a balance between adaptiveness and robustness, AdaBound presents a compelling option for your machine learning optimization needs.
