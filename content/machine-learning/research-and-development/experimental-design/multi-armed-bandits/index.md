---
linkTitle: "Multi-Armed Bandits"
title: "Multi-Armed Bandits: Optimizing the Exploration-Exploitation Trade-Off in Experiments"
description: "An advanced technique to balance exploration and exploitation during experiments, particularly useful in A/B testing, online advertising, and recommendation systems."
categories:
- Research and Development
tags:
- Experimental Design
- A/B Testing
- Online Advertising
- Recommendation Systems
- Optimization
date: 2023-10-18
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/research-and-development/experimental-design/multi-armed-bandits"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction
The Multi-Armed Bandit (MAB) problem is a classic problem in probability theory and machine learning, commonly used in scenarios where you must maximize the cumulative reward over time. The name comes from the analogy of a gambler at a casino choosing between multiple slot machines (bandits), each with an unknown but potentially different payout rate. The gambler faces the challenge of deciding which machines to play and how often, balancing the exploration of new machines and the exploitation of those machines known to provide high rewards.

## Multi-Armed Bandits in Machine Learning
In machine learning, the MAB framework is particularly powerful for applications needing to balance exploration and exploitation. These include online advertising, A/B testing, recommendation systems, and adaptive clinical trials.

### The Problem
The problem can be formalized as follows:

- You have \\( n \\) arms (choices/options/strategies).
- Each arm provides a reward from a probability distribution unknown to the algorithm.
- The objective is to maximize the total reward over a sequence of decisions.

### The Trade-Off: Exploration vs. Exploitation
The two primary strategies in MAB are:
- **Exploration**: Trying different strategies to gather more information about their potential rewards.
- **Exploitation**: Using the known information to choose the strategy with the highest expected reward.

## Algorithms for Multi-Armed Bandits

### Epsilon-Greedy Algorithm
The simplest approach is the epsilon-greedy algorithm:
1. With probability \\( \epsilon \\), explore by selecting a random arm.
2. With probability \\( 1 - \epsilon \\), exploit by selecting the arm with the highest expected reward.

```python
import random

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = [0] * n_arms  # Number of times each arm was pulled
        self.values = [0.0] * n_arms  # Average reward for each arm

    def select_arm(self):
        if random.random() > self.epsilon:
            return self.values.index(max(self.values))
        else:
            return random.randint(0, self.n_arms - 1)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

epsilon_greedy = EpsilonGreedy(n_arms=10, epsilon=0.1)
selected_arm = epsilon_greedy.select_arm()
epsilon_greedy.update(selected_arm, reward)
```

### Upper Confidence Bound (UCB)
The UCB algorithm selects arms based on upper confidence bounds for the expected reward, balancing exploration and exploitation intrinsically.

```python
import math

class UCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

    def select_arm(self):
        total_counts = sum(self.counts)
        if 0 in self.counts:
            return self.counts.index(0)
        
        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

ucb = UCB(n_arms=10)
selected_arm = ucb.select_arm()
ucb.update(selected_arm, reward)
```

### Thompson Sampling
Thompson Sampling is a Bayesian approach where arms are selected based on sampling their reward distribution estimates.

```python
import random

class Beta:
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta

    def sample(self):
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward):
        if reward == 1:
            self.alpha += 1
        else:
            self.beta += 1

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.dists = [Beta() for _ in range(n_arms)]

    def select_arm(self):
        samples = [dist.sample() for dist in self.dists]
        return samples.index(max(samples))

    def update(self, chosen_arm, reward):
        self.dists[chosen_arm].update(reward)

ts = ThompsonSampling(n_arms=10)
selected_arm = ts.select_arm()
ts.update(selected_arm, reward)
```

## Related Design Patterns

### A/B Testing
A/B testing involves comparing two versions of a product to determine which performs better. Multi-Armed Bandits offer an improved method by continuously allocating more traffic to better-performing versions.

### Contextual Bandits
This is an extension of the MAB problem where the algorithm can also consider the context (i.e., user features, environmental conditions) before making a decision.

### Reinforcement Learning
Reinforcement Learning (RL) generalizes the MAB problem to more complex decision processes involving multiple states and actions, maximizing cumulative rewards over time.

## Additional Resources

- ["Bandit Algorithms for Website Optimization"](https://www.cornell.edu/): This book by John Myles White offers extensive coverage of MAB algorithms applied to web optimization.
- ["Multi-Armed Bandit Algorithms and Empirical Designs"](https://arxiv.org/): A detailed survey article covering various MAB algorithms and their applications.
- ["Thompson Sampling for the Multi-Armed Bandit Problem"](http://proceedings.mlr.press/): This paper provides a foundational discussion on Thompson Sampling.

## Summary
The Multi-Armed Bandit problem stands at the intersection of probability, decision theory, and machine learning. It offers robust solutions to real-world problems needing efficient exploration and exploitation strategies, such as online advertisements, recommendation systems, and experimental designs like A/B testing. By carefully selecting an appropriate algorithm, one can significantly optimize the trade-off, leading to higher cumulative rewards or more effective experiments.
{{< katex />}}

