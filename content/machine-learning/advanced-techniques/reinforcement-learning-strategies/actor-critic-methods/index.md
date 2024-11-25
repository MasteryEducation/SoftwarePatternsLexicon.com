---
linkTitle: "Actor-Critic Methods"
title: "Actor-Critic Methods: Combining Value-based and Policy-based Methods for Stabilization"
description: "Actor-Critic Methods combine value-based and policy-based approaches to address stabilization issues in reinforcement learning through sample-efficiency and convergence improvement."
categories:
- Reinforcement Learning Strategies
- Advanced Techniques
tags:
- Actor-Critic
- Reinforcement Learning
- Machine Learning
- Algorithms
- Policy Gradient
date: 2023-10-09
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/reinforcement-learning-strategies/actor-critic-methods"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Actor-Critic methods are a sophisticated approach in reinforcement learning (RL) that blend value-based and policy-based methods to improve stability and convergence. These methods address some limitations of purely value-based methods (e.g., Q-learning) and purely policy-based methods (e.g., REINFORCE). By maintaining two distinct functions—one to represent the policy (the actor) and another to estimate the value function (the critic)—these methods achieve more efficient learning and better performance.

## Theoretical Background

Actor-Critic methods are grounded in two main components:

1. **Actor**: The actor is responsible for selecting actions according to a certain policy $\pi(a|s)$, which is the probability distribution over actions given the state.
2. **Critic**: The critic evaluates the action taken by the actor by estimating the value function $V^\pi(s)$ or the action-value function $Q^\pi(s, a)$.

### Mathematical Formulation

The policy $\pi_\theta(a|s)$, parameterized by $\theta$, guides the choosing of actions. The critic, on the other hand, uses a value function $V_w(s)$ or $Q_w(s, a)$, parameterized by $w$, to provide feedback on the actions taken.

Using the Temporal-Difference (TD) error $\delta_t$, the critic updates its parameters as follows:
{{< katex >}} \delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t) {{< /katex >}}

For the actor, the policy parameters $\theta$ are updated in the direction suggested by the critic:
{{< katex >}} \theta \leftarrow \theta + \alpha \delta_t \nabla_\theta \log \pi_\theta(a_t|s_t) {{< /katex >}}

The critic updates according to:
{{< katex >}} w \leftarrow w + \beta \delta_t \nabla_w V_w(s_t) {{< /katex >}}

### Combined Approach

By combining these approaches, Actor-Critic methods utilize the critic's feedback to stabilize the actor's updates, while the actor ensures exploration of the policy space.

## Implementation Examples

### Python Implementation using TensorFlow

Here's a Python example implementing an Actor-Critic method using TensorFlow:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ActorCritic:

    def __init__(self, state_size, action_size, stddev=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.stddev = stddev
    
        self.actor_model = self._build_actor()
        self.critic_model = self._build_critic()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_size,))
        hidden = layers.Dense(24, activation='relu')(inputs)
        outputs = layers.Dense(self.action_size, activation='softmax')(hidden)
        return tf.keras.Model(inputs, outputs)

    def _build_critic(self):
        inputs = layers.Input(shape=(self.state_size,))
        hidden = layers.Dense(24, activation='relu')(inputs)
        outputs = layers.Dense(1, activation='linear')(hidden)
        return tf.keras.Model(inputs, outputs)

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0

            while True:
                action_probs = self.actor_model(state)
                action = np.random.choice(self.action_size, p=np.squeeze(action_probs))
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                value = self.critic_model(state)
                next_value = self.critic_model(next_state)

                td_target = reward + (1 - done) * 0.99 * np.squeeze(next_value)
                td_error = td_target - np.squeeze(value)

                with tf.GradientTape() as tape:
                    actor_loss = -tf.math.log(action_probs[0, action]) * td_error
                actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                self.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
                
                with tf.GradientTape() as tape:
                    value_loss = td_error ** 2
                critic_grads = tape.gradient(value_loss, self.critic_model.trainable_variables)
                self.optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

                state = next_state
                total_reward += reward
                
                if done:
                    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
                    break

import gym
env = gym.make('CartPole-v1')
agent = ActorCritic(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
agent.train(env, num_episodes=500)
```

### Related Design Patterns

1. **Q-Learning**: A value-based method that focuses solely on assessing the value of actions to derive optimal policies. It does not directly model policies, unlike Actor-Critic.
2. **Policy Gradient Methods**: Directly represent and update the policy without explicitly modeling value functions. Examples include REINFORCE and its variants.
3. **Deep Deterministic Policy Gradient (DDPG)**: An algorithm that extends Actor-Critic methods into continuous action spaces by combining DQN (value-based) with deterministic policy learning.

### Additional Resources

1. [David Silver’s Reinforcement Learning Course](https://www.davidsilver.uk/teaching/)
2. [DeepMind Resources on Deep Reinforcement Learning](https://deepmind.com/research)
3. [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)

## Summary

Actor-Critic methods provide a balanced approach by combining the strengths of value-based and policy-based reinforcement learning techniques. This hybrid strategy results in more stable and efficient learning by utilizing a policy to guide actions and a value function to critique outcomes. Accessible implementations in frameworks like TensorFlow make it practical and valuable for various applications ranging from game-playing to autonomous control.

Mastering Actor-Critic methods opens up numerous possibilities for implementing robust, efficient, and scalable RL models. Its contribution to the field of reinforcement learning showcases the power of combining different methodologies to achieve superior results.
