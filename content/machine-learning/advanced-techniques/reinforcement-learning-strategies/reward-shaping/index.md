---
linkTitle: "Reward Shaping"
title: "Reward Shaping: Improving Learning in Reinforcement Learning"
description: "Adjusting the reward functions to guide the learning process efficiently"
categories:
- Reinforcement Learning Strategies
- Advanced Techniques
tags:
- Machine Learning
- Reinforcement Learning
- Reward Shaping
- Reward Function
- Training Efficiency
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/advanced-techniques/reinforcement-learning-strategies/reward-shaping"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Reward Shaping is a technique in reinforcement learning (RL) where the reward function is adjusted to guide the learning process more efficiently. By modifying the reward signals that an agent receives, we can expedite its learning and improve its performance in complex environments. This powerful method helps address issues like sparse rewards and long-term dependencies, making it easier for an agent to learn effective policies.

## Detailed Explanation

In reinforcement learning, the goal of the agent is to maximize the cumulative reward from its environment by taking a series of actions. The learning process is driven by a reward function, which provides feedback to the agent about the quality of its actions.

### formal definition

Let \\( R(s,a,s') \\) be the reward function that provides a scalar reward for transitioning from state \\( s \\) to state \\( s' \\) after taking action \\( a \\). Reward shaping involves designing potential-based reward functions of the form \\( F(s,s') \\) that modify the reward structure without altering the optimal policy.

### Potential Design

A common framework for reward shaping is potential-based reward shaping, where a potential function \\( \Phi \\) maps states to real numbers and the shaped reward \\( F \\) is given by:

{{< katex >}}
F(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)
{{< /katex >}}

Here:
- \\( \gamma \\) is the discount factor.
- \\( \Phi(s) \\) and \\( \Phi(s') \\) are potential functions at state \\( s \\) and resulting state \\( s' \\), respectively.

This ensures that the difference between successive potential values doesn't disturb the optimal policies, effectively guiding the agent without changing the fundamentals of the agents learning process.

## Examples

Let's illustrate reward shaping with examples in popular RL frameworks: OpenAI Gym with Python, and Unity ML-Agents with C#.

### Example in Python (OpenAI Gym)

Consider a simple grid world where an agent must reach a goal state. The original environment might only provide a reward when the agent reaches the goal, making learning slow. Using reward shaping, we can add intermediate rewards to guide the agent more efficiently.

```python
import gym
import numpy as np

class ShapedRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.goal_state = (4, 4)
    
    def reward_shaping(self, old_state, new_state):
        ''' A example potential-based reward shaping function. '''
        old_distance = np.linalg.norm(np.array(old_state) - np.array(self.goal_state))
        new_distance = np.linalg.norm(np.array(new_state) - np.array(self.goal_state))
        # Shaping reward based on distance reduction to goal.
        return old_distance - new_distance
    
    def step(self, action):
        old_state = self.env.agent_pos
        new_state, original_reward, done, info = self.env.step(action)
        shaping_reward = self.reward_shaping(old_state, new_state)
        # combined reward
        total_reward = original_reward + shaping_reward
        return new_state, total_reward, done, info

env = gym.make('GridWorld-v0')
shaped_env = ShapedRewardEnv(env)
```

### Example in C# (Unity ML-Agents)

In a similar grid world implemented in Unity with ML-Agents, the reward shaping concept can be applied directly using C#.

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class ShapedRewardAgent : Agent
{
    public Transform goal;
    
    public override void OnEpisodeBegin()
    {
        // Reset agent and environment
        transform.localPosition = new Vector3(start_x, 0, start_z);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(goal.position);
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = vectorAction[0];
        controlSignal.z = vectorAction[1];
        transform.Translate(controlSignal * Time.deltaTime * speed);

        // Calculate shaped reward based on distance to the goal
        float distanceToGoal = Vector3.Distance(transform.localPosition, goal.localPosition);
        float shapedReward = -distanceToGoal;
        
        // Provide the shaped reward
        SetReward(shapedReward);

        if (distanceToGoal < 1.42f)
        {
            // Successfully achieved the goal
            SetReward(1.0f);
            EndEpisode();
        }
    }
}
```

## Related Design Patterns

- **Curriculum Learning**: Sequentially increasing the complexity of tasks to build up the agent's skills gradually.
  
  In Curriculum Learning, simpler tasks are introduced first and complexity is scaled up as performance improves. It is closely related to Reward Shaping in guiding the learning process.

- **Hindsight Experience Replay (HER)**: Originally used in policy learning via trial and error, HER helps improve sample efficiency by learning from past experiences that didn't necessarily lead to success.

  HER can complement Reward Shaping by providing augmented rewards in scenarios where traditional methods may fail due to sparse rewards.

## Additional Resources

- **Books**:
  - "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto
  - "Deep Reinforcement Learning Hands-On" by Maxim Lapan

- **Research Papers**:
  - Ng, Andrew Y., and Stuart Russell. "Algorithms for Inverse Reinforcement Learning." ICML 2000.
  - Devlin, Sam, and Daniel Kudenko. "Theoretical considerations of potential-based reward shaping for multi-agent systems." AAMAS, 2011.

- **Online Courses**:
  - "Reinforcement Learning Specialization" on Coursera by University of Alberta
  - "Deep Reinforcement Learning (DRL) Nanodegree" on Udacity

## Summary

Reward Shaping in reinforcement learning focuses on making adjustments to the reward function received by an agent, thus guiding the agent more effectively toward optimal policies. By implementing potential-based reward shaping, the agent can receive more informative feedback, helping it deal with sparse rewards and complex environments efficiently. This technique, closely related to Curriculum Learning and Hindsight Experience Replay, is indispensable for developing high-performing RL agents.

The provided Python and C# examples demonstrate practical implementations in OpenAI Gym and Unity ML-Agents, highlighting the achievable benefits of reward shaping in diverse scenarios. With insightful extra resources mentioned, readers are encouraged to explore further and integrate these principles into their RL projects.
