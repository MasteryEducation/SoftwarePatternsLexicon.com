---
linkTitle: "Escape Hatches"
title: "Escape Hatches: Providing Ways for Users or Systems to Override Automated Decisions"
description: "The Escape Hatches design pattern aims to offer mechanisms that allow users or systems to override automated decisions, ensuring safety and fairness in machine learning applications."
categories:
- Safety and Fairness Patterns
- Safety Assurance
tags:
- machine learning
- design patterns
- safety
- fairness
- override
date: 2024-10-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/safety-and-fairness-patterns/safety-assurance/escape-hatches"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

The **Escape Hatches** design pattern enables users or systems to override automated decisions made by machine learning models. This safety assurance approach can prevent catastrophic failures or unfair outcomes by providing built-in mechanisms that allow interventions when necessary. These intervening mechanisms are critical in promoting transparency, trust, and responsibility in machine learning systems.

## Why Escape Hatches Are Important

Automated systems are not infallible and can sometimes make erroneous or biased decisions. Allowing a way to override such decisions can:

- **Increase Trust**: Users are more likely to trust the system if they believe they have control over its decisions.
- **Improve Safety**: Immediate human intervention can prevent potential harm caused by wrong decisions.
- **Enhance Fairness**: Reduce biases and ensure fair decision-making by allowing checks and balances.

## Implementation Examples

### Example 1: Aircraft Control Systems

In aviation, autopilot systems are equipped with escape hatches in the form of manual takeovers. Pilots can override the autopilot if they notice the system making unsafe maneuvers.

- **Language**: Pseudocode

```pseudocode
if pilot_override:
    disengage_autopilot()
    switch_to_manual_control()
else:
    continue_autopilot_operations()
```

### Example 2: Recommendation Systems

Online platforms often provide users a way to see or modify their recommendation preferences, thus overriding automated suggestions.

- **Language**: Python

```python
class RecommendationSystem:
    def __init__(self, model):
        self.model = model
        self.user_inputs = {}

    def recommend(self, user_id):
        recommendations = self.model.predict(user_id)
        if user_id in self.user_inputs:
            return self.user_inputs[user_id]
        return recommendations
    
    def add_user_input(self, user_id, user_preferences):
        self.user_inputs[user_id] = user_preferences

system = RecommendationSystem(model)
system.add_user_input(user_id=123, user_preferences=[item1, item2])
recommendations = system.recommend(user_id=123)
```

### Example 3: Autonomous Vehicles

In autonomous driving systems, there is always an option for the driver to manually take control of the vehicle.

- **Language**: JavaScript (using a state machine library)

```javascript
const vehicleStates = {
  AUTONOMOUS: 'autonomous',
  MANUAL: 'manual'
};

class AutonomousVehicle {
  constructor() {
    this.state = vehicleStates.AUTONOMOUS;
  }

  manualOverride() {
    if (this.state === vehicleStates.AUTONOMOUS) {
      this.state = vehicleStates.MANUAL;
      console.log("Switching to manual control.");
    }
  }

  resumeAutonomous() {
    if (this.state === vehicleStates.MANUAL) {
      this.state = vehicleStates.AUTONOMOUS;
      console.log("Resuming autonomous control.");
    }
  }
}
```

## Related Design Patterns

### Human-in-the-Loop (HITL)

**Description**: In this pattern, human operators are involved in the decision-making process of a machine learning system. HITL can range from review and approval of decisions to manual labeling of data to improve model accuracy.

### Shadow Mode

**Description**: In shadow mode, a new model runs in parallel with the primary decision-making system without affecting its operation. This approach allows monitoring and comparison without risk, which can identify when human overrides might be necessary.

### Safe Exit

**Description**: This pattern involves creating a safe fallback mechanism when the automated system fails. Similar to escape hatches, it aims to mitigate risks by smoothly transitioning the control to an alternative system.

## Additional Resources

- [Trust in AI: How to Ensure Safety and Fairness in Machine Learning](https://www.trustedai.org)
- [The Role of Human-in-the-Loop in AI Systems](https://www.hitl-ai.org)
- [Autonomous Vehicle Safety Measures](https://www.autonomousvehiclessafety.com)

## Summary

The **Escape Hatches** design pattern is vital for ensuring the safety and fairness of machine learning systems. By allowing users or systems to override automated decisions, this pattern helps in increasing trust, improving safety, and enhancing fairness. Life-critical systems, recommendation engines, and autonomous vehicles are just a few examples where escape hatches play an essential role. Understanding and implementing this design pattern is a key step towards responsible machine learning applications.

---
