---
linkTitle: "State"
title: "State: Allowing an object to alter its behavior when its internal state changes."
description: "In functional programming, the 'State' design pattern encapsulates state management within an object, enabling it to change its behavior when its internal state changes without using mutable shared data. This pattern helps maintain immutability and promotes a declarative approach to managing state."
categories:
- Functional Programming
- Design Patterns
tags:
- State
- Functional Programming
- Immutability
- State Management
- Behavioral Patterns
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/behavioral-patterns/interactions/state"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


The **State** design pattern is a behavioral software design pattern that allows an object to change its behavior when its internal state changes. In the context of functional programming, it focuses on state management without relying on mutable shared data, aligning with the principles of immutability and declarative code. Here's a detailed exploration of the **State** design pattern and how it can be applied using functional programming techniques.

## Key Concepts

### State Representation
In functional programming, state can be represented using pure functions, avoiding side effects. This leads to code that is easier to reason about, test, and debug. 

### Immutability
State should remain immutable. When a state transition occurs, a new state is produced instead of modifying the existing state.

### Pure Functions
State transitions should be handled by pure functions that compute the new state based on the current state and the input.

## Detailed Explanation

### Traditional Example in Object-Oriented Programming
In Object-Oriented Programming (OOP), the State pattern typically involves a `Context` object that maintains an instance of a `State` subclass representing the current state. As the state changes, the behavior of the `Context` object changes.

Here’s a basic OOP example:

```java
abstract class State {
    abstract void handle(Context context);
}

class Context {
    private State state;

    void setState(State state) {
        this.state = state;
    }

    void request() {
        state.handle(this);
    }
}

class ConcreteStateA extends State {
    void handle(Context context) {
        System.out.println("State A");
        context.setState(new ConcreteStateB());
    }
}

class ConcreteStateB extends State {
     void handle(Context context) {
        System.out.println("State B");
        context.setState(new ConcreteStateA());
     }
}
```

### Functional Programming Variant
In functional programming, we can represent the state and transitions as pure functions and use function composition to manage state changes. Here’s how one might implement the State pattern functionally using JavaScript:

```javascript
const stateA = {
    name: 'StateA',
    handle: (context) => {
        console.log('State A');
        return { ...context, state: stateB };
    }
};

const stateB = {
    name: 'StateB',
    handle: (context) => {
        console.log('State B');
        return { ...context, state: stateA };
    }
};

const initialContext = {
    state: stateA
};

const transition = (context) => context.state.handle(context);

let currentContext = initialContext;
currentContext = transition(currentContext); // State A
currentContext = transition(currentContext); // State B
currentContext = transition(currentContext); // State A
```

### State Transition Representation in Haskell

In Haskell, we usually deal with state transitions using State monads. Here’s a simple representation:

```haskell
import Control.Monad.State

data StateType = StateA | StateB deriving (Show, Eq)

transition :: StateType -> State StateType ()
transition StateA = do
    liftIO $ putStrLn "State A"
    put StateB
transition StateB = do
    liftIO $ putStrLn "State B"
    put StateA

runStateTransitions :: State StateType ()
runStateTransitions = do
    transition StateA
    transition StateB
    transition StateA

main :: IO ()
main = evalStateT runStateTransitions StateA
```

### State Transition as a Director

We can think of states and transitions as being coordinated by a director function:

```typescript
type State = (context: Context) => Context;

interface Context {
    state: State;
}

const stateA: State = (context: Context): Context => {
    console.log('State A');
    return { ...context, state: stateB };
};

const stateB: State = (context: Context): Context => {
    console.log('State B');
    return { ...context, state: stateA };
};

const transition = (context: Context) => context.state(context);

let context: Context = { state: stateA };
context = transition(context); // Logs: State A
context = transition(context); // Logs: State B
context = transition(context); // Logs: State A
```

## Related Design Patterns

### State Monad
The State Monad in functional programming is a common design pattern related to the State design pattern. It allows for encapsulated state transformation functions, preserving immutability and functional purity.

### Strategy Pattern
The Strategy pattern is related to the State pattern wherein the strategy, like the state, can be changed at runtime. They both encapsulate an algorithm's details, but Strategy does not represent the state of an object.

## Additional Resources

- "Functional Programming in Scala" by Paul Chiusano and Runar Bjarnason
- "Real-World Haskell" by Bryan O'Sullivan, Don Stewart, and John Goerzen
- [Haskell State Monad Tutorial](http://learnyouahaskell.com/for-a-few-monads-more#state)

## Summary

The **State** design pattern in functional programming facilitates state management with a focus on immutability and pure functions. By leveraging function composition and encapsulating state transitions within pure functions, this pattern maintains the core principles of functional programming while providing flexible, easily understandable state changes. Related patterns like the State Monad and the Strategy pattern further enhance the functional approach to handle state and behavior.

Implementing this pattern helps in achieving clean, maintainable, and highly testable code, addressing the complex task of state management in a declarative and functional way.
