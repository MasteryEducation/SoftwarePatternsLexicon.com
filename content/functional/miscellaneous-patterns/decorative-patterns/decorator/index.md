---
linkTitle: "Decorator"
title: "Decorator: Attaching Additional Responsibilities"
description: "The Decorator design pattern helps to dynamically attach additional responsibilities to objects, contingent upon class or interface."
categories:
- Functional Programming
- Design Patterns
tags:
- Functional Programming
- Decorator
- Design Patterns
- Higher-Order Functions
- Functional Architecture
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/miscellaneous-patterns/decorative-patterns/decorator"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


The **Decorator** design pattern is a structural pattern that allows behavior to be added to individual objects, either statically or dynamically, without affecting the behavior of other objects from the same class. This pattern is typically employed to adhere to the Open/Closed Principle and is closely related to higher-order functions in functional programming.

## Overview of the Decorator Pattern

In traditional object-oriented programming (OOP), the Decorator pattern is used to "decorate" or "wrap" an object in another object that extends its behavior. In functional programming (FP), the pattern translates naturally to the use of higher-order functions that take a function as an argument and return a new function that extends the original's behavior.

### Components of the Decorator Pattern

1. **Component**: Defines the interface for objects to which additional responsibilities can be attached.
2. **Concrete Component**: Defines an object to which additional responsibilities can be attached.
3. **Decorator**: Maintains a reference to a Component object and defines an interface conforming to Component's interface.
4. **Concrete Decorator**: Adds responsibilities to the Component.

### Functional Programming Version

In FP, the higher-order functions serve as the decorators. These functions make the code modular and reusable by allowing enhanced behavior through functional composition.

### Example in Functional Programming

Here’s an example of a Decorator pattern implemented in a purely functional style using JavaScript:

```javascript
// Basic component function.
const basicLogger = (message) => console.log(`Basic Logger: ${message}`);

// Higher-order function that adds timestamp functionality.
const withTimestamp = (logger) => (message) => logger(`${new Date().toISOString()} - ${message}`);

// Higher-order function that adds error level functionality.
const withErrorLevel = (logger) => (message) => logger(`ERROR: ${message}`);

// Compose new functionality using decorators.
const enhancedLogger = withErrorLevel(withTimestamp(basicLogger));

// Usage
enhancedLogger("This is a decorated log message.");
// Output: ERROR: 2023-10-07T10:20:30.000Z - This is a decorated log message.
```

## Related Design Patterns

- **Chain of Responsibility**: Both patterns involve passing requests along a chain of handlers. However, the Decorator pattern focuses on adding responsibilities incrementally.
- **Adapter**: The Adapter pattern is similar in that it changes an object’s interface. However, it focuses more on interoperability between incompatible interfaces rather than enhancing the capabilities of an object.
- **Proxy**: The Proxy pattern provides a substitute or placeholder for another object to control access to it, which often involves using the Decorator’s wrapping strategy.

## Benefits of the Decorator Pattern

- **Single Responsibility Principle**: By distributing functionalities among classes, each class having a singular concern.
- **Flexibility and Extensibility**: Addresses complexities by allowing functionalities to be extended without modifying existing code (adhering to the Open/Closed Principle).

## Drawbacks of the Decorator Pattern

- **Complex Code**: Excessive use of decorators can lead to a system that is difficult to understand and maintain.
- **Debugging Difficulty**: The added layers of wrapping can make pinpointing the origin of an issue challenging.

## Tools and Libraries

- **Lodash**: The Lodash library in JavaScript provides utilities that can be considered decorators.
- **Scala**: Scala has built-in support for decorators through implicit classes.
- **Haskell**: Higher-order functions in Haskell naturally support the decorator pattern through functional composition.

## Additional Resources

- [Functional Programming Principles in Scala by Martin Odersky](https://www.coursera.org/learn/scala-functional-programming)
- _"Functional Programming in JavaScript"_ by Luis Atencio
- [Design Patterns: Elements of Reusable Object-Oriented Software](https://en.wikipedia.org/wiki/Design_Patterns) by Erich Gamma et al.

## Summary

The **Decorator** pattern in functional programming allows attaching additional responsibilities to functions through higher-order functions. This approach provides flexibility and improves code modularity, adhering to key principles like the Single Responsibility Principle and the Open/Closed Principle. While powerful, it should be used judicionsly to avoid unnecessary complexity.

By understanding and leveraging the Decorator pattern, software engineers can create highly extensible and maintainable systems, all while adhering to key functional programming principles.

---

Ready to learn more about this and other design patterns? Check out our other articles on functional programming principles and design patterns!
