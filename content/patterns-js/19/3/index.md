---

linkTitle: "19.3 Design Pattern Reference Chart"
title: "Design Pattern Reference Chart for JavaScript and TypeScript"
description: "A comprehensive reference chart summarizing key design patterns in JavaScript and TypeScript, including their intent, applicability, and consequences."
categories:
- Software Design
- JavaScript
- TypeScript
tags:
- Design Patterns
- JavaScript
- TypeScript
- Software Architecture
- Programming
date: 2024-10-25
type: docs
nav_weight: 1930000
canonical: "https://softwarepatternslexicon.com/patterns-js/19/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 19.3 Design Pattern Reference Chart

Design patterns are essential tools in a developer's toolkit, providing proven solutions to common problems in software design. This section presents a comprehensive reference chart summarizing key design patterns in JavaScript and TypeScript, focusing on their intent, applicability, and consequences. This chart serves as a quick reference guide to help developers select the most appropriate pattern for their specific needs.

### Understanding Design Patterns

Design patterns are categorized into three main types:

1. **Creational Patterns:** Deal with object creation mechanisms, trying to create objects in a manner suitable to the situation.
2. **Structural Patterns:** Concerned with object composition or the structure of classes.
3. **Behavioral Patterns:** Focus on communication between objects.

### Comparative Chart of Design Patterns

Below is a comprehensive chart summarizing various design patterns, their intent, applicability, and consequences. This chart is designed to be a quick reference for developers working with JavaScript and TypeScript.

| Pattern Name       | Category     | Intent                                                                 | Applicability                                                                 | Consequences                                                                 |
|--------------------|--------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Singleton          | Creational   | Ensure a class has only one instance and provide a global point of access. | Use when exactly one instance of a class is needed.                           | Controlled access to sole instance, but can introduce global state issues.    |
| Factory Method     | Creational   | Define an interface for creating an object, but let subclasses alter the type of objects that will be created. | Use when a class can't anticipate the class of objects it must create.        | Promotes loose coupling, but can complicate the code with many subclasses.    |
| Abstract Factory   | Creational   | Provide an interface for creating families of related or dependent objects without specifying their concrete classes. | Use when the system needs to be independent of how its products are created.  | Enhances consistency among products, but increases complexity with many interfaces. |
| Builder            | Creational   | Separate the construction of a complex object from its representation. | Use when the construction process must allow different representations.       | Allows step-by-step construction, but can lead to a large number of classes.  |
| Prototype          | Creational   | Specify the kinds of objects to create using a prototypical instance. | Use when a system should be independent of how its products are created.      | Reduces need for subclasses, but can be difficult to implement with complex objects. |
| Adapter            | Structural   | Convert the interface of a class into another interface clients expect. | Use when you want to use an existing class, and its interface does not match the one you need. | Increases class compatibility, but can introduce additional complexity.       |
| Composite          | Structural   | Compose objects into tree structures to represent part-whole hierarchies. | Use when you want to represent part-whole hierarchies of objects.             | Simplifies client code, but can make the system overly general.               |
| Decorator          | Structural   | Attach additional responsibilities to an object dynamically. | Use when you need to add responsibilities to individual objects without affecting others. | Provides flexible alternatives to subclassing, but can lead to many small objects. |
| Facade             | Structural   | Provide a unified interface to a set of interfaces in a subsystem. | Use when you want to provide a simple interface to a complex subsystem.       | Simplifies usage of complex systems, but can hide important details.          |
| Flyweight          | Structural   | Use sharing to support large numbers of fine-grained objects efficiently. | Use when many objects must be manipulated and storage costs are high.         | Reduces memory usage, but can complicate the code with shared state management. |
| Proxy              | Structural   | Provide a surrogate or placeholder for another object to control access to it. | Use when you need a more versatile or sophisticated reference to an object.   | Controls access and reduces complexity, but can introduce additional layers.  |
| Chain of Responsibility | Behavioral | Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. | Use when more than one object can handle a request and the handler isn't known a priori. | Reduces coupling, but can lead to unhandled requests.                         |
| Command            | Behavioral   | Encapsulate a request as an object, thereby allowing for parameterization of clients with queues, requests, and operations. | Use when you need to parameterize objects with operations.                    | Decouples sender and receiver, but can increase the number of classes.        |
| Interpreter        | Behavioral   | Define a representation for a language's grammar along with an interpreter that uses the representation to interpret sentences in the language. | Use when you have a language to interpret and you can represent statements as abstract syntax trees. | Easy to change and extend the grammar, but can be inefficient for complex grammars. |
| Iterator           | Behavioral   | Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation. | Use when you need to traverse a collection without exposing its representation. | Simplifies traversal, but can complicate the iterator's implementation.       |
| Mediator           | Behavioral   | Define an object that encapsulates how a set of objects interact. | Use when you want to reduce the complexity of communication between multiple objects. | Reduces dependencies, but can centralize too much control.                    |
| Memento            | Behavioral   | Without violating encapsulation, capture and externalize an object's internal state so that the object can be restored to this state later. | Use when you need to save and restore an object's state.                      | Provides state restoration, but can increase memory usage.                    |
| Observer           | Behavioral   | Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically. | Use when an object change should trigger updates to other objects.            | Promotes loose coupling, but can lead to unexpected updates.                  |
| State              | Behavioral   | Allow an object to alter its behavior when its internal state changes. | Use when an object's behavior depends on its state and it must change behavior at runtime. | Localizes state-specific behavior, but can lead to many state classes.        |
| Strategy           | Behavioral   | Define a family of algorithms, encapsulate each one, and make them interchangeable. | Use when you need to use different variants of an algorithm.                  | Promotes flexibility, but can increase the number of objects.                 |
| Template Method    | Behavioral   | Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. | Use when you want to let subclasses redefine certain steps of an algorithm.   | Promotes code reuse, but can lead to a rigid class hierarchy.                 |
| Visitor            | Behavioral   | Represent an operation to be performed on the elements of an object structure. | Use when you need to perform operations on elements of a complex object structure. | Adds new operations easily, but can complicate adding new element classes.    |

### Implementation Steps

#### Identify Key Attributes

To create a meaningful comparison, we have identified key attributes for each design pattern:

- **Category:** The type of pattern (Creational, Structural, Behavioral).
- **Intent:** The primary purpose of the pattern.
- **Applicability:** Situations where the pattern is most useful.
- **Consequences:** The results of applying the pattern, including benefits and potential drawbacks.

#### Create the Chart

The chart above is structured to clearly present each pattern's attributes, making it easy to compare and contrast different patterns.

### Use Cases

The reference chart is particularly useful in the following scenarios:

- **Quick Reference:** Developers can quickly identify which pattern might be suitable for a given problem.
- **Design Decisions:** Teams can use the chart to discuss and decide on the best pattern for their project requirements.
- **Educational Tool:** New developers can use the chart to familiarize themselves with common design patterns and their applications.

### Practice

To effectively use this chart, consider the following practice:

- **Problem Identification:** Clearly define the problem you are trying to solve.
- **Pattern Matching:** Use the chart to identify patterns that match the problem's intent and applicability.
- **Evaluate Consequences:** Consider the consequences of implementing each pattern to ensure it aligns with your project's goals.

## Quiz Time!

{{< quizdown >}}

### Which category does the Singleton pattern belong to?

- [x] Creational
- [ ] Structural
- [ ] Behavioral
- [ ] None of the above

> **Explanation:** The Singleton pattern is a Creational pattern because it deals with object creation.

### What is the primary intent of the Factory Method pattern?

- [x] Define an interface for creating an object, but let subclasses alter the type of objects that will be created.
- [ ] Provide a way to access the elements of an aggregate object sequentially.
- [ ] Convert the interface of a class into another interface clients expect.
- [ ] Attach additional responsibilities to an object dynamically.

> **Explanation:** The Factory Method pattern's intent is to define an interface for creating objects, allowing subclasses to decide which class to instantiate.

### In which scenario is the Adapter pattern most applicable?

- [x] When you want to use an existing class, and its interface does not match the one you need.
- [ ] When you need to traverse a collection without exposing its representation.
- [ ] When you need to save and restore an object's state.
- [ ] When you need to perform operations on elements of a complex object structure.

> **Explanation:** The Adapter pattern is used to convert the interface of a class into another interface clients expect, making it applicable when existing class interfaces do not match.

### What is a consequence of using the Composite pattern?

- [x] Simplifies client code, but can make the system overly general.
- [ ] Provides state restoration, but can increase memory usage.
- [ ] Reduces dependencies, but can centralize too much control.
- [ ] Promotes code reuse, but can lead to a rigid class hierarchy.

> **Explanation:** The Composite pattern simplifies client code by allowing individual objects and compositions to be treated uniformly, but it can make the system overly general.

### Which pattern is best suited for implementing undo functionality?

- [x] Memento
- [ ] Observer
- [ ] Strategy
- [ ] Visitor

> **Explanation:** The Memento pattern is used to capture and externalize an object's internal state, making it suitable for implementing undo functionality.

### What is the main advantage of the Strategy pattern?

- [x] Promotes flexibility by allowing the selection of algorithms at runtime.
- [ ] Reduces memory usage by sharing objects.
- [ ] Provides a simple interface to a complex subsystem.
- [ ] Allows an object to alter its behavior when its internal state changes.

> **Explanation:** The Strategy pattern promotes flexibility by allowing different algorithms to be selected and used at runtime.

### Which pattern is used to provide a unified interface to a set of interfaces in a subsystem?

- [x] Facade
- [ ] Proxy
- [ ] Decorator
- [ ] Chain of Responsibility

> **Explanation:** The Facade pattern provides a unified interface to a set of interfaces in a subsystem, simplifying its usage.

### What is a potential drawback of the Observer pattern?

- [x] Can lead to unexpected updates due to loose coupling.
- [ ] Can complicate the code with shared state management.
- [ ] Can introduce global state issues.
- [ ] Can centralize too much control.

> **Explanation:** The Observer pattern can lead to unexpected updates because it promotes loose coupling between the subject and observers.

### Which pattern allows an object to alter its behavior when its internal state changes?

- [x] State
- [ ] Command
- [ ] Template Method
- [ ] Interpreter

> **Explanation:** The State pattern allows an object to change its behavior when its internal state changes.

### True or False: The Visitor pattern is used to define a one-to-many dependency between objects.

- [ ] True
- [x] False

> **Explanation:** The Observer pattern, not the Visitor pattern, is used to define a one-to-many dependency between objects.

{{< /quizdown >}}
