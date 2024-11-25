---
linkTitle: "2.3.9 Strategy"
title: "Strategy Design Pattern in JavaScript and TypeScript: A Comprehensive Guide"
description: "Explore the Strategy Design Pattern in JavaScript and TypeScript, its intent, key components, implementation steps, and practical use cases."
categories:
- Design Patterns
- JavaScript
- TypeScript
tags:
- Strategy Pattern
- Behavioral Patterns
- JavaScript Design Patterns
- TypeScript Design Patterns
- Software Architecture
date: 2024-10-25
type: docs
nav_weight: 239000
canonical: "https://softwarepatternslexicon.com/patterns-js/2/3/9"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.3.9 Strategy

### Introduction

The Strategy Design Pattern is one of the classic Gang of Four (GoF) behavioral patterns. It is used to define a family of algorithms, encapsulate each one, and make them interchangeable. This pattern allows the algorithm to vary independently from clients that use it, promoting flexibility and reusability in code.

### Understand the Intent

The primary intent of the Strategy Pattern is to enable the selection of an algorithm at runtime. By encapsulating algorithms within classes, the Strategy Pattern allows clients to choose different algorithms without modifying the context in which they operate.

### Key Components

- **Strategy Interface:** Defines the interface for the algorithm.
- **Concrete Strategies:** Implement the algorithm using the strategy interface.
- **Context:** Maintains a reference to a strategy object and is configured with a concrete strategy. The context delegates the algorithm to the strategy object.

### Implementation Steps

1. **Define the Strategy Interface:** Create an interface that outlines the algorithm's structure.
2. **Implement Concrete Strategies:** Develop classes that implement the strategy interface, each providing a different algorithm.
3. **Modify the Context:** Adjust the context to use a strategy instance, allowing clients to pass a concrete strategy to the context.

### Code Examples

Let's explore the Strategy Pattern through a practical example using sorting algorithms. We'll demonstrate how different sorting strategies can be applied to a list of numbers.

#### JavaScript Example

```javascript
// Strategy Interface
class SortStrategy {
  sort(data) {
    throw new Error("This method should be overridden!");
  }
}

// Concrete Strategy: Bubble Sort
class BubbleSortStrategy extends SortStrategy {
  sort(data) {
    console.log("Sorting using bubble sort");
    // Implement bubble sort algorithm
    return data.sort((a, b) => a - b);
  }
}

// Concrete Strategy: Quick Sort
class QuickSortStrategy extends SortStrategy {
  sort(data) {
    console.log("Sorting using quick sort");
    // Implement quick sort algorithm
    return data.sort((a, b) => a - b); // Simplified for demonstration
  }
}

// Context
class SortContext {
  constructor(strategy) {
    this.strategy = strategy;
  }

  setStrategy(strategy) {
    this.strategy = strategy;
  }

  executeStrategy(data) {
    return this.strategy.sort(data);
  }
}

// Client code
const data = [5, 2, 9, 1, 5, 6];
const context = new SortContext(new BubbleSortStrategy());
console.log(context.executeStrategy(data));

context.setStrategy(new QuickSortStrategy());
console.log(context.executeStrategy(data));
```

#### TypeScript Example

```typescript
// Strategy Interface
interface SortStrategy {
  sort(data: number[]): number[];
}

// Concrete Strategy: Bubble Sort
class BubbleSortStrategy implements SortStrategy {
  sort(data: number[]): number[] {
    console.log("Sorting using bubble sort");
    // Implement bubble sort algorithm
    return data.sort((a, b) => a - b);
  }
}

// Concrete Strategy: Quick Sort
class QuickSortStrategy implements SortStrategy {
  sort(data: number[]): number[] {
    console.log("Sorting using quick sort");
    // Implement quick sort algorithm
    return data.sort((a, b) => a - b); // Simplified for demonstration
  }
}

// Context
class SortContext {
  private strategy: SortStrategy;

  constructor(strategy: SortStrategy) {
    this.strategy = strategy;
  }

  setStrategy(strategy: SortStrategy) {
    this.strategy = strategy;
  }

  executeStrategy(data: number[]): number[] {
    return this.strategy.sort(data);
  }
}

// Client code
const data: number[] = [5, 2, 9, 1, 5, 6];
const context = new SortContext(new BubbleSortStrategy());
console.log(context.executeStrategy(data));

context.setStrategy(new QuickSortStrategy());
console.log(context.executeStrategy(data));
```

### Use Cases

- **Sorting Algorithms:** As demonstrated, the Strategy Pattern is ideal for implementing different sorting algorithms.
- **Payment Methods in E-commerce:** Different payment strategies (e.g., credit card, PayPal, cryptocurrency) can be encapsulated and selected at runtime.
- **Data Compression:** Implement different compression strategies (e.g., ZIP, GZIP) to compress data.

### Practice

Try implementing a context for data compression with strategies for different compression algorithms. This exercise will help solidify your understanding of the Strategy Pattern.

### Considerations

- **Flexibility:** The Strategy Pattern enables swapping algorithms without changing the context code, promoting flexibility.
- **Client Awareness:** Clients must be aware of different strategies to select the appropriate one, which can increase complexity.

### Advantages and Disadvantages

#### Advantages

- **Interchangeability:** Algorithms can be changed without modifying the context.
- **Open/Closed Principle:** New strategies can be added without altering existing code.
- **Code Reusability:** Common algorithm interfaces promote code reuse.

#### Disadvantages

- **Increased Complexity:** More classes and interfaces can lead to increased complexity.
- **Client Knowledge:** Clients need to understand the available strategies to choose the correct one.

### Best Practices

- **Encapsulation:** Ensure each strategy encapsulates a single algorithm.
- **Interface Design:** Design a clear and concise strategy interface.
- **Strategy Selection:** Provide a mechanism for clients to select strategies easily.

### Comparisons

The Strategy Pattern is often compared with the State Pattern. While both patterns involve changing behavior at runtime, the Strategy Pattern focuses on interchangeable algorithms, whereas the State Pattern deals with state-dependent behavior.

### Conclusion

The Strategy Design Pattern is a powerful tool for managing interchangeable algorithms in JavaScript and TypeScript applications. By encapsulating algorithms within classes, developers can create flexible and reusable code that adheres to the Open/Closed Principle.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Strategy Design Pattern?

- [x] To define a family of algorithms, encapsulate each one, and make them interchangeable.
- [ ] To manage object creation and lifecycle.
- [ ] To provide a way to access the elements of an aggregate object sequentially.
- [ ] To ensure a class has only one instance and provide a global point of access to it.

> **Explanation:** The Strategy Pattern's primary intent is to define a family of algorithms, encapsulate each one, and make them interchangeable.

### Which component of the Strategy Pattern defines the algorithm interface?

- [x] Strategy Interface
- [ ] Concrete Strategies
- [ ] Context
- [ ] Client

> **Explanation:** The Strategy Interface defines the algorithm interface in the Strategy Pattern.

### In the Strategy Pattern, what role does the Context play?

- [x] It uses a strategy object and is configured with a concrete strategy.
- [ ] It defines the algorithm interface.
- [ ] It implements the algorithm using the strategy interface.
- [ ] It selects the appropriate strategy for the client.

> **Explanation:** The Context uses a strategy object and is configured with a concrete strategy, delegating the algorithm to the strategy object.

### What is a practical use case for the Strategy Pattern?

- [x] Implementing different payment methods in an e-commerce system.
- [ ] Ensuring a class has only one instance.
- [ ] Providing a way to access elements of an aggregate object sequentially.
- [ ] Managing object creation and lifecycle.

> **Explanation:** A practical use case for the Strategy Pattern is implementing different payment methods in an e-commerce system.

### What is a disadvantage of the Strategy Pattern?

- [x] Increased complexity due to more classes and interfaces.
- [ ] Difficulty in adding new strategies.
- [ ] Lack of flexibility in swapping algorithms.
- [ ] Inability to adhere to the Open/Closed Principle.

> **Explanation:** A disadvantage of the Strategy Pattern is increased complexity due to more classes and interfaces.

### How does the Strategy Pattern adhere to the Open/Closed Principle?

- [x] By allowing new strategies to be added without altering existing code.
- [ ] By ensuring a class has only one instance.
- [ ] By providing a way to access elements of an aggregate object sequentially.
- [ ] By managing object creation and lifecycle.

> **Explanation:** The Strategy Pattern adheres to the Open/Closed Principle by allowing new strategies to be added without altering existing code.

### Which of the following is NOT a component of the Strategy Pattern?

- [x] Singleton
- [ ] Strategy Interface
- [ ] Concrete Strategies
- [ ] Context

> **Explanation:** Singleton is not a component of the Strategy Pattern.

### What must clients be aware of when using the Strategy Pattern?

- [x] Different strategies to select the appropriate one.
- [ ] The internal implementation of each strategy.
- [ ] The lifecycle management of strategy objects.
- [ ] The singleton nature of the context.

> **Explanation:** Clients must be aware of different strategies to select the appropriate one when using the Strategy Pattern.

### What is a key advantage of using the Strategy Pattern?

- [x] Algorithms can be changed without modifying the context.
- [ ] It ensures a class has only one instance.
- [ ] It provides a way to access elements of an aggregate object sequentially.
- [ ] It manages object creation and lifecycle.

> **Explanation:** A key advantage of the Strategy Pattern is that algorithms can be changed without modifying the context.

### True or False: The Strategy Pattern is used to manage object creation and lifecycle.

- [ ] True
- [x] False

> **Explanation:** False. The Strategy Pattern is not used to manage object creation and lifecycle; it is used to define a family of algorithms and make them interchangeable.

{{< /quizdown >}}
