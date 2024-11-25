---
linkTitle: "Chain of Operations"
title: "Chain of Operations: Functional Composition for Ordered Transformation"
description: "The Chain of Operations design pattern involves linking a series of transformations on data using function composition, enabling an elegant and declarative approach to processing data."
categories:
- Functional Programming Principles
- Design Patterns
tags:
- functional programming
- function composition
- data transformation
- immutability
- design patterns
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/function-composition-and-transformation-patterns/function-composition/chain-of-operations"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


The Chain of Operations design pattern leverages functional composition to sequence data transformations in a clean and manageable way. This pattern is foundational in functional programming and allows operations to be expressed declaratively, resulting in more readable, maintainable, and testable code.

## Core Concepts

### Functional Composition

Functional composition is the process of combining two or more functions to produce a new function. The idea is to apply functions in a pipeline, where the output of one function becomes the input of the next. Mathematically, if you have functions \\( f \\) and \\( g \\), composing them yields a new function \\( h \\) such that \\( h(x) = g(f(x)) \\).

Using function composition in code:

```javascript
const compose = (f, g) => (x) => f(g(x));

// Example functions
const double = x => x * 2;
const increment = x => x + 1;

// Composed function
const doubleThenIncrement = compose(increment, double);

console.log(doubleThenIncrement(3)); // Outputs 7
```

### Immutability

In functional programming, data structures are immutable. This means that once they are created, they cannot be changed. Instead of modifying an existing data structure, operations generate new data structures. This aligns perfectly with Chain of Operations, as each step in the transformation chain yields new, immutable data.

## Pattern Mechanics

The Chain of Operations pattern typically involves these components:

- **Initial Data:** The starting point for the chain.
- **Transformations:** A series of pure functions that each take some data and transform it in an application-specific way.
- **Composition Function:** A higher-order function that sequences the transformations.

### Example

Suppose you have a list of students and want to process the list to extract data and format it in a particular way:

```javascript
// Data
const students = [
  { name: "Alice", grades: [85, 80, 90] },
  { name: "Bob", grades: [75, 70, 65] },
  { name: "Charlie", grades: [95, 90, 95] },
];

// Transformations
const getNames = (students) => students.map(({ name }) => name);
const toUpperCase = (names) => names.map((name) => name.toUpperCase());
const joinWithComma = (names) => names.join(", ");

// Composition of transformations
const processStudents = compose(joinWithComma, toUpperCase, getNames);

console.log(processStudents(students)); // Outputs: "ALICE, BOB, CHARLIE"
```

Here, the `compose` function could be implemented as follows to accept multiple functions:

```javascript
const compose = (...funcs) => (x) =>
  funcs.reduceRight((acc, fn) => fn(acc), x);
```

## Related Design Patterns

### 1. **Pipeline Pattern**
The Pipeline pattern is closely related to Chain of Operations but emphasizes linear sequences of operations rather than composition. It's especially common in data processing pipelines.

### 2. **Strategy Pattern**
In both object-oriented and functional paradigms, the Strategy pattern involves selecting an algorithm or a method to accomplish a certain task. In functional programming, this can be achieved via function passing and composition.

### 3. **Builder Pattern**
Though typically discussed in the context of object-oriented programming, a functional equivalent can be achieved via chaining operations to gradually accumulate configuration and state before building the final, immutable object.

## Additional Resources

- **Books**:
    - *Functional Programming in JavaScript* by Luis Atencio
    - *Functional Programming in Scala* by Paul Chiusano and Runar Bjarnason
    
- **Online Articles**:
    - [Modern JS Cheatsheet](https://github.com/mbeaudru/modern-js-cheatsheet)

- **Courses**:
    - [Functional Programming Principles in Scala](https://www.coursera.org/learn/scala-functional-programming) by Martin Odersky on Coursera
    - [Functional Programming in JavaScript: How to Write Code for Scalability](https://www.pluralsight.com/courses/javascript-development-scalability-functional-programming) on Pluralsight

## Summary

The Chain of Operations design pattern provides a structured approach to process data through a series of transformations using function composition. It promotes code clarity, immutability, and an elegant flow of data processing. Understanding this pattern empowers developers to harness the full potential of functional programming concepts, leading to improved readability and maintainability of codebases.


{{< katex />}}

