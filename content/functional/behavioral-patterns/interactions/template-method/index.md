---
linkTitle: "Template Method"
title: "Template Method: Defining the Skeleton of an Algorithm"
description: "The Template Method pattern is a design pattern that defines the skeleton of an algorithm in a base class, allowing subclasses to provide specific implementations for various steps in the algorithm."
categories:
- Functional Programming
- Design Patterns
tags:
- Template Method
- Behavioral Patterns
- Functional Programming
- Higher-Order Functions
- Algorithm Design
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/behavioral-patterns/interactions/template-method"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


The Template Method pattern is a behavioral design pattern that defines the structure of an algorithm within a method, delegating some steps to subclasses. It allows the overarching algorithm to remain consistent while enabling flexibility in the implementation of individual steps.

## Core Concepts

### Definition

In object-oriented programming, the Template Method is typically realized with an abstract class that provides the template method. This method includes steps that are either fully implemented or abstract, with the latter being implemented by subclasses to complete the algorithm.

### Components

1. **Abstract Class**: Contains the template method and outlines the algorithm’s structure using concrete methods, abstract methods, or calls to other methods.
2. **Template Method**: Defines the skeleton of the algorithm, delegating specific steps to methods that may be either concrete (implemented) or abstract.
3. **Concrete SubClasses**: Provide the specific implementations for the abstract methods used by the template method.

## Functional Programming Perspective

While traditionally implemented with inheritance in OOP, the Template Method pattern can also be implemented using higher-order functions and function composition in functional programming. In FP, it offers a way to construct algorithms in a more flexible manner, leveraging first-class functions.

### Higher-Order Functions

A higher-order function is a function that takes one or more functions as arguments and/or returns a function as a result. This capability allows the definition of skeleton algorithms where specific steps can be passed as parameters.

### Function Composition

Function composition allows the combination of simple functions to build more complex ones. This aspect fits naturally with the concept of the Template Method by creating a pipeline of operations.

## Example Implementation

Here is an example of the Template Method pattern implemented in Haskell:

```haskell
-- Define the template function
templateMethod :: (a -> b) -> (b -> c) -> a -> c
templateMethod step1 step2 input =
  let result1 = step1 input
  in step2 result1

-- Concrete steps
step1 :: Int -> Double
step1 x = fromIntegral x * 1.5

step2 :: Double -> String
step2 y = "Result: " ++ show (y + 2.0)

-- Use the template function
main :: IO ()
main = do
  let result = templateMethod step1 step2 4
  putStrLn result
```

In this example, `templateMethod` outlines the structure of the algorithm. The actual steps (`step1` and `step2`) are passed as arguments to the template function.

## Related Design Patterns

- **Strategy Pattern**: Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Similar to the Template Method but defines the entire strategy independently rather than defining only parts of it.
- **Decorator Pattern**: Allows behavior to be added to an individual object, either statically or dynamically, without affecting the behavior of other objects from the same class.
- **Composite Pattern**: Composes objects into tree structures to represent part-whole hierarchies.

### Overview of Related Patterns

#### Strategy

The Strategy pattern involves creating an interface common to all supported algorithms, and making each concrete strategy implement this interface. It's useful to have a concrete encapsulation of algorithms that don't share a common interface.

#### Decorator

In contrast to Template Method, where subclasses change parts of an algorithm, Decorator pattern provides a flexible alternative to subclassing for extending functionality by wrapping objects.

#### Composite

Where Template Method defines the structure for a single step and allows extensions, Composite patterns help more with hierarchical object structures, letting individual objects and compositions be treated uniformly.

## Additional Resources

For further study and deeper understanding of the Template Method pattern and its applications in functional programming, consider the following resources:

- *"Design Patterns: Elements of Reusable Object-Oriented Software"* by Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides (The GoF book)
- "Functional Programming for the Object-Oriented Programmer" by Brian Marick
- "Haskell Design Patterns" by Ryan Lemmer
- Online course: "Functional Programming Principles in Scala" by Martin Odersky on Coursera

## Summary

The Template Method pattern in functional programming achieves the same goal as in object-oriented programming: defining the skeleton of an algorithm and deferring the implementation of certain steps to help achieve code reuse and flexibility. By leveraging higher-order functions and function composition, the pattern can be effectively utilized in any language that supports first-class functions.

With a solid understanding of the Template Method pattern and its functional approach, one can design more flexible, reusable, and maintainable code structures. This pattern blends well with other functional design patterns, offering robust solutions to common algorithmic problems.
