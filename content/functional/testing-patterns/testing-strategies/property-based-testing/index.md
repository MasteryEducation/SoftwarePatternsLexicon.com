---
linkTitle: "Property-Based Testing"
title: "Property-Based Testing: Automatic Test Case Generation"
description: "Property-Based Testing is an advanced functional programming technique that generates test cases based on invariants or properties that should always be true for the software being tested."
categories:
- Functional Programming
- Testing
tags:
- Property-Based Testing
- Functional Programming
- Software Quality
- Invariants
- Automated Testing
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/testing-patterns/testing-strategies/property-based-testing"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

Property-Based Testing (PBT) is an automated testing technique primarily used in functional programming to improve software quality and reliability. Unlike traditional example-based testing (sometimes referred to as "unit testing"), which verifies the correctness of individual examples, PBT involves specifying properties that describe the expected behavior of the system and automatically generating test cases based on these properties.

## Core Principles of Property-Based Testing

### Properties

Properties are assertions about the behavior of a function or system that must always hold true. They describe general attributes of your system or the relationships between the inputs and outputs of your functions. Some common examples include:
- Commutativity: `f(a, b) === f(b, a)`
- Associativity: `f(a, f(b, c)) === f(f(a, b), c)`
- Idempotency: `f(f(a)) === f(a)`

### Random Test Case Generation

One of the distinguishing features of PBT is its ability to automatically generate numerous random test cases. This helps in exploring a wide range of input scenarios, including edge cases that manual example-based tests might miss.

### Shrinking

When a test case fails, PBT libraries often attempt to minimize (or "shrink") the failing input to its simplest form that still reproduces the failure. This process helps in debugging by isolating the simplest test case that triggers the bug.

## Tools for Property-Based Testing

Several libraries are available for implementing PBT in various programming languages:

- **Haskell**: QuickCheck
- **Scala**: ScalaCheck
- **JavaScript/TypeScript**: fast-check
- **Python**: Hypothesis
- **Java**: jqwik

### Example: QuickCheck in Haskell

Here is an example of how QuickCheck can be used to test a simple property in Haskell:

```haskell
import Test.QuickCheck

prop_reverseTwice :: [Int] -> Bool
prop_reverseTwice xs = reverse (reverse xs) == xs

main :: IO ()
main = quickCheck prop_reverseTwice
```

In this example, `prop_reverseTwice` is a property that states that reversing a list twice should yield the original list. QuickCheck will automatically generate random lists of integers (`[Int]`) and test the property.

## Related Design Patterns

### Contract-Based Programming

Contract-Based Programming involves specifying preconditions, postconditions, and invariants in the form of contracts. While property-based testing automatically generates test cases, contract-based programming often relies on manually written contracts that describe the expected behavior of functions.

### Model-Based Testing

Model-Based Testing involves creating models that describe the desired behavior of a system. These models can then be used to generate test cases. This approach is similar to PBT but focuses more on high-level system behaviors rather than individual properties.

## Additional Resources

- [QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs](https://dl.acm.org/doi/10.1145/266635.266643)
- [Property-Based Testing with PropEr, Erlang, and Elixir (Book)](http://propertesting.com/)
- [Fast-check Documentation](https://dubzzz.github.io/fast-check.github.com/)

## Summary

Property-Based Testing is a powerful technique for ensuring software reliability by automatically generating and running test cases based on defined properties. It extends beyond traditional example-based testing by exploring a broad spectrum of inputs, including unforeseen edge cases, thereby enhancing software robustness. Tools like QuickCheck, ScalaCheck, and Hypothesis facilitate easy adoption of PBT across various programming languages, making it a valuable asset for developers focused on maintaining high-quality codebases.
