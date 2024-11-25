---
linkTitle: "Either"
title: "Either: Encapsulating alternatives, often for error handling"
description: "The Either type encapsulates a value of one of two possible types, often used for error handling in functional programming."
categories:
- Functional Programming
- Design Patterns
tags:
- Either
- Error Handling
- Functional Programming
- Design Patterns
- Encapsulation
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/miscellaneous-patterns/error-handling/either"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

In functional programming, managing and expressing different outcomes of computations, such as success or failure, is a fundamental concern. The `Either` type is a powerful design pattern that encapsulates a value of one of two possible types. Typically, it is used to represent computations that can result in one of two states: success (often known as `Right`) and failure (often known as `Left`). This pattern ensures that errors are handled explicitly within the type system, encouraging robust and maintainable code.

## Understanding the Either Type

The `Either` type in functional programming languages can be defined as:

```haskell
data Either a b = Left a | Right b
```

Here, `Left a` usually represents an error or exceptional condition, while `Right b` represents a successful result.

### Basic Usage

Consider a scenario where we need to perform division but handle the potential for division by zero gracefully:

```haskell
safeDivide :: Double -> Double -> Either String Double
safeDivide _ 0 = Left "Division by zero error"
safeDivide x y = Right (x / y)
```

In the above code:
- `Left "Division by zero error"` is used to return an error message for the division by zero case.
- `Right (x / y)` returns the result of the division operation when the denominator is not zero.

### Working with Either

To work with values of type `Either`, pattern matching is commonly used in functional languages like Haskell:

```haskell
handleResult :: Either String Double -> String
handleResult (Left errMsg) = "Error: " ++ errMsg
handleResult (Right value) = "Result: " ++ show value
```

In more feature-rich functional programming environments, you often find combinators and higher-order functions that allow working with `Either` in a more functional and composable way without explicitly pattern matching on values.

```haskell
import Control.Monad (liftM, liftM2)

example :: Either String Double
example = safeDivide 10 0 >>= return . (* 2)

combinedResults :: Either String Double
combinedResults = liftM2 (+) (safeDivide 10 2) (safeDivide 8 2)
```

## Related Design Patterns

### Maybe (Option)
The `Maybe` or `Option` type is similar to `Either`, but it encapsulates an optional value, avoiding null references. `Option` is typically used when you have a computation that may not yield a result but does not require detailed error information. It has two variants: `Some(value)` and `None`.

#### Comparison:
- `Either` is useful when you want to distinguish between two types of outcomes explicitly (like success and failure).
- `Maybe` is useful when you only care whether or not there is a result.

### Result Type
The `Result` type is a variation of `Either` used in some functional languages, like Rust, where it directly represents success and error cases. The `Result` type in Rust is defined as:

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

### Try Monad
The `Try` monad, often found in Scala, is another pattern for handling computations that might fail. It is conceptually similar to `Either`, but explicitly models the failure case as an exception.

## Additional Resources

- **Books**:
  - *Functional Programming in Scala* by Paul Chiusano and Rúnar Bjarnason
  - *Haskell Programming from First Principles* by Christopher Allen and Julie Moronuki

- **Articles and Tutorials**:
  - [Learn You a Haskell for Great Good!](http://learnyouahaskell.com/): A beginner's guide to Haskell.
  - [Functional Programming in Java with Vavr](https://www.baeldung.com/vavr-either)
  
- **Online Courses**:
  - *Functional Programming Principles in Scala* by Martin Odersky on Coursera
  - *Functional Programming in Haskell* on Udacity

## Summary

The `Either` type is a fundamental concept in functional programming for handling computations that can fail or succeed. By encapsulating possible outcomes in a single algebraic data type, `Either` ensures that error handling is explicit and composable. Understanding and utilizing `Either` can lead to more robust and maintainable code, where failure cases are handled effectively within the type system.

Adopting related design patterns, such as `Maybe`, `Result`, or the `Try` monad, further allows for expressive and type-safe error handling in functional programming. These patterns help developers write safer and more predictable code by making invalid states unrepresentable.
