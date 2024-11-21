---
linkTitle: "Writer Monad"
title: "Writer Monad: Collecting Logs and Other Outputs During Computations"
description: "An in-depth exploration of the Writer Monad, a functional programming pattern used to collect logs or other side outputs alongside computations, allowing for a clean separation of concerns and maintaining referential transparency."
categories:
- Functional Programming
- Design Patterns
tags:
- Writer Monad
- Monads
- Functional Programming
- Logging
- Referential Transparency
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/effect-handling-patterns/logging-and-metrics/writer-monad"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In the realm of functional programming, side effects are often handled in a manner that allows for referential transparency and pure function execution. The **Writer Monad** is a powerful design pattern that enables functional programs to produce auxiliary outputs such as logs, without compromising the purity of the functions. This article explores the concept, implementation, and use-cases of the Writer Monad in detail.

## Concept

The Writer Monad encapsulates a computation alongside an auxiliary output value, typically used for logging or accumulating intermediate results. The primary characteristic of the Writer Monad is that it maintains separation between the computational logic and the side outputs, facilitating clean, maintainable code.

In formal terms, a Writer Monad can be thought of as a combination of a value and a log. For any type `A`, the type `Writer[A]` models a computation that results in a value of type `A` and an additional output (often a log message).

## Formal Definition

The type definition of a Writer Monad can be represented as follows:

```haskell
newtype Writer w a = Writer { runWriter :: (a, w) }
```

Here, the type `w` represents the log (or writer) and `a` represents the actual computation result.

### Monad Implementation

To implement the Writer Monad, we need to define the following operations:

1. **unit (or return)**:

The `unit` operation takes a value and yields a Writer Monad wrapping that value with an initial, empty log.

```haskell
unit :: (Monoid w) => a -> Writer w a
unit x = Writer (x, mempty)
```

2. **bind (or >>=)**:

The `bind` operation sequences two computations, combining their logs.

```haskell
(>>=) :: (Monoid w) => Writer w a -> (a -> Writer w b) -> Writer w b
Writer (x, log1) >>= f =
  let Writer (y, log2) = f x
  in Writer (y, log1 `mappend` log2)
```

The `mappend` function is used to combine the logs, leveraging the Monoid properties.

## Usage Examples

### Logging Computations

Consider a simple example where you want to compute the sum of a list of numbers while logging each addition operation:

```haskell
import Control.Monad.Writer

sumWithLog :: [Int] -> Writer [String] Int
sumWithLog [] = return 0
sumWithLog (x:xs) = do
  result <- sumWithLog xs
  tell ["Adding " ++ show x ++ " to " ++ show result]
  return (x + result)

main = do
  let (result, log) = runWriter (sumWithLog [1, 2, 3])
  mapM_ putStrLn log
  print result
```

In this example, `tell` is used to append messages to the log.

### Collecting Outputs

The Writer Monad can also be used to collect other outputs, such as tracking intermediate steps in a calculation:

```haskell
import Control.Monad.Writer

factorial :: Int -> Writer [String] Int
factorial 0 = do
  tell ["Start with 1"]
  return 1
factorial n = do
  prev <- factorial (n - 1)
  let result = n * prev
  tell [show n ++ " * " ++ show prev ++ " = " ++ show result]
  return result

main = do
  let (result, log) = runWriter (factorial 5)
  mapM_ putStrLn log
  print result
```

## Related Design Patterns

### Reader Monad

The Reader Monad provides a way to pass read-only environment information through a computation. It contrasts with the Writer Monad, which instead accumulates output information. Both share the goal of maintaining functional purity by handling auxiliary information in a structured manner.

### State Monad

The State Monad encapsulates state manipulations within a computation, providing a framework to thread state transformations through function calls. Like the Writer Monad, it manages side-effects but focuses on state rather than logging.

### List Monad

While not directly related, the List Monad models nondeterministic computations, where a single input might map to multiple outputs. Its structure shares similarities with Writer in terms of handling computations that have additional outcomes or considerations beyond single values.

## Additional Resources

1. [*Learn You a Haskell for Great Good!*](http://learnyouahaskell.com/)
2. [*Real World Haskell*](http://book.realworldhaskell.org/)
3. [Haskell Documentation](https://www.haskell.org/documentation/)
4. [Functional Programming in Scala](https://www.manning.com/books/functional-programming-in-scala)

## Summary

The Writer Monad is an invaluable tool for functional programmers, allowing seamless integration of side outputs like logs into pure functions. By leveraging the monadic structure, it maintains separation between core calculations and auxiliary actions, enabling cleaner and more maintainable code. Understanding and using the Writer Monad empowers developers to handle complex tasks in a transparent and functional way.
