---
linkTitle: "Writer Monad"
title: "Writer Monad: Accumulating Logging Information Alongside Computational Results"
description: "The Writer Monad is a design pattern that allows accumulating log information alongside computational results in functional programming. It provides a way to capture additional context such as logging, debugging, or profiling without the need to pass extra parameters through the entire computation."
categories:
- Functional Programming
- Design Patterns
tags:
- Writer Monad
- Functional Programming
- Logging
- Monads
- Side Effects
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/effect-handling-patterns/side-effects/writer-monad"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


The **Writer Monad** is a powerful design pattern in the realm of functional programming, aimed at accumulating additional context information—such as logs—alongside computational results. This pattern elegantly addresses the need to carry forward supplemental data without explicitly threading it through every function call, thereby maintaining clean and readable code.

## Introduction

### What is a Monad?

Before diving into the specifics of the Writer Monad, it is essential to understand what a monad is in functional programming:

Monads are abstract data types that encapsulate values along with a context of computation. They are used to chain operations together in a flexible yet structured manner. The primary components of a monad are:

1. **Bind (`>>=`)**: Chains operations and passes the monadic value to the next function.
2. **Return (`unit` or `pure`)**: Wraps a regular value into a monad.

### The Essence of the Writer Monad

The Writer Monad can be thought of as a pair comprising of:

1. **Value**: The result of a computation.
2. **Log**: The accumulated log or context that accompanies the computation.

Formally, it adheres to the structure `Writer (a, w)` where `a` represents the result and `w` represents the log. It enables values to carry along auxiliary information without modifying the functions operating on these values.

## Definition and Syntax

In a hypothetical functional programming language, the Writer Monad could be defined as follows:

```haskell
data Writer w a = Writer { runWriter :: (a, w) }

instance (Monoid w) => Monad (Writer w) where
    return x = Writer (x, mempty)
    (Writer (x, log)) >>= f = let (y, newLog) = runWriter (f x)
                               in Writer (y, log `mappend` newLog)
```

Here, `w` must be a Monoid to combine logs. The `mempty` provides the identity element, and `mappend` is used to concatenate logs.

## Key Operations

1. **Return (`unit` or `pure`)**:
   ```haskell
   return x = Writer (x, mempty)
   ```
   It lifts a value into the Writer Monad with an empty log.

2. **Bind (`>>=`)**:
   ```haskell
   (Writer (x, log)) >>= f = let (y, newLog) = runWriter (f x)
                              in Writer (y, log `mappend` newLog)
   ```
   It applies a function to the value within the monad, concatenating the logs.

3. **Tell**:
   ```haskell
   tell :: w -> Writer w ()
   tell log = Writer ((), log)
   ```
   It appends log information to the current computation.

## Example

Consider a scenario where we perform computations alongside logging:

```haskell
import Control.Monad.Writer

addWithLog :: Int -> Int -> Writer [String] Int
addWithLog x y = do
    tell ["Adding " ++ show x ++ " and " ++ show y]
    return (x + y)

multiplyWithLog :: Int -> Int -> Writer [String] Int
multiplyWithLog x y = do
    tell ["Multiplying " ++ show x ++ " and " ++ show y]
    return (x * y)

computation :: Writer [String] Int
computation = do
    sumResult <- addWithLog 3 5
    productResult <- multiplyWithLog sumResult 2
    return productResult
    
main :: IO ()
main = do
    let (result, log) = runWriter computation
    putStrLn $ "Result: " ++ show result
    putStrLn "Log: "
    mapM_ putStrLn log

-- Output:
-- Result: 16
-- Log:
-- Adding 3 and 5
-- Multiplying 8 and 2
```

In this example, the `computation` function carries out arithmetic operations while logging every step.

## Related Design Patterns

### State Monad

The **State Monad** carries state along with computations, which can be mutated purely. Both Writer and State Monads handle additional context but differ in their use cases—State Monad deals with mutable state, while Writer Monad focuses on accumulating context like logs.

### Reader Monad

The **Reader Monad** deals with reading shared configuration or environment. Unlike the Writer Monad, it does not accumulate context but provides a mechanism to access shared read-only data across computations.

## Additional Resources

For further exploration of the Writer Monad and related idioms in functional programming, consult the following resources:

1. **"Learn You a Haskell for Great Good!"** by Miran Lipovača
2. **"Haskell Programming from First Principles"** by Christopher Allen and Julie Moronuki
3. **Haskell Documentation** - [Hackage: Control.Monad.Writer](https://hackage.haskell.org/package/mtl-2.2.1/docs/Control-Monad-Writer.html)

## Summary

The Writer Monad provides an elegant solution for carrying log information alongside computational results in functional programming. By avoiding the need to thread extra parameters through every function call, it maintains cleaner and more maintainable code. Understanding the Writer Monad, along with its related State and Reader Monads, allows for more sophisticated functional design patterns and conveys a tangible benefit in dealing with side effects in a pure functional paradigm.
