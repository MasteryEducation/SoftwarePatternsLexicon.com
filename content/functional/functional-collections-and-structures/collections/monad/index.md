---
linkTitle: "Monad"
title: "Monad: A Structure that Supports Chaining Operations and Managing Side Effects"
description: "Detailed overview of Monads in functional programming, their structure, usage, and related patterns."
categories:
- Functional Programming
- Design Patterns
tags:
- Monad
- Functional Programming
- Higher-Order Functions
- Side Effects
- Chaining
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/functional-collections-and-structures/collections/monad"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In functional programming, a **Monad** is a powerful design pattern used to encapsulate and handle computational effects such as state, IO, exceptions, and others, in a purely functional way. A Monad is a type that implements specific combinatorial operations which enable the chaining of computations.

### Monad Basics

A Monad must satisfy the following three laws:

1. **Left Identity:** `return a >>= k` is the same as `k a`
2. **Right Identity:** `m >>= return` is the same as `m`
3. **Associativity:** `(m >>= k) >>= h` is the same as `m >>= (\x -> k x >>= h)`

In Haskell notation, the standard Monad type class is defined as:

```haskell
class Monad m where
    (>>=) :: m a -> (a -> m b) -> m b
    return :: a -> m a
```

Where:

- `>>=` (bind) is the operation used to chain computations.
- `return` (or unit) injects a value into the monadic context.

## Monad Instances

Common Monad instances include:

### **Maybe Monad**

The `Maybe` Monad represents computations which might fail. It captures the notion of failure through two constructors: `Just` for success and `Nothing` for failure.

```haskell
instance Monad Maybe where
    (Just x) >>= k = k x
    Nothing  >>= _ = Nothing
    return = Just
```

### **List Monad**

The List Monad represents non-deterministic computations, where each computation might have multiple possible results.

```haskell
instance Monad [] where
    xs >>= k = concat (map k xs)
    return x = [x]
```

### **Either Monad**

The `Either` Monad represents computations which might fail with an error. It uses `Left` for an error and `Right` for a successful computation.

```haskell
instance Monad (Either e) where
    (Right x) >>= k = k x
    (Left e)  >>= _ = Left e
    return = Right
```

## Applying Monads

The power of Monads comes from their ability to sequence operations while abstracting away the handling of side effects and computational contexts.

### Example: Safe Division with Maybe Monad

Here's an example of a safe division operation using the `Maybe` Monad in Haskell:

```haskell
safeDiv :: Double -> Double -> Maybe Double
safeDiv _ 0 = Nothing
safeDiv x y = Just (x / y)

compute :: Double -> Double -> Double -> Maybe Double
compute x y z = do
    a <- safeDiv x y
    safeDiv a z
```

In this example, `compute` sequences two division operations while safely handling the potential of division by zero.

```sequenceDiagram
participant Client
participant Division
Client->>Division: call safeDiv(x, y)
alt y == 0
    Division->>Client: return Nothing
else
    Division->>Client: return Just(x / y)
end
Client->>Division: bind result and call safeDiv(a, z)
alt z == 0
    Division->>Client: return Nothing
else
    Division->>Client: return Just(a / z)
end
```

## Related Design Patterns

### **Functor**

A Functor is a design pattern that models computational contexts that can be mapped over. All Monads are Functors, as they support the `fmap` operation for mapping a function over the encapsulated value.

```haskell
class Functor f where
    fmap :: (a -> b) -> f a -> f b
```

### **Applicative**

An Applicative is an intermediate structure between Functor and Monad. It allows for function application lifted over computational contexts.

```haskell
class Functor f => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b
```

## Summary

Monads are a fundamental design pattern in functional programming, enabling the encapsulation of side effects and the composable chaining of operations. By adhering to specific laws and leveraging the power of algebraic structures, Monads provide a robust framework for handling complex computation flows in a clean and elegant manner.

## Additional Resources

- [Learn You a Haskell for Greater Good!](http://learnyouahaskell.com/)
- [Haskell Programming from First Principles](http://haskellbook.com/)
- [Real World Haskell](http://book.realworldhaskell.org/)
- [Functional Programming in Scala](https://www.manning.com/books/functional-programming-in-scala)

