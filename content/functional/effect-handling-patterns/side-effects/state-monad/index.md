---
linkTitle: "State Monad"
title: "State Monad: Managing State Immutably within Computations"
description: "The State Monad helps in managing state immutably in functional programming by threading state through computations in a controlled and compositional manner."
categories:
- functional-programming
- design-patterns
tags:
- state-monad
- functional-programming
- immutability
- monads
- state-management
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/effect-handling-patterns/side-effects/state-monad"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In functional programming, managing state immutably can be challenging due to the constraint that functions should be pure (i.e., without side effects). The **State Monad** addresses this challenge by encapsulating state within computations and threading it through a sequence of operations in a controlled and compositional manner.

This article delves into the State Monad design pattern, exploring its principles, usage, and benefits. By the end of this article, you should understand how the State Monad works and how to apply it in your functional programming endeavors.

## What is the State Monad?

The State Monad is a monadic design pattern used to manage state in functional programming by passing the state explicitly through computational steps. It can be defined mathematically as:

{{< katex >}}
\text{State} \, S \, A = S \rightarrow (A, S)
{{< /katex >}}

Where `S` is the type of the state, and `A` is the type of the result.

This pattern offers a way to maintain state without violating the principles of immutability and makes an application of state transformations more modular and readable.

## The Structure of State Monad

In Haskell, a typical definition of the State Monad looks as follows:

```haskell
newtype State s a = State { runState :: s -> (a, s) }
```

Here, `State` is a newtype wrapper around a function that takes a state `s` and returns a tuple consisting of a value `a` and a new state `s`.

## Basic Operations

### `return` or `pure`

This operation encapsulates a value into the State Monad without modifying the state.

```haskell
return :: a -> State s a
return a = State $ \s -> (a, s)
```

### `bind` (>>=)

This operation sequences computations in the State Monad.

```haskell
(>>=) :: State s a -> (a -> State s b) -> State s b
(State sa) >>= f = State $ \s -> 
  let (a, s1) = sa(s)
      (State sb) = f(a)
  in sb(s1)
```

### `get` and `put`

`get` retrieves the current state, and `put` updates the state.

```haskell
get :: State s s
get = State $ \s -> (s, s)

put :: s -> State s ()
put s = State $ \_ -> ((), s)
```

## Example Usage

Given the State Monad structure and basic operations, let's create a simple example. Consider a counter that increments its state:

```haskell
import Control.Monad.State

type Counter = Int

increment :: State Counter ()
increment = do
  counter <- get
  put (counter + 1)

runCounter :: State Counter () -> Counter -> Counter
runCounter computation initial = snd $ runState computation initial

main :: IO ()
main = do
  let finalState = runCounter (replicateM_ 10 increment) 0
  print finalState  -- Output: 10
```

## Related Design Patterns

- **Reader Monad**: Useful for dependency injection where configurations or environments are passed through computations.
- **Writer Monad**: Helps in accumulating logs or outputs alongside performing computations.
- **Monad Transformer**: Stack different monads to combine their functionalities and manage multiple aspects such as state and side effects.

## Additional Resources

- [Haskell Wiki on State Monad](https://wiki.haskell.org/State_Monad)
- [Learn You a Haskell for Great Good!: State Monad](http://learnyouahaskell.com/for-a-few-monads-more#state)

## Summary

The State Monad is a powerful design pattern in functional programming for managing state without compromising on purity and immutability. It provides a clean and compositional way to thread state through computations, making your functional code more modular and maintainable. By understanding and applying the State Monad, you can better handle stateful operations while adhering to functional programming principles.
