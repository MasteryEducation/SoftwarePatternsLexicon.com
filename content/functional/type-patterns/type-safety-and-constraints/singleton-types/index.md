---
linkTitle: "Singleton Types"
title: "Singleton Types: Types with Only One Possible Value"
description: "An in-depth exploration of the Singleton Types design pattern in functional programming, focusing on types that have only one possible value."
categories:
- Functional Programming
- Design Patterns
tags:
- Singleton Types
- Types
- Functional Programming Patterns
- Design Patterns
- Programming
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/type-patterns/type-safety-and-constraints/singleton-types"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


In functional programming, singleton types are types that possess only one possible value. These types are particularly useful in expressing more precise type information and enforcing stronger type guarantees. This article delves into the uses, benefits, examples, and related patterns of singleton types in functional programming.

## Introduction to Singleton Types

Singleton types are specialized types that could only ever have one value. This concept aids in elevating the type system's expressiveness by narrowing down possible values a type can hold, thus adding an extra layer of type safety.

```haskell
data Unit = ()
```

In the example above, `Unit` is a singleton type since it can only take the single value `()`.

### Uses of Singleton Types

1. **Type-level Programming**: Singleton types are instrumental in type-level computations and compile-time guarantees. 
2. **Configuration**: Singleton types often represent configuration options that don't change.
3. **Parameterization**: They allow parameterizing types and functions in a way that constrains them to a specific, singleton value.

## Examples in Haskell

### Unit Type

The Unit type is the canonical example of a singleton type in Haskell.

```haskell
data Unit = Unit deriving (Eq, Show)
```

Here, `Unit` is a type with a single value, also named `Unit`.

### Singleton for Nats and Type-level Literals

Singleton types extend to more complex constructs like encoding numbers at the type level using singletons.

```haskell
data Nat = Zero | Succ Nat

data SNat :: Nat -> Type where
  SZero :: SNat 'Zero
  SSucc :: SNat n -> SNat ('Succ n)
```

The `SNat` type ensures type-safe natural numbers by leveraging Haskell's type-level programming features.

## Benefits of Singleton Types

1. **Enhanced Type Safety**: They restrict types to one value, catching errors at compile-time.
2. **Clarity and Documentation**: Singleton types provide clearer intentions and self-documenting code.
3. **Reduced Runtime Checks**: Since possible values are constrained by the type system, fewer runtime checks are necessary.

## Related Design Patterns

### Phantom Types

Phantom types use types that do not hold values but serve to provide additional type safety. They often work with singleton types to encode more information at the type level.

```haskell
data Phantom a = Phantom -- 'a' is not used at runtime
```

### Property-Based Testing

Singleton types can enhance property-based testing by constraining types and thus simplifying the properties that need to be verified.

### Type-Level Computations

Alongside Generalized Algebraic Data Types (GADTs) and Type Families, singleton types are integral to type-level computations and state machines that offer compile-time guarantees and verification.

## Using Singleton Types in a Practical Application

Consider an API where configurations need to be set. Singleton types ensure configuration parameters are used correctly.

```haskell
data Config = DefaultConfig | CustomConfig

data Mode :: Config -> Type where
    DefaultMode :: Mode 'DefaultConfig
    CustomMode  :: Mode 'CustomConfig

initialize :: Mode 'DefaultConfig -> IO ()
initialize DefaultMode = putStrLn "Initializing with default config."

-- This ensures 'initialize' can only be called with 'DefaultMode'.
```

## Additional Resources

- [Type-Level Programming in Haskell](https://www.oreilly.com/library/view/type-level-programming-in/9781484245270/)
- [Haskell Programming from First Principles](http://haskellbook.com/)
- [Haskell: The Craft of Functional Programming](https://www.amazon.com/Haskell-Craft-Functional-Programming-3rd/dp/0201882957)

## Final Summary

Singleton types are a powerful tool in the functional programmer’s toolbox, allowing for more precise and safer code by constraining the values a type can take. They enhance type safety, clarity, and performance through reduced runtime checks. When used alongside related patterns such as phantom types and type-level computations, singleton types contribute significantly to robust and maintainable code design.

By leveraging singleton types, Haskell developers can write more precise, safer, and expressive programs, embodying the principles of functional programming to their fullest.

---

This comprehensive overview of singleton types should equip you with a deeper understanding of their principles, applications, and benefits in functional programming.
