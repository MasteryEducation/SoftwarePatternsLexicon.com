---
linkTitle: "Peano Types"
title: "Peano Types: Structuring Natural Numbers by Type-Level Encoding"
description: "An exploration of Peano Types, a design pattern for encoding natural numbers at the type level, facilitating type-safe arithmetic operations in functional programming."
categories:
- Functional Programming
- Design Patterns
tags:
- Peano Types
- Type-Level Programming
- Haskell
- Type Safety
- Arithmetic
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/type-patterns/type-safety-and-constraints/peano-types"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Peano Types: Structuring Natural Numbers by Type-Level Encoding

### Introduction

In functional programming, ensuring correctness and safety at compile-time is a powerful paradigm. Peano types allow encoding natural numbers at the type level, promoting type safety in arithmetic operations. This pattern leverages type-level programming capabilities and is prevalent in strongly-typed languages such as Haskell. By encoding natural numbers at the type level, we can derive a system where operations on these numbers ensure correctness by construction.

### Peano Axioms

Peano types are based on Peano axioms, a set of axioms for the natural numbers proposed by Giuseppe Peano. The axioms define natural numbers starting with a base case and a recursive case.

1. **Zero (0) is a natural number**.
2. **The successor of any natural number is also a natural number**.

From these axioms, we can derive an algebraic data structure capturing these properties.

### Peano Types in Haskell

The type-level encoding involves defining a base type for zero and an inductive type for the successor of a given number.

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- Data kind for natural numbers
data Nat = Zero | Succ Nat

-- Type-level natural numbers
data SNat (n :: Nat) where
    SZero :: SNat 'Zero
    SSucc :: SNat n -> SNat ('Succ n)
```

### Arithmetic Operations

#### Addition

Type-level addition can be defined using type families.

```haskell
-- Type family for addition
type family Add (m :: Nat) (n :: Nat) :: Nat where
    Add 'Zero n    = n
    Add ('Succ m) n = 'Succ (Add m n)
```

#### Multiplication

Similarly, multiplication can leverage type families to operate at the type level.

```haskell
-- Type family for multiplication
type family Mul (m :: Nat) (n :: Nat) :: Nat where
    Mul 'Zero n = 'Zero
    Mul ('Succ m) n = Add n (Mul m n)
```

### Ensuring Type Safety

By using Peano types, addition and multiplication operations are type-checked to ensure correctness. Any invalid operations will be caught at compile-time.

### Related Design Patterns

**1. **Phantom Types**:
    Phantom types are types that do not impact runtime behavior but enforce constraints at compile-time. Peano types use phantom types to represent numbers at the type level.

**2. **Dependent Types**:
    While Haskell’s type system is not fully dependent, using Peano types is a step towards dependent type programming where types can depend on values, ensuring more rigorous correctness.

### Resources

- **[Type-Level Programming in Haskell](https://www.example.com)**: A comprehensive guide to type-level programming in Haskell.
- **[Haskell Definitive Guide](https://www.example.com)**: Detailed content on Haskell's type system and functional programming patterns.
- **[Functional Programming with Type-Level Arithmetic](https://www.example.com)**: Articles and applied examples on arithmetic performed at the type level.

### Summary

Peano types offer a structured way to encode natural numbers at the type level, ensuring that arithmetic operations respect type constraints and fostering compile-time correctness. By adopting Peano types in functional programming, one can leverage type-level programming to build robust and error-free systems. This pattern intersects with phantom types and hints towards the more expressive domain of dependent types, enhancing the mathematical rigor in software design.

Exploiting type-level primitives in functional programming aligns with the broader objective of building safe, maintainable, and correct-by-construction software systems.
