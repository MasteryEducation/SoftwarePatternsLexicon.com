---
linkTitle: "Range"
title: "Range: Generate a Sequence of Values Given a Start, End, and Step"
description: "In functional programming, the range pattern is utilized to generate a sequence of values by specifying a start, an end, and a step. This technique is often used for iteration, computation, and lazy evaluation, especially in scenarios where sequences need to be defined succinctly and efficiently."
categories:
- Functional Programming
- Design Patterns
tags:
- functional programming
- range
- sequence generation
- iteration
- lazy evaluation
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/recursion-and-iteration-patterns/iteration/range"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Overview
The **Range** design pattern is a fundamental concept in functional programming used to create a sequence of values between a defined start and end point with a specific step value. This enables the concise and efficient generation of sequences, which is useful for various purposes, such as iteration, computation, and lazy evaluation.

## Detailed Explanation
In functional programming, the Immutable Range feature frequently appears as part of standard library functions in languages such as Haskell, Scala, and F#. The **Range** pattern is not just a loop but a definition of a list or stream of values that can be processed functionally.

### Signature
A typical signature of a range generating function might look like:
```haskell
range :: (Integral a) => a -> a -> a -> [a]
range start end step = ...
```

### Example Implementations

#### Haskell
In Haskell, the range generation is often achieved through list comprehension:
```haskell
range :: (Enum a, Ord a) => a -> a -> a -> [a]
range start end step 
  | start > end = []
  | otherwise   = start : range (start + step) end step 
```

#### Scala
Scala uses a similar approach with richer type annotation and collection operations:
```scala
def range(start: Int, end: Int, step: Int): Seq[Int] = {
  if (start >= end) Seq.empty
  else start +: range(start + step, end, step)
}
```

#### F#
In F#, the [Seq.initR] function helps generate a sequence based on computation:
```fsharp
let range start end step =
    Seq.unfold (fun state -> if state > end then None else Some(state, state + step)) start
```

### Lazy Evaluation
In languages supporting lazy evaluation, such as Haskell, range sequences are efficiently implemented and executed only when needed. This allows for infinite sequences and high performance in computational tasks:
```haskell
let infiniteRange = [0, 2 .. ]
take 10 infiniteRange -- Output: [0, 2, 4, 6, 8, 10, 12, 14, 16]
```

## Related Design Patterns

### Iterator Pattern
The **Iterator Pattern** allows traversal of a data structure without exposing its underlying representation. The range pattern can create sequences that are iterated through.

### Stream Processing
**Stream Processing** deals with sequences of data items where range values can be used as input streams or computational sequences in functional reactive programming.

### Lazy Evaluation
**Lazy Evaluation** defers computation until the value is needed, making it a critical optimization technique that pairs well with the range pattern.

## Additional Resources
1. "Learn You a Haskell for Great Good!" by Miran Lipovača - Provides a gentle introduction to Haskell and functional programming principles.
2. "Functional Programming in Scala" by Paul Chiusano and Rúnar Bjarnason - Explores fundamental concepts in Scala.
3. "Programming F#" by Chris Smith - A comprehensive guide to F# programming and functional paradigms.

## Summary
The **Range** design pattern is an indispensable tool in functional programming for generating sequences. Its applications in iteration and lazy computations make it powerful and versatile. Understanding and effectively utilizing the range pattern can significantly enhance code efficiency and expressiveness in functional programming languages.
