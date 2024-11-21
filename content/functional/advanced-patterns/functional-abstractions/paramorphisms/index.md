---
linkTitle: "Paramorphisms"
title: "Paramorphisms: Generalizing folds to access the pre-folded structure"
description: "A detailed exploration of Paramorphisms in functional programming, how they generalize folds, and access the pre-folded structure."
categories:
- Functional Programming
- Design Patterns
tags:
- Paramorphisms
- Folds
- Recursion Schemes
- Functional Programming Patterns
- Higher-Order Functions
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/advanced-patterns/functional-abstractions/paramorphisms"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Paramorphisms are a fundamental concept in functional programming that generalize the familiar concept of folds (also known as catamorphisms). While a fold processes a data structure to produce a single accumulated result by recursively reducing its elements, a paramorphism retains access to the original pre-folded structure during its recursive processing. This enables more complex recursion patterns and is useful when the result at each step depends not only on the already reduced values but also on the original structure.

## Definition

Formally, a paramorphism can be defined as a recursive function that, at each step, takes two arguments:
1. The current element of the structure being folded.
2. The structure itself.

The result of a paramorphism is constructed by combining the current element and the result of the paramorphism applied to the rest of the structure, along with the unprocessed structure.

## Mathematical Representation

Let's represent a paramorphism for a list in Haskell-like pseudo-code:

{{< katex >}} \text{para} \ f \ z \ [\,] = z {{< /katex >}}
{{< katex >}} \text{para} \ f \ z \ (x:xs) = f \ x \ xs \ (\text{para} \ f \ z \ xs) {{< /katex >}}

Where `f` is the combining function, `z` is the base case (usually the identity value for the result type), `x` is the current element, and `xs` is the remaining list.

## Example

Consider computing the list of suffixes of a list using a paramorphism:

```haskell
suffixes :: [a] -> [[a]]
suffixes = para (\x xs suffixesXs -> (x : xs) : suffixesXs) []
```

Here, the function builds the list of suffixes by using the paramorphism to keep the access to the list `xs` in each recursive step.

### Breakdown:
- `x : xs` adds the current element `x` to the rest of the list `xs`.
- `(x : xs) : suffixesXs` includes the current list as the first element of the result and proceeds recursively.

## Relationships to Other Patterns

### Catamorphisms (Folds)

- **Catamorphism**: A fold is a simple reduction.
  
  ```haskell
  foldr :: (a -> b -> b) -> b -> [a] -> b
  foldr f z []     = z
  foldr f z (x:xs) = f x (foldr f z xs)
  ```

- **Paramorphism**: A paramorphism can be seen as an extension where the folding function has access to more information, namely the rest of the list.

### Anamorphisms

An anamorphism is the opposite of a catamorphism, a function that unfolds a value into a more complex structure.

- **Anamorphism**: 

  ```haskell
  unfoldr :: (b -> Maybe (a, b)) -> b -> [a]
  unfoldr f b = case f b of
                  Just (x, b') -> x : unfoldr f b'
                  Nothing      -> []
  ```

### Apomorphisms

An apomorphism extends anamorphisms by allowing the option to produce the next state or terminate the unfolding early.

- **Apomorphism**: Starts with an initial seed and continues to produce values as long as a continuation function applies.

### Hylomorphisms

Hylo = Ana + Cata; hylo combines both (folding while unfolding).

- **Hylomorphism**:

```haskell
hylo :: (b -> c) -> (a -> (b, a)) -> a -> c
hylo c f a = c (f a)
```

### Zygomorphisms

Zygomorphisms allow simultaneous computation of two folds that may share some intermediate state.

- **Zygomorphism**: Uses two combining functions and one shared intermediate state across folds.

## Practical Use-Cases

1. **Tree Structures**: Accessing subtrees in binary trees or n-ary trees.
2. **Dynamic Algorithms**: Examples include dynamic programming where retained intermediate results access both computed and the original data structure.
3. **Suffix Trees**: Generating suffixes or subsets of lists, as demonstrated earlier, is simplified with paramorphisms.

## Additional Resources

- "Functional Programming Patterns in Scala and Clojure" by Michael Bevilacqua-Linn for practical insights.
- "The Science of Functional Programming" by Benjamin C. Pierce for theoretical underpinnings.
- Haskell documentation for understanding idiomatic constructs and additional examples of paramorphisms.

## Summary

Paramorphisms are a sophisticated functional programming pattern that extend simple folds to give more power and flexibility by retaining access to the original structure during the computation. This makes paramorphisms a valuable tool in many advanced recursive algorithms, allowing programmers to process structures such as lists or trees with greater nuance.

Understanding how to leverage paramorphisms can greatly enhance one’s functional programming toolkit, especially when working on algorithms requiring access to both computed and original data states. They stand as a critical intersection between various recursion schemes, bridging simple folds and more complex recursive patterns.
