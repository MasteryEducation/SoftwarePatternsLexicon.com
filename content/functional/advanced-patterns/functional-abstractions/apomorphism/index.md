---
linkTitle: "Apomorphism"
title: "Apomorphism: Dual of Paramorphism"
description: "An in-depth study of Apomorphism: an unfold pattern with access to the structure before unfolding, functioning as the dual of Paramorphism."
categories:
- Functional Programming
- Design Patterns
tags:
- Apomorphism
- Functional Programming
- Unfold
- Dual
- Paramorphism
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/advanced-patterns/functional-abstractions/apomorphism"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In functional programming, **Apomorphism** is a concept that deals with decomposing data structures. It is the dual of paramorphism and can be thought of as an unfold pattern with access to the structure of the data before it gets unfolded. This offers a powerful way to construct complex data by specifying how to continue 'unfolding' a structure while maintaining a handle on the current state.

## Mathematical Background

Mathematically, apomorphisms can be described using category theory and involve co-algebraic constructs. They essentially reverse the process of anamorphisms (standard unfolds) and can be specifically structured as follows:

1. **Corecursion**: Extending natural transformations to handle recursive structures more effectively.
2. **Accessing States**: Like paramorphisms, which allow access to a recursive structure during a fold, apomorphisms allow access during an unfold.

### Formal Definition

An apomorphism for some co-algebra \\( \alpha: A \rightarrow F(A + X) \\) can be generalized as:

{{< katex >}} f :: A \rightarrow F(A + X) {{< /katex >}}

Where `F` is a functor that defines the structure of our data.

This can be simplified and visualized using Haskell syntax:

```haskell
apo :: Functor f => (b -> f (Either a b)) -> b -> Fix f
```

In this case, `Fix f` refers to the fixed-point type of the functor `f`.

## Understanding Apomorphism with an Example

To grasp apomorphisms better, consider constructing a list that terminates early given a certain condition:

```haskell
data ListF a r = Nil | Cons a r

-- Type alias for our ListF functor
type List a = Fix (ListF a)

apo :: (b -> ListF a (Either b (List a))) -> b -> List a
apo f x = Fix (fmap (either (apo f) id) (f x))
```

In this example, `apo` takes a seed and a function that either terminates or continues unfolding, deciding at each step whether to complete the construction or keep building.

### Practical Example

Suppose we wish to build a list of integers where the construction stops once we hit a specific number:

```haskell
import Data.Functor.Foldable (Fix(Fix), para, apo)

-- Define our co-algebra for generating a list
coAlg :: Int -> ListF Int (Either Int [Int])
coAlg 0 = Nil
coAlg n = Cons n (Right (n - 1))

-- Generate a list from 5 to 0
myList = apo coAlg 5
```

The resultant list will be `[5, 4, 3, 2, 1, 0]`.

## Related Design Patterns

### Paramorphism

As the dual of apomorphism, **paramorphism** allows for folding with access to the original structure. While paramorphisms deconstruct existing structures, apomorphisms assist in constructing structures with access to the in-progress state:

```haskell
para :: Functor f => (f (Fix f, b) -> b) -> Fix f -> b
```

### Anamorphism

This is the standard unfold pattern, serving as the simpler cousin to apomorphisms:

```haskell
ana :: Functor f => (b -> f b) -> b -> Fix f
```

Anamorphisms do not provide access to the structure during unfolding, unlike apomorphisms.

### Catamorphism

Catamorphisms, or folds, are the opposite patterns of anamorphisms, focusing on collapsing structures:

```haskell
cata :: Functor f => (f a -> a) -> Fix f -> a
```

## Additional Resources

- [Category Theory for Programmers](https://bartoszmilewski.com/2014/10/28/category-theory-for-programmers)
- [Learn You a Haskell for Great Good!](http://learnyouahaskell.com)
- [Haskell Wiki: Apomorphisms](https://wiki.haskell.org/Apomorphism)
- Bird, R. (2010). *Pearls of Functional Algorithm Design*. Cambridge University Press.

## Summary

**Apomorphism** is an advanced unfolding pattern that facilitates building structures with recursive elements while retaining access to the state of the structure before complete unfolding. It serves as the dual to paramorphisms and is closely related to other recursion schemes such as anamorphisms and catamorphisms. Mastery of apomorphisms provides deeper insight into the construction and manipulation of complex functional data structures.

Understanding and leveraging apomorphisms enhance the functional programmer's toolkit, allowing for more expressive and powerful data processing capabilities.
