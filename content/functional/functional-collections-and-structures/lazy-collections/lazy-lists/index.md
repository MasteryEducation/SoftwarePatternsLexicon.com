---
linkTitle: "Lazy Lists"
title: "Lazy Lists: A linked list whose elements are computed lazily"
description: "Lazy lists, a powerful concept in functional programming, allow for the creation of potentially infinite data structures whose elements are computed only as needed. This design pattern offers efficiency and flexibility in dealing with large or infinite sequences."
categories:
- Functional Programming
- Design Patterns
tags:
- Lazy Evaluation
- Infinite Data Structures
- Efficiency
- Functional Programming
- Haskell
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/functional-collections-and-structures/lazy-collections/lazy-lists"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction to Lazy Lists

Lazy lists are a fundamental design pattern in functional programming that play a crucial role in managing potentially infinite data structures. In contrast to eagerly evaluated lists, where all elements are computed upfront, lazy lists defer computation until the elements are actually accessed. This pattern leverages the principle of *lazy evaluation*, allowing for more efficient memory usage and the ability to work with seemingly infinite data.

### Why Use Lazy Lists?

- **Efficiency:** Only the necessary elements are computed, reducing overhead.
- **Memory Management:** Lazy lists enable handling large datasets without requiring substantial memory since not all data need to be held in memory simultaneously.
- **Modularity:** Providing a convenient abstraction for generating sequences of data that can be composed to achieve more complex behaviors.

## Implementing Lazy Lists

### Example in Haskell

Haskell, a language known for its lazy evaluation, provides a straightforward way to implement lazy lists. Here’s an example of a simple infinite lazy list:

```haskell
naturals :: [Integer]
naturals = [0..] -- an infinite list of natural numbers
```

To ensure only specific elements are computed, you can define a custom lazy list generator:

```haskell
data LazyList a = Cons a (LazyList a)

-- A simple example creating an infinite list of integers starting from n
from :: Integer -> LazyList Integer
from n = Cons n (from (n + 1))

-- Taking the first n elements of a lazy list
takeL :: Int -> LazyList a -> [a]
takeL 0 _            = []
takeL _ (Cons h t) = h : takeL (n-1) t
```

### Example in Scala

In Scala, lazy lists are handled with the `LazyList` (formerly `Stream`) class. Here is how you can create an infinite lazy list:

```scala
val natuals: LazyList[Int] = LazyList.from(0) // an infinite list of natural numbers
```

Custom generators can also be created using similar function patterns:

```scala
def from(n: Int): LazyList[Int] = n #:: from(n + 1)

def takeL[A](n: Int, s: LazyList[A]): List[A] = 
  if (n <= 0) Nil
  else s.head :: takeL(n - 1, s.tail)
```

## Related Design Patterns

### **Streams**

Streams in functional programming often use lazy evaluation and are similar to lazy lists, focusing on data processing rather than storage. They allow for efficient operations on large datasets, such as transformations and reductions, providing a pipeline for processing data elements as they become available.

### **Generators**

Generators can be seen as an imperative counterpart to functional lazy lists. They yield values on demand, providing another way to handle potentially infinite sequences. While immutability is less emphasized in generators, they still encapsulate the concept of deferring computation.

### **Iterators**

Iterators provide a way to access elements of a collection sequentially without exposing its underlying representation. In functional programming, iterators can be combined with lazy evaluation to efficiently handle large datasets.

## Best Practices

1. **Avoid Premature Optimization:** While lazy lists can optimize resource usage, ensure they provide a clear benefit in context before implementing.
2. **Mind the Accumulated Thunks:** Thunks (unevaluated expressions) can accumulate and lead to memory issues if not managed properly.
3. **Composable Generators:** Leverage composability of functions to build complex lazy lists from simpler ones.
4. **Unit Tests:** Ensure good test coverage to identify any lazy evaluation issues (e.g., ensuring termination when expected).

## Additional Resources

- [Real World Haskell, Chapter 24 - Profiling and Optimization](http://book.realworldhaskell.org/read/profiling-and-optimization.html)
- [Functional Programming in Scala, Chapter 5 - Strictness and Laziness](https://www.manning.com/books/functional-programming-in-scala)
- [Haskell 2010 Language Report](https://www.haskell.org/onlinereport/haskell2010)

## Summary

Lazy lists encapsulate a powerful paradigm in functional programming, promoting efficient handling of large or infinite collections by deferring computation until needed. Their application spans a broad spectrum of use cases, from infinite sequence generation to efficient large-data processing pipelines. While providing substantial benefits, they require careful handling of laziness to avoid pitfalls like memory bloat due to unforced thunks. By adhering to best practices, lazy lists can significantly enhance functional code's efficiency and readability.
