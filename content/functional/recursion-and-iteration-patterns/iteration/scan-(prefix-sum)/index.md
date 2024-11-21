---
linkTitle: "Scan (Prefix Sum)"
title: "Scan (Prefix Sum): Producing a Sequence of Running Totals from an Accumulator Function"
description: "The Scan or Prefix Sum design pattern involves producing a sequence of running totals by applying an accumulator function. This pattern is particularly useful in functional programming for parallelization and incremental computations."
categories:
- functional-programming
- design-patterns
tags:
- functional-design-patterns
- scan
- prefix-sum
- accumulator
- functional-programming
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/recursion-and-iteration-patterns/iteration/scan-(prefix-sum)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

The **Scan** (also known as **Prefix Sum**) design pattern involves generating an output sequence that contains the running totals (partial sums) resulting from applying an accumulator function to an input sequence. It is a fundamental concept in functional programming, used for tasks that require cumulative results, such as parallel computing, incremental computations, and stream processing.

The key characteristic of the Scan pattern is that it produces not just a single accumulated result like `fold` or `reduce`, but an entire sequence of partial results, enabling further computations. In languages embracing functional paradigms, this pattern promotes immutability and helps optimize performance and parallel execution.

## Mathematical Definition and Notation

Given a sequence of elements \\( [x_1, x_2, \ldots, x_n] \\) and an associative binary function \\( f \\), the **Scan** function computes a new sequence \\( [y_1, y_2, \ldots, y_n] \\) where:

{{< katex >}} y_i = f(y_{i-1}, x_i) \quad \text{for} \quad i \in [2, n], \quad \text{with} \quad y_1 = x_1 {{< /katex >}}

More formally, the elements in the resulting sequence \\( [y_1, y_2, \ldots, y_n] \\) are defined as:

{{< katex >}}
\begin{align*}
y_1 &= x_1 \\
y_2 &= f(y_1, x_2) \\
y_3 &= f(y_2, x_3) \\
&\vdots \\
y_n &= f(y_{n-1}, x_n)
\end{align*}
{{< /katex >}}

In functional programming terms, the Scan pattern can be implemented using recursive constructs or built-in higher-order functions.

## Use Cases and Examples

### Example in Haskell

```haskell
scanl :: (b -> a -> b) -> b -> [a] -> [b]
scanl f q ls = q : (case ls of
                      []   -> []
                      x:xs -> scanl f (f q x) xs)
```

#### Example Usage

```haskell
-- Computes the prefix sums of a list of numbers
> scanl (+) 0 [1, 2, 3, 4]
[0, 1, 3, 6, 10]
```

In this example, `scanl` calculates the prefix sums of the list `[1, 2, 3, 4]` starting from an initial value of `0`.

### Example in Scala

```scala
def scanLeft[A, B](z: B)(op: (B, A) => B)(seq: Seq[A]): Seq[B] = {
  seq.foldLeft(Seq(z)) { (acc, x) => 
    acc :+ op(acc.last, x)
  }
}
```

#### Example Usage

```scala
val nums = List(1, 2, 3, 4)
nums.scanLeft(0)(_ + _)
// List(0, 1, 3, 6, 10)
```

In this Scala example, `scanLeft` computes the prefix sums of the list `List(1, 2, 3, 4)` starting from `0`.

## Benefits of the Scan Pattern

1. **Incremental Computation:** It provides all intermediate results, making it useful for incremental computations where each result builds on the previous one.
2. **Parallelism:** In specific implementations, especially in divide-and-conquer strategies, Scan can be parallelized efficiently.
3. **State Management:** Useful in managing state transformations, especially in stateful computations or stream processing.
4. **Immutability:** Enforces immutability, reducing side effects and enhancing code reliability within functional paradigms.

## Related Design Patterns

### **Fold/Reduce**

- **Description:** Produces a single accumulated result from a collection rather than a sequence of intermediate results.
- **Example Use Case:** Summing a list of numbers.
- **Difference from Scan:** Returns only the final result as opposed to an entire sequence of partial results.

### **Map**

- **Description:** Applies a function to every element of a collection independently.
- **Example Use Case:** Transforming a list of numbers by doubling each element.
- **Difference from Scan:** Each transformation is independent of others, whereas Scan involves dependency on previous results.

### **Filter**

- **Description:** Produces a subset of elements that satisfy a predicate.
- **Example Use Case:** Extracting even numbers from a list.
- **Difference from Scan:** Only selects elements based on a condition without modification or accumulation.

## Additional Resources

- **Books:** 
  - *"Introduction to Functional Programming using Haskell"* by Richard Bird
  - *"Functional Programming in Scala"* by Paul Chiusano and Runar Bjarnason
- **Online Articles and Documentation:**
  - [Haskell Data.List.scanl](https://hackage.haskell.org/package/base-4.14.1.0/docs/Data-List.html#v:scanl)
  - [Scala API Documentation for Seq](https://www.scala-lang.org/api/current/scala/collection/Seq.html#scanLeft[B>:A](z:B,op:(B,A)=>B):Seq[B])
- **Tutorials:**
  - Functional Programming Principles in Scala by Martin Odersky on Coursera

## Final Summary

The **Scan (Prefix Sum)** design pattern is a fundamental tool in functional programming used to generate a sequence of running totals from an input sequence and an accumulator function. It is characterized by its ability to produce intermediate results, which is beneficial for parallel computations, state management, and incremental updates. By understanding and applying this pattern, developers can write more efficient, reliable, and parallelizable code. It stands as a critical counterpart to other functional patterns like `fold`, `map`, and `filter`, each serving distinct purposes with overlaps in their application fields.
