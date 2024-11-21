---
linkTitle: "Generator"
title: "Generator: Functions That Allow Iteration Over Sequences"
description: "In functional programming, the Generator design pattern enables functions to produce a sequence of values, one at a time, allowing calling code to iterate over these values without handling the entire sequence at once."
categories:
- Functional Programming
- Design Patterns
tags:
- Generator
- Sequences
- Iteration
- Laziness
- Yield
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/recursion-and-iteration-patterns/iteration/generator"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

In functional programming, the Generator design pattern is a powerful tool for working with sequences of values. Unlike traditional iterators that require computation of the entire sequence upfront, generators allow for the generation of values on the fly, one at a time. This enables more memory-efficient and computationally-efficient code, especially when dealing with large or potentially infinite sequences.

## Key Concepts

### Lazy Evaluation

Generators operate on the principle of lazy evaluation, where values are computed only when needed. This contrasts with eager evaluation, where all values are computed immediately.

### Yielding Values

A generator function does not return a single value. Instead, it `yields` multiple values, pausing its execution and preserving its state between yields.

### Stateful Iteration

Unlike stateless functions, generator functions maintain their execution context, enabling them to remember where they left off during the last invocation.

## Implementation

Here is an example of a simple generator function in Haskell, a popular functional programming language:

```haskell
-- Define a generator function using lazy lists
numsFrom :: Int -> [Int]
numsFrom n = n : numsFrom (n + 1)

-- Take the first 10 numbers from the generator
take 10 (numsFrom 1)
```

This example demonstrates a generator function `numsFrom` that produces an infinite sequence of integers starting from a given number.

## Using Generators in JavaScript

Generators are supported natively in JavaScript through the `function*` and `yield` syntax:

```javascript
// Generator function in JavaScript
function* numsFrom(n) {
    while (true) {
        yield n++;
    }
}

// Create the generator
const gen = numsFrom(1);

// Iterate over the sequence
for (let i = 0; i < 10; i++) {
    console.log(gen.next().value);
}
```

In this JavaScript example, calling `next()` on the generator object `gen` produces the next value in the sequence, maintaining its internal state.

## Related Design Patterns

### Iterator Pattern

The Iterator pattern provides a way to access elements of a collection without exposing its internal representation. While similar, generators inherently support creating sequences lazily and maintaining state between iterations.

### Producer-Consumer Pattern

In the Producer-Consumer pattern, one part of a system produces data while another part consumes it. Generators can act as producers in this context, generating data items incrementally.

### Stream Processing

Stream processing involves handling sequences of data, often in real-time and within limits of memory and processing power. Generators align with this paradigm by providing data elements one at a time, as needed.

## Additional Resources

- [Eloquent JavaScript: Generators](https://eloquentjavascript.net/13_async.html#h_hxCcCZRz8E)
- [Haskell Wiki: Infinite Lists](https://wiki.haskell.org/Infinite_list)
- [Functional Programming in JavaScript](https://drboolean.gitbooks.io/mostly-adequate-guide/content/ch9.html)

## Summary

The Generator design pattern is an essential tool in functional programming, enabling efficient and memory-conserving iteration over sequences. By leveraging lazy evaluation and stateful iteration, generators can produce complex sequences of values incrementally. Understanding and utilizing generators can lead to more performant and maintainable code, especially when dealing with large data sets or streams of data.

This article has explored the foundational principles of the Generator pattern, provided practical examples in Haskell and JavaScript, and highlighted related design patterns. Utilizing generators can greatly enhance the capabilities and efficiency of your functional programming projects.
