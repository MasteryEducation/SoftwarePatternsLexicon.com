---
linkTitle: "Generator"
title: "Generator: A function that returns an iterator which yields items one at a time"
description: "The Generator pattern in functional programming involves a function returning an iterator, which yields items one at a time, facilitating lazy evaluation and efficient data processing."
categories:
- Functional Programming
- Design Patterns
tags:
- Functional Programming
- Lazy Evaluation
- Iterators
- Efficiency
- Higher-Order Functions
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/functional-collections-and-structures/lazy-collections/generator"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

The **Generator** pattern is an essential design pattern in functional programming that focuses on creating functions which return iterators. These iterators can sequentially produce items one at a time, a powerful technique that supports *lazy evaluation*, enabling efficient data processing especially for large datasets.

## Core Concept

In functional programming, a generator function maintains its local state, yielding outputs one at a time and resuming its execution where it left off after each yield. This mechanism helps in managing memory effectively, as items are generated on-the-fly without loading the entire dataset into memory.

## Key Properties

- **Lazy Evaluation**: Values are computed on demand rather than upfront. This avoids unnecessary computations and potential overhead in memory usage.
- **Stateful Iteration**: The generator retains its state between yields, enabling complex iteration patterns with minimal code complexity.
- **Composable and Reusable**: By leveraging higher-order functions, generators can be composed and reused effectively, enhancing modularity and maintainability.

## Example

Let's illustrate a simple generator in Python, which generates an infinite sequence of Fibonacci numbers:

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib_gen = fibonacci()
for _ in range(10):
    print(next(fib_gen))
```

In this example, the `fibonacci` function is a generator that infinitely yields Fibonacci numbers one at a time. The `yield` statement pauses the function’s execution and sends a value back to the caller.

## Generator Patterns in Various Languages

### Haskell

In Haskell, lazy evaluation is a built-in feature, and generators can be elegantly expressed with list comprehensions or `unfoldr`:

```haskell
import Data.List (unfoldr)

fibonacci :: [Integer]
fibonacci = unfoldr (\\(a,b) -> Just (a, (b, a + b))) (0, 1)
```

Here, `unfoldr` builds a list by successively applying a function and unwinding the state recursively.

### JavaScript (ES6)

JavaScript ES6 supports generators natively via the `function*` syntax:

```javascript
function* fibonacci() {
    let [a, b] = [0, 1];
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}

// Usage
const fibGen = fibonacci();
for (let i = 0; i < 10; i++) {
    console.log(fibGen.next().value);
}
```

The `function*` declaration defines a generator function, and `yield` effectively manages the state and output.

## Related Design Patterns

- **Iterator Pattern**: Both generators and iterators facilitate traversing a container without exposing its underlying representation. However, while iterators access elements in a pre-existing collection, generators generate those elements on-the-fly.
- **Lazy Evaluation**: Generators are a common tool for implementing lazy evaluation, deferring computation until absolutely necessary and minimizing resource usage.
- **Monad**: In functional programming, the generator can be conceptualized with monadic operations that encapsulate and chain computations, guiding complex state management.

## Additional Resources

1. **Books and Papers**
    - "Functional Programming in Scala" by Paul Chiusano and Rúnar Bjarnason: Delves into functional programming patterns including generators.
    - "Learn You a Haskell for Great Good!" by Miran Lipovaca: Offers insights into Haskell's approach to infinite lists and lazy evaluation.

2. **Online Articles and Tutorials**
    - [Python Generators](https://docs.python.org/3/howto/functional.html): Official Python documentation.
    - [JavaScript Generators](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Iterators_and_Generators): Comprehensive guide on MDN.

## Summary

The Generator design pattern embodies core functional programming principles, facilitating efficient and lazy data processing through the creation of functions that return iterators yielding items sequentially. Its use across different programming languages underlines its versatility and utility in various domains and applications. By leveraging generators, developers can create more efficient, readable, and maintainable code, particularly for handling large and potentially infinite data sequences.

This write-up captures both the theoretical and practical aspects of the Generator pattern, ensuring a robust understanding for functional programming enthusiasts and practitioners.
{{< katex />}}

