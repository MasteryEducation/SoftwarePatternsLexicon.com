---
linkTitle: "Memoization"
title: "Memoization: Caching the Results of Expensive Function Calls"
description: "Memoization is a technique to improve the performance of functional programs by caching the results of expensive function calls and reusing them when the same inputs occur again."
categories:
- Functional Programming
- Design Patterns
tags:
- Performance
- Caching
- Functional Programming
- Optimization
- Pure Functions
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/optimizations/caching-and-specialization/memoization"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Memoization is a powerful optimization technique used in functional programming to enhance the performance of programs. By caching the results of expensive function calls and reusing them when the same inputs occur again, memoization can significantly reduce the number of computations, leading to faster and more efficient programs.

## What is Memoization?

Memoization involves storing the results of function calls along with their inputs. When a function is called, the memoization system checks whether it has computed the result for the given input before. If it has, the cached result is returned immediately, bypassing the need for recomputation. Otherwise, the function is executed, and the result is stored for future use.

### Key Characteristics of Memoization:
- **Immutability**: Memoized functions should be pure, meaning they do not cause side effects and always produce the same output for the same input.
- **Cache Efficiency**: The system should efficiently manage the storage and retrieval of cached results.
- **Reusability**: Cached results can be reused any number of times when their respective inputs recur.

## Memoization Implementation

### 1. Basic Memoization in JavaScript

Below is a simple example of memoization in JavaScript:

```javascript
const memoize = (fn) => {
  const cache = {};
  return (...args) => {
    const key = JSON.stringify(args);
    if (cache[key]) {
      return cache[key];
    }
    const result = fn(...args);
    cache[key] = result;
    return result;
  };
};

const expensiveFunction = (num) => {
  // Some expensive computation
  return num * num;
};

const memoizedExpensiveFunction = memoize(expensiveFunction);

console.log(memoizedExpensiveFunction(5)); // Computed
console.log(memoizedExpensiveFunction(5)); // Cached
```

### 2. Memoization in Haskell

In Haskell, memoization can be gracefully handled due to its lazy evaluation. This use case involves memoizing Fibonacci numbers:

```haskell
import Data.Map (Map)
import qualified Data.Map as Map

memoize :: (Ord k) => (k -> a) -> k -> Map k a -> (a, Map k a)
memoize f x cache =
  case Map.lookup x cache of
    Just result -> (result, cache)
    Nothing ->
      let result = f x
       in (result, Map.insert x result cache)

main = do
  let fibonacci = \n -> if n < 2 then n else (fst $ memoize fibonacci (n-1) Map.empty) + (fst $ memoize fibonacci (n-2) Map.empty)
  print $ fibonacci 10
```

## Related Design Patterns

### 1. **Lazy Evaluation**
Lazy evaluation delays computation until the result is required. This can be seen as a form of memoization since unnecessary computations are avoided.

### 2. **Caching**
Caching in general is related to memoization. While memoization caches function outputs, caching can be applied to a wider variety of use cases like web content, database results, etc.

### 3. **Decorator Pattern**
The decorator pattern, when used for memoization, involves wrapping a function to add caching behavior transparently.

## Additional Resources

- [Wikipedia: Memoization](https://en.wikipedia.org/wiki/Memoization)
- [Functional Programming in JavaScript by Luis Atencio](https://www.manning.com/books/functional-programming-in-javascript)
- [Learn You a Haskell for Great Good! by Miran Lipovaca](http://learnyouahaskell.com/)

## Summary

Memoization is an effective optimization technique in functional programming that stores the results of expensive operations and returns the cached result when the same inputs occur again. It leverages the immutability and determinism of pure functions to achieve substantial performance gains. By understanding and applying memoization, software engineers can create more efficient and faster applications, especially when dealing with computationally intensive tasks.


