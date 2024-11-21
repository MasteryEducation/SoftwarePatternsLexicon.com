---
linkTitle: "Future"
title: "Future: A Placeholder for Asynchronous Results"
description: "A Future represents a value that may not yet be available, enabling asynchronous programming by capturing the eventual result."
categories:
- Functional Programming
- Asynchronous Programming
tags:
- Future
- Async
- Concurrency
- Promises
- Non-blocking
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/parallel-and-concurrent-programming-patterns/concurrency/future"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

In functional programming and concurrent paradigms, a **Future** represents a value that may not yet be available but will be computed and possibly retrieved at some later time. Futures provide a powerful abstraction for working with asynchronous computations, making it possible to handle tasks such as I/O operations, network requests, or long-running calculations in a non-blocking, efficient manner.

## Essentials of Future

A Future's lifecycle includes:

1. **Creation**: A Future is initially created as a placeholder.
2. **Assignment**: The placeholder eventually gets assigned with a value upon completion of the computation.
3. **Computation Handling**: While the value is pending, you can attach callbacks or transformations that will process the result once it becomes available.

### Key Properties

- **Non-blocking**: While a Future represents a computation that runs in the background, other operations can continue without waiting for the Future's result.
- **Composability**: Futures can be combined and transformed using various higher-order functions. 
- **Error Handling**: They provide mechanisms for dealing with errors that may occur during the asynchronous operations.
  
Below is an example of a simple Future implementation in Scala:

```scala
import scala.concurrent.{Future, ExecutionContext}
import ExecutionContext.Implicits.global

val futureResult: Future[Int] = Future {
  // Some long-running computation
  42
}

futureResult.map(result => println(s"Result: $result"))
```

## Implementation Details

### Creating Futures

Creating a Future typically involves starting a computation on a separate thread and immediately returning a value representing this future result.

### Transformations

Transformations can be applied to Futures via operations like `map`, `flatMap`, `filter`, etc. These operations allow you to describe a chain of actions without blocking.

```scala
val futureComputation: Future[Int] = Future { 
  // Complex Calculation
  10
}

val transformedFuture: Future[String] = futureComputation.map(result => s"The result is $result")
```

### Handling Errors

Futures provide utilities to handle errors using combinators like `recover`, `recoverWith`, and `fallbackTo`.

```scala
val failedFuture: Future[Int] = Future {
  throw new RuntimeException("Failed computation")
}

val recoveredFuture: Future[Int] = failedFuture.recover {
  case _: RuntimeException => 0
}
```

## Related Design Patterns

### Promises 

A **Promise** is a more granular construct representing an object that can be manually completed with a Future's value. Promises allow separating the creation and the assignment of a Future's value.

### Task/Rx Observable

A **Task** or an **Observable** in Reactive Extensions provides a more elaborate control of asynchronous computations, including rich composition mechanisms like merging, concatenating, and error propagation.

### Monad

The `Future` can be seen as a special kind of monad, providing `flatMap` and `map` functions to sequence asynchronous computations in a more structured way.

## Additional Resources

- [Scala Futures and Promises](https://docs.scala-lang.org/overviews/core/futures.html)
- [Java CompletableFuture](https://docs.oracle.com/javase/8/docs/api/java/util/concurrent/CompletableFuture.html)
- [JavaScript Promises](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises)

## Summary

Futures are essential abstractions in functional programming for managing asynchronous computations without blocking. They enable developers to write efficient, readable, and maintainable code for handling tasks that require waiting for results from background operations, network requests, or I/O activities. Futures allow transformations, error handling, and composition, making them a robust tool for concurrent programming. By providing a clear interface for managing eventual values, Futures help simplify the complexity associated with asynchronous programming.

---

This article showcases the **Future** design pattern, elucidating its core principles, implementation details, and relationship with related patterns. Through straightforward examples and additional resources, it aims to provide a comprehensive understanding of Futures in functional programming.
