---
linkTitle: "Coroutine"
title: "Coroutine: General Control Structures for Non-Preemptive Multitasking"
description: "Coroutines provide a mechanism to allow different parts of code to cooperatively multitask, meaning that they can implicitly yield and resume execution at appropriate times."
categories:
- Functional Programming
- Design Patterns
tags:
- coroutine
- multitasking
- yield
- resumption
- non-preemptive
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/parallel-and-concurrent-programming-patterns/concurrency/coroutine"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

Coroutines are a fundamental construct for non-preemptive multitasking, designed to allow functions to pause their execution (`yield`) and subsequently resume from the same point. Unlike preemptive multitasking, coroutines require the developer to explicitly manage the interruption and continuation of the tasks. This makes them particularly impactful for simplifying asynchronous programming and creating state machines without relying on complex callback structures or heavy thread-based synchronization mechanisms.

## Basic Concepts

### Definition

A coroutine is a routine that can suspend its execution to be resumed later. This means that a coroutine can pause at some point and transfer control to another coroutine, which can then potentially do the same. This back-and-forth switching can continue, facilitating cooperative multitasking in lieu of threads or processes.

### How Coroutines Work

1. **Yielding**: When a coroutine hits a `yield` statement, it saves its current state (including variables, execution point, etc.) and returns control to the parent function, scheduler, or invoking code.
  
2. **Resuming**: When the coroutine is resumed, it starts executing exactly where it yielded control, maintaining its previous state.

3. **Continuation**: Execution continues until the coroutine explicitly yields control again or naturally terminates.

### Example in Python

```python
def simple_coroutine():
    print("Coroutine started")
    yield
    print("Coroutine resumed")

coro = simple_coroutine()

next(coro)  # Output: Coroutine started
coro.send(None)  # Output: Coroutine resumed
```

## Comparison with Generators

While coroutines and generators might seem similar, they serve distinct purposes:

- **Generators**: Primarily designed for generating sequences of values and provide simpler control over iteration (single-direction).
- **Coroutines**: Allow bidirectional data flow and control transfer, making them more suited for cooperative concurrency.

### Example Distinguishing Coroutines from Generators

```python
def counter_generator(n):
    for i in range(n):
        yield i

def echo_coroutine():
    while True:
        value = (yield)
        print(f"Echo received: {value}")

for count in counter_generator(3):
    print(count)  # Output: 0, 1, 2

echo = echo_coroutine()
next(echo)  # Prime the coroutine
echo.send("Hello")  # Output: Echo received: Hello
```

## Use-Cases of Coroutines

1. **Asynchronous Programming**: Manage asynchronous tasks without callback hell or deeply nested code.
2. **State Machines**: Simplify creation of state machines by maintaining state inside the coroutine.
3. **Data Pipelines**: Handle streaming data transformations on-the-fly.
4. **Iterative Algorithms**: Implement algorithms that require pause-and-resume capabilities.

## Related Design Patterns

### [State Pattern](./state-pattern)

- **Description**: Encapsulates varying behavior for the same object, depending on its state.
- **Relation to Coroutines**: Coroutines simplify state management by embedding state within the coroutine rather than maintaining it externally.

### [Iterator Pattern](./iterator-pattern)

- **Description**: Provides a way to access elements of a collection without exposing the underlying representation.
- **Relation to Coroutines**: Generators (a specific type of coroutine) naturally implement this pattern in a straightforward and compact manner.

### [Actor Model](./actor-model)

- **Description**: Concurrent computation model relying on message passing between actors.
- **Relation to Coroutines**: Coroutines can simplify writing code that fits into the actor model by yielding control when waiting for messages, thus improving responsiveness.

## Practical Implementations

### Example in JavaScript (using async/await)

```javascript
async function pauseAndResume() {
    console.log("Coroutine started");
    await new Promise(resolve => setTimeout(resolve, 1000));  // Simulate async operation
    console.log("Coroutine resumed after 1 second");
}

pauseAndResume();
// Output: 
// Coroutine started
// (after 1 second)
// Coroutine resumed after 1 second
```

### Advanced Usage in C#

```csharp
using System;
using System.Threading.Tasks;

public class CoroutineExample
{
    public static async Task Main()
    {
        await Coroutine();
    }

    public static async Task Coroutine()
    {
        Console.WriteLine("Coroutine started");
        await Task.Delay(1000);  // Simulate asynchronous work
        Console.WriteLine("Coroutine resumed after 1 second");
    }
}

```

## Summary

Coroutines provide a powerful mechanism for building non-preemptive multitasking environments. They allow for interruption and resumption of execution, making them ideal for asynchronous programming, state machines, and more. They simplify control flow and state management, especially compared to using threads or complex callback structures.

By embracing coroutines, developers can write more understandable and maintainable asynchronous code, avoiding common pitfalls like callback hell and complex state management.

## Additional Resources

- **Books**:
  - "Python Concurrency with asyncio" by Matthew Fowler
  - "Concurrency in C# Cookbook" by Stephen Cleary

- **Online References**:
  - [Python Official Documentation on Coroutines](https://docs.python.org/3/library/asyncio-task.html)
  - [JavaScript Async Functions](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function)

---

By understanding the intricate details of coroutines, you can unlock new possibilities for writing effective, clean, and maintainable concurrent programs in various programming languages.

