---
linkTitle: "5.5 Context for Cancellation"
title: "Context for Cancellation in Go: Managing Concurrency with Contexts"
description: "Explore how to effectively use context for cancellation in Go to manage concurrency, deadlines, and resource cleanup in goroutines."
categories:
- Concurrency
- Go Programming
- Software Design Patterns
tags:
- Go
- Concurrency
- Context
- Cancellation
- Best Practices
date: 2024-10-25
type: docs
nav_weight: 550000
canonical: "https://softwarepatternslexicon.com/patterns-go/5/5"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 5.5 Context for Cancellation

In modern software development, managing concurrency effectively is crucial for building responsive and efficient applications. Go, with its powerful concurrency model, provides the `context` package as a fundamental tool for managing deadlines, cancellations, and request-scoped data across goroutines. This section delves into the usage of context for cancellation, highlighting best practices and providing practical examples to illustrate its application.

### Introduction to Context in Go

The `context` package in Go is designed to carry deadlines, cancellation signals, and other request-scoped values across API boundaries and between goroutines. It is a key component in writing robust concurrent programs, allowing developers to manage the lifecycle of operations and ensure resources are cleaned up when operations are no longer needed.

### Context Usage

#### Passing `context.Context` Through Call Chains

In Go, it is a common practice to pass `context.Context` as the first parameter to functions that perform I/O operations or other long-running tasks. This allows the caller to control the execution of these functions, including the ability to cancel them if necessary.

```go
package main

import (
    "context"
    "fmt"
    "time"
)

func doWork(ctx context.Context) {
    select {
    case <-time.After(5 * time.Second):
        fmt.Println("Work completed")
    case <-ctx.Done():
        fmt.Println("Work cancelled:", ctx.Err())
    }
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    go doWork(ctx)

    time.Sleep(2 * time.Second)
    cancel() // Cancel the context to stop the work
    time.Sleep(1 * time.Second)
}
```

In this example, `doWork` is a function that simulates a long-running task. By passing a context to it, we can cancel the task after 2 seconds, demonstrating how context can be used to manage the lifecycle of goroutines.

#### Creating Cancellable Contexts

Go provides several functions to create contexts with cancellation capabilities:

- `context.WithCancel`: Returns a copy of the parent context with a new Done channel. The cancel function should be called to cancel the context.
- `context.WithTimeout`: Returns a copy of the parent context with a timeout. The context is automatically cancelled after the specified duration.
- `context.WithDeadline`: Similar to `WithTimeout`, but allows specifying an exact time for the context to be cancelled.

```go
package main

import (
    "context"
    "fmt"
    "time"
)

func doWorkWithTimeout(ctx context.Context) {
    select {
    case <-time.After(5 * time.Second):
        fmt.Println("Work completed")
    case <-ctx.Done():
        fmt.Println("Work cancelled:", ctx.Err())
    }
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
    defer cancel() // Ensure resources are cleaned up

    go doWorkWithTimeout(ctx)

    time.Sleep(4 * time.Second)
}
```

Here, `doWorkWithTimeout` uses a context with a timeout of 3 seconds. The work is cancelled automatically when the timeout is reached, demonstrating how contexts can manage time-bound operations.

### Cancellation Propagation

One of the key benefits of using context is its ability to propagate cancellation signals across goroutines. This ensures that when a parent context is cancelled, all derived contexts and their associated operations are also cancelled.

#### Ensuring Goroutines Check the Context

Goroutines should regularly check the context's Done channel to determine if they should exit early. This is crucial for ensuring that resources are not wasted on operations that are no longer needed.

```go
func doWorkWithCheck(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            fmt.Println("Goroutine cancelled:", ctx.Err())
            return
        default:
            fmt.Println("Working...")
            time.Sleep(500 * time.Millisecond)
        }
    }
}
```

In this example, the goroutine checks the context's Done channel in each iteration of the loop, allowing it to exit promptly when the context is cancelled.

#### Cleaning Up Resources

When a context is cancelled, it is important to clean up any resources that were allocated for the operation. This includes closing files, network connections, or any other resources that need explicit cleanup.

```go
func doWorkWithCleanup(ctx context.Context) {
    defer fmt.Println("Cleaning up resources")

    for {
        select {
        case <-ctx.Done():
            fmt.Println("Goroutine cancelled:", ctx.Err())
            return
        default:
            fmt.Println("Working...")
            time.Sleep(500 * time.Millisecond)
        }
    }
}
```

By using `defer`, we ensure that resources are cleaned up when the goroutine exits, whether due to completion or cancellation.

### Best Practices

#### Do Not Store Contexts Inside Structs

Contexts should not be stored inside structs or passed around as global variables. Instead, they should be passed explicitly to functions that need them. This ensures that the context is always relevant to the current operation and avoids unintended side effects.

#### Use Contexts for Request-Scoped Data

Contexts are ideal for carrying request-scoped data, such as authentication tokens or user IDs, across API boundaries. However, they should not be used for passing optional parameters or configuration settings.

```go
func processRequest(ctx context.Context) {
    userID := ctx.Value("userID")
    fmt.Println("Processing request for user:", userID)
}

func main() {
    ctx := context.WithValue(context.Background(), "userID", 42)
    processRequest(ctx)
}
```

In this example, the context carries a user ID that is relevant to the current request, demonstrating how contexts can be used to pass request-scoped data.

### Advantages and Disadvantages

#### Advantages

- **Simplifies Cancellation:** Contexts provide a unified way to manage cancellation across goroutines, simplifying code and reducing the risk of resource leaks.
- **Propagates Deadlines:** Contexts can carry deadlines across API boundaries, ensuring that timeouts are consistently enforced.
- **Encapsulates Request-Scoped Data:** Contexts allow passing request-scoped data without modifying function signatures to include additional parameters.

#### Disadvantages

- **Potential Misuse:** Using contexts for non-request-scoped data or storing them in structs can lead to code that is difficult to understand and maintain.
- **Overhead:** Passing contexts through call chains can introduce some overhead, particularly if contexts are used inappropriately.

### Conclusion

The `context` package is a powerful tool for managing concurrency in Go applications. By using contexts to handle cancellations and deadlines, developers can write more robust and efficient code. Following best practices, such as passing contexts explicitly and using them for request-scoped data, ensures that contexts are used effectively and appropriately.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of the `context` package in Go?

- [x] To manage deadlines and cancellations in concurrent operations
- [ ] To store global configuration settings
- [ ] To handle file I/O operations
- [ ] To manage database connections

> **Explanation:** The `context` package is primarily used to manage deadlines, cancellations, and request-scoped data in concurrent operations.

### Which function creates a context that is automatically cancelled after a specified duration?

- [ ] context.WithCancel
- [x] context.WithTimeout
- [ ] context.WithValue
- [ ] context.Background

> **Explanation:** `context.WithTimeout` creates a context that is automatically cancelled after the specified duration.

### How should contexts be passed in Go functions?

- [x] As the first parameter to functions
- [ ] As a global variable
- [ ] Inside a struct
- [ ] As an optional parameter

> **Explanation:** Contexts should be passed as the first parameter to functions to ensure they are relevant to the current operation.

### What should goroutines do to handle context cancellation?

- [x] Regularly check the context's Done channel
- [ ] Ignore the context
- [ ] Store the context in a global variable
- [ ] Use context.WithValue

> **Explanation:** Goroutines should regularly check the context's Done channel to determine if they should exit early.

### What is a disadvantage of misusing contexts?

- [x] Code becomes difficult to understand and maintain
- [ ] It improves performance
- [ ] It simplifies code
- [ ] It reduces resource usage

> **Explanation:** Misusing contexts, such as storing them in structs or using them for non-request-scoped data, can lead to code that is difficult to understand and maintain.

### What is a best practice when using contexts in Go?

- [x] Pass contexts explicitly to functions
- [ ] Store contexts in structs
- [ ] Use contexts for optional parameters
- [ ] Use contexts as global variables

> **Explanation:** A best practice is to pass contexts explicitly to functions to ensure they are relevant to the current operation.

### What is the effect of calling the cancel function returned by `context.WithCancel`?

- [x] It cancels the context and all derived contexts
- [ ] It creates a new context
- [ ] It extends the context's deadline
- [ ] It has no effect

> **Explanation:** Calling the cancel function cancels the context and all derived contexts, propagating the cancellation signal.

### What should be done when a context is cancelled?

- [x] Clean up resources and exit the goroutine
- [ ] Continue executing the goroutine
- [ ] Ignore the cancellation
- [ ] Restart the operation

> **Explanation:** When a context is cancelled, resources should be cleaned up, and the goroutine should exit to prevent resource leaks.

### Which function is used to create a context with a specific deadline?

- [ ] context.WithCancel
- [ ] context.WithTimeout
- [x] context.WithDeadline
- [ ] context.Background

> **Explanation:** `context.WithDeadline` is used to create a context with a specific deadline.

### True or False: Contexts should be used to pass optional parameters in Go.

- [ ] True
- [x] False

> **Explanation:** Contexts should not be used to pass optional parameters; they are intended for request-scoped data and cancellation signals.

{{< /quizdown >}}
