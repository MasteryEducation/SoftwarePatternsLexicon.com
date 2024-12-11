---
canonical: "https://softwarepatternslexicon.com/patterns-java/9/7/3"

title: "Asynchronous Computations with `CompletableFuture` in Java"
description: "Explore the power of `CompletableFuture` for asynchronous programming in Java, leveraging functional programming patterns to enhance code efficiency and readability."
linkTitle: "9.7.3 `CompletableFuture` for Asynchronous Computations"
tags:
- "Java"
- "CompletableFuture"
- "Asynchronous Programming"
- "Functional Programming"
- "Concurrency"
- "Java 8"
- "Monads"
- "Best Practices"
date: 2024-11-25
type: docs
nav_weight: 97300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 9.7.3 `CompletableFuture` for Asynchronous Computations

### Introduction to `CompletableFuture`

In the realm of Java programming, handling asynchronous computations has always been a challenging task. With the introduction of `CompletableFuture` in Java 8, developers gained a powerful tool to manage asynchronous tasks more effectively. `CompletableFuture` is a part of the `java.util.concurrent` package and serves as a monad-like construct that simplifies the process of writing non-blocking code. It allows developers to create complex asynchronous pipelines with ease, leveraging functional programming paradigms.

### The Role of `CompletableFuture` in Asynchronous Programming

`CompletableFuture` is designed to represent a future result of an asynchronous computation. It provides a comprehensive API that allows developers to chain and compose asynchronous operations seamlessly. By using `CompletableFuture`, you can write code that is both efficient and easy to understand, avoiding the traditional callback hell associated with asynchronous programming.

#### Key Features of `CompletableFuture`

- **Chaining Operations**: `CompletableFuture` allows you to chain multiple asynchronous operations using methods like `thenApply`, `thenCompose`, and `thenAccept`.
- **Combining Futures**: You can combine multiple `CompletableFuture` instances using methods such as `thenCombine` and `allOf`.
- **Exception Handling**: It provides robust exception handling mechanisms with methods like `handle`, `exceptionally`, and `whenComplete`.
- **Non-blocking**: `CompletableFuture` operations are non-blocking, enabling efficient use of system resources.

### Chaining and Composing Asynchronous Operations

One of the most powerful features of `CompletableFuture` is its ability to chain and compose asynchronous operations. This is achieved through a fluent API that allows you to define a sequence of transformations and actions on the result of a computation.

#### Using `thenApply` and `thenCompose`

- **`thenApply`**: This method is used to apply a function to the result of a `CompletableFuture` once it completes. It is suitable for synchronous transformations.

```java
CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> 42)
    .thenApply(result -> result * 2);

future.thenAccept(System.out::println); // Outputs: 84
```

- **`thenCompose`**: This method is used to chain another `CompletableFuture` based on the result of the current one. It is ideal for asynchronous transformations.

```java
CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> 42)
    .thenCompose(result -> CompletableFuture.supplyAsync(() -> result * 2));

future.thenAccept(System.out::println); // Outputs: 84
```

#### Handling Results with `thenAccept` and `thenRun`

- **`thenAccept`**: Consumes the result of a `CompletableFuture` without returning a new future.

```java
CompletableFuture.supplyAsync(() -> "Hello, World!")
    .thenAccept(System.out::println); // Outputs: Hello, World!
```

- **`thenRun`**: Executes a runnable after the `CompletableFuture` completes, without using its result.

```java
CompletableFuture.supplyAsync(() -> 42)
    .thenRun(() -> System.out.println("Computation finished"));
```

### Exception Handling in Asynchronous Code

Handling exceptions in asynchronous code can be tricky, but `CompletableFuture` provides several methods to manage exceptions gracefully.

#### Using `handle`, `exceptionally`, and `whenComplete`

- **`handle`**: This method allows you to handle both the result and the exception of a `CompletableFuture`.

```java
CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> {
    if (Math.random() > 0.5) throw new RuntimeException("Failed");
    return 42;
}).handle((result, ex) -> {
    if (ex != null) {
        System.out.println("Exception: " + ex.getMessage());
        return 0;
    }
    return result;
});

future.thenAccept(System.out::println);
```

- **`exceptionally`**: This method provides a way to handle exceptions and return a default value.

```java
CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> {
    if (Math.random() > 0.5) throw new RuntimeException("Failed");
    return 42;
}).exceptionally(ex -> {
    System.out.println("Exception: " + ex.getMessage());
    return 0;
});

future.thenAccept(System.out::println);
```

- **`whenComplete`**: This method allows you to perform an action after the `CompletableFuture` completes, regardless of whether it completed normally or exceptionally.

```java
CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> {
    if (Math.random() > 0.5) throw new RuntimeException("Failed");
    return 42;
}).whenComplete((result, ex) -> {
    if (ex != null) {
        System.out.println("Exception: " + ex.getMessage());
    } else {
        System.out.println("Result: " + result);
    }
});
```

### Simplifying Callback Management

`CompletableFuture` significantly simplifies callback management by providing a structured way to define asynchronous workflows. Instead of nesting callbacks, you can define a clear sequence of operations, improving code readability and maintainability.

#### Example: Asynchronous Computation Pipeline

Consider a scenario where you need to fetch data from a remote service, process it, and then store the result in a database. Using `CompletableFuture`, you can create a clean and efficient pipeline for this task.

```java
CompletableFuture.supplyAsync(() -> fetchDataFromService())
    .thenApply(data -> processData(data))
    .thenAccept(result -> storeResultInDatabase(result))
    .exceptionally(ex -> {
        System.out.println("Error occurred: " + ex.getMessage());
        return null;
    });
```

### Performance Considerations and Thread Management

While `CompletableFuture` provides a powerful abstraction for asynchronous programming, it is essential to consider performance and thread management to ensure efficient execution.

#### Default Executor and Custom Executors

By default, `CompletableFuture` uses the `ForkJoinPool.commonPool()` for executing tasks. However, for better control over thread management, you can provide a custom executor.

```java
ExecutorService executor = Executors.newFixedThreadPool(10);

CompletableFuture.supplyAsync(() -> 42, executor)
    .thenApplyAsync(result -> result * 2, executor)
    .thenAcceptAsync(System.out::println, executor);

executor.shutdown();
```

#### Best Practices for Performance Optimization

- **Avoid Blocking Operations**: Ensure that the tasks executed by `CompletableFuture` are non-blocking to prevent thread starvation.
- **Use Appropriate Pool Sizes**: Configure thread pools based on the nature of your tasks and the available system resources.
- **Monitor and Tune**: Regularly monitor the performance of your asynchronous tasks and tune the thread pool settings as needed.

### Conclusion

`CompletableFuture` is a versatile and powerful tool for managing asynchronous computations in Java. By leveraging its functional programming capabilities, developers can create efficient and maintainable code that handles complex asynchronous workflows with ease. Understanding how to use `CompletableFuture` effectively is crucial for modern Java developers, enabling them to build responsive and scalable applications.

### Key Takeaways

- `CompletableFuture` simplifies asynchronous programming by providing a fluent API for chaining and composing tasks.
- It offers robust exception handling mechanisms, improving the reliability of asynchronous code.
- Proper thread management and performance considerations are essential for optimizing the use of `CompletableFuture`.

### Encouragement for Further Exploration

Experiment with `CompletableFuture` in your projects to explore its full potential. Consider how you can apply its features to improve the responsiveness and scalability of your applications. Reflect on the patterns and best practices discussed in this guide to enhance your understanding of asynchronous programming in Java.

---

## Test Your Knowledge: Asynchronous Programming with `CompletableFuture` Quiz

{{< quizdown >}}

### What is the primary purpose of `CompletableFuture` in Java?

- [x] To handle asynchronous computations efficiently.
- [ ] To manage database connections.
- [ ] To simplify file I/O operations.
- [ ] To improve GUI rendering.

> **Explanation:** `CompletableFuture` is designed to handle asynchronous computations, providing a comprehensive API for chaining and composing tasks.

### Which method is used to apply a function to the result of a `CompletableFuture`?

- [x] thenApply
- [ ] thenCompose
- [ ] thenAccept
- [ ] handle

> **Explanation:** `thenApply` is used to apply a function to the result of a `CompletableFuture` once it completes.

### How does `thenCompose` differ from `thenApply`?

- [x] `thenCompose` is used for chaining asynchronous operations.
- [ ] `thenCompose` is used for synchronous transformations.
- [ ] `thenCompose` handles exceptions.
- [ ] `thenCompose` is used for combining futures.

> **Explanation:** `thenCompose` is used to chain another `CompletableFuture` based on the result of the current one, ideal for asynchronous transformations.

### Which method provides a way to handle exceptions in `CompletableFuture`?

- [x] exceptionally
- [ ] thenApply
- [ ] thenRun
- [ ] thenCombine

> **Explanation:** `exceptionally` provides a way to handle exceptions and return a default value in case of an error.

### What is the default executor used by `CompletableFuture`?

- [x] ForkJoinPool.commonPool()
- [ ] Executors.newCachedThreadPool()
- [ ] Executors.newFixedThreadPool()
- [ ] Executors.newSingleThreadExecutor()

> **Explanation:** By default, `CompletableFuture` uses the `ForkJoinPool.commonPool()` for executing tasks.

### Which method allows you to perform an action after a `CompletableFuture` completes, regardless of its outcome?

- [x] whenComplete
- [ ] thenApply
- [ ] thenCompose
- [ ] thenAccept

> **Explanation:** `whenComplete` allows you to perform an action after the `CompletableFuture` completes, regardless of whether it completed normally or exceptionally.

### What is a key benefit of using `CompletableFuture` over traditional callbacks?

- [x] It simplifies callback management and improves code readability.
- [ ] It increases the execution speed of tasks.
- [ ] It reduces memory usage.
- [ ] It automatically handles all exceptions.

> **Explanation:** `CompletableFuture` simplifies callback management by providing a structured way to define asynchronous workflows, improving code readability and maintainability.

### How can you provide a custom executor to a `CompletableFuture`?

- [x] By passing the executor as a parameter to supplyAsync or thenApplyAsync.
- [ ] By setting a global executor for all futures.
- [ ] By using a static method on `CompletableFuture`.
- [ ] By overriding the default executor in the JVM settings.

> **Explanation:** You can provide a custom executor by passing it as a parameter to methods like `supplyAsync` or `thenApplyAsync`.

### What should you avoid in tasks executed by `CompletableFuture` to prevent thread starvation?

- [x] Blocking operations
- [ ] Asynchronous operations
- [ ] Exception handling
- [ ] Logging

> **Explanation:** Avoid blocking operations in tasks executed by `CompletableFuture` to prevent thread starvation and ensure efficient use of system resources.

### True or False: `CompletableFuture` can only be used for asynchronous computations.

- [x] True
- [ ] False

> **Explanation:** `CompletableFuture` is specifically designed for handling asynchronous computations, providing a comprehensive API for managing such tasks.

{{< /quizdown >}}

---
