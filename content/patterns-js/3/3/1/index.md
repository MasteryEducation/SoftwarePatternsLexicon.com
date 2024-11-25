---

linkTitle: "3.3.1 Function Decorators"
title: "Function Decorators in JavaScript and TypeScript: Enhance Functionality with Ease"
description: "Explore the use of function decorators in JavaScript and TypeScript to extend function behavior without altering their code. Learn implementation steps, use cases, and best practices."
categories:
- JavaScript
- TypeScript
- Design Patterns
tags:
- Function Decorators
- JavaScript Patterns
- TypeScript Patterns
- Code Optimization
- Software Design
date: 2024-10-25
type: docs
nav_weight: 331000
canonical: "https://softwarepatternslexicon.com/patterns-js/3/3/1"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 3.3.1 Function Decorators

Function decorators are a powerful pattern in JavaScript and TypeScript that allow developers to extend the behavior of functions without modifying their original code. This pattern is particularly useful for adding cross-cutting concerns such as logging, caching, or validation.

### Understand the Intent

The primary intent of function decorators is to wrap functions to extend their behavior seamlessly. This approach promotes code reusability and separation of concerns, making it easier to maintain and scale applications.

### Implementation Steps

To implement a function decorator, follow these steps:

1. **Define a Decorator Function:** Create a function that accepts another function as an argument.
2. **Return a New Function:** The decorator should return a new function that adds the desired functionality before or after executing the original function.

### Code Examples

Let's explore some practical examples of function decorators in JavaScript and TypeScript.

#### Example 1: Logging Execution Time

This decorator logs the execution time of a function, which is useful for performance monitoring.

```typescript
function logExecutionTime<T extends (...args: any[]) => any>(fn: T): T {
    return function(...args: Parameters<T>): ReturnType<T> {
        const start = performance.now();
        const result = fn(...args);
        const end = performance.now();
        console.log(`Execution time: ${end - start}ms`);
        return result;
    } as T;
}

// Usage
const slowFunction = (num: number) => {
    for (let i = 0; i < num; i++) {}
    return num;
};

const decoratedFunction = logExecutionTime(slowFunction);
decoratedFunction(1000000);
```

#### Example 2: Error Handling and Retry

This decorator handles errors and retries the function a specified number of times.

```typescript
function retry<T extends (...args: any[]) => any>(fn: T, retries: number = 3): T {
    return function(...args: Parameters<T>): ReturnType<T> {
        let attempts = 0;
        while (attempts < retries) {
            try {
                return fn(...args);
            } catch (error) {
                attempts++;
                console.log(`Attempt ${attempts} failed. Retrying...`);
            }
        }
        throw new Error(`Function failed after ${retries} attempts`);
    } as T;
}

// Usage
const unreliableFunction = (shouldFail: boolean) => {
    if (shouldFail) throw new Error("Failed!");
    return "Success!";
};

const safeFunction = retry(unreliableFunction, 3);
console.log(safeFunction(false)); // Success!
```

### Use Cases

Function decorators are ideal for scenarios where you need to add cross-cutting concerns, such as:

- **Logging:** Track function calls and execution times.
- **Caching:** Store results of expensive function calls to improve performance.
- **Validation:** Ensure input data meets certain criteria before processing.
- **Error Handling:** Gracefully handle errors and implement retry logic.

### Practice

Try writing a decorator that caches the results of an expensive function call:

```typescript
function cache<T extends (...args: any[]) => any>(fn: T): T {
    const cacheMap = new Map<string, ReturnType<T>>();
    return function(...args: Parameters<T>): ReturnType<T> {
        const key = JSON.stringify(args);
        if (cacheMap.has(key)) {
            console.log("Returning cached result");
            return cacheMap.get(key) as ReturnType<T>;
        }
        const result = fn(...args);
        cacheMap.set(key, result);
        return result;
    } as T;
}

// Usage
const expensiveFunction = (num: number) => {
    console.log("Computing...");
    return num * num;
};

const cachedFunction = cache(expensiveFunction);
console.log(cachedFunction(5)); // Computing... 25
console.log(cachedFunction(5)); // Returning cached result 25
```

### Considerations

When implementing function decorators, keep the following in mind:

- **Preserve Context and Arguments:** Ensure the original function's context (`this`) and arguments are preserved when calling it within the decorator.
- **Asynchronous Functions:** Be cautious with async functions, as decorators may need to handle promises or async/await syntax appropriately.

### Best Practices

- **SOLID Principles:** Adhere to the Single Responsibility Principle by ensuring each decorator has a single purpose.
- **Code Maintainability:** Keep decorators simple and focused to enhance code readability and maintainability.
- **Testing:** Thoroughly test decorators to ensure they work correctly with various function signatures and edge cases.

### Conclusion

Function decorators in JavaScript and TypeScript provide a flexible way to extend function behavior without altering their code. By understanding their intent and implementation, you can effectively apply decorators to add cross-cutting concerns like logging, caching, and error handling. Remember to preserve the original function's context and arguments, especially when dealing with asynchronous operations.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of function decorators?

- [x] To wrap functions and extend their behavior without modifying their code.
- [ ] To replace functions with new implementations.
- [ ] To convert synchronous functions to asynchronous ones.
- [ ] To optimize function performance by default.

> **Explanation:** Function decorators are designed to wrap functions and extend their behavior without altering the original code, promoting code reusability and separation of concerns.

### Which of the following is a common use case for function decorators?

- [x] Logging
- [ ] Compiling
- [x] Caching
- [ ] Debugging

> **Explanation:** Function decorators are commonly used for logging and caching, among other cross-cutting concerns like validation and error handling.

### How do you ensure the original function's context is preserved in a decorator?

- [x] Use the `apply` or `call` method to invoke the original function.
- [ ] Use a global variable to store the context.
- [ ] Use a closure to capture the context.
- [ ] Use a class to encapsulate the function.

> **Explanation:** The `apply` or `call` method can be used to invoke the original function with the correct context (`this`), ensuring it behaves as expected.

### What should a decorator return?

- [x] A new function that wraps the original function.
- [ ] The original function with added properties.
- [ ] A promise that resolves to the original function.
- [ ] A class instance representing the function.

> **Explanation:** A decorator should return a new function that wraps the original function, allowing for additional behavior to be added.

### What is a potential issue when using decorators with asynchronous functions?

- [x] Handling promises or async/await syntax appropriately.
- [ ] Losing the original function's name.
- [ ] Increasing the function's execution time.
- [ ] Decreasing the function's readability.

> **Explanation:** When dealing with asynchronous functions, decorators need to handle promises or async/await syntax correctly to ensure proper execution flow.

### Which SOLID principle is most relevant to function decorators?

- [x] Single Responsibility Principle
- [ ] Open/Closed Principle
- [ ] Liskov Substitution Principle
- [ ] Interface Segregation Principle

> **Explanation:** The Single Responsibility Principle is relevant because each decorator should have a single, focused purpose to maintain simplicity and clarity.

### What is the benefit of caching results in a decorator?

- [x] It improves performance by avoiding repeated computations.
- [ ] It increases the function's execution time.
- [ ] It makes the function more readable.
- [ ] It ensures the function always returns the same result.

> **Explanation:** Caching results in a decorator can significantly improve performance by avoiding repeated computations for the same input.

### How can you test a function decorator effectively?

- [x] By testing it with various function signatures and edge cases.
- [ ] By only testing it with synchronous functions.
- [ ] By using it in production code without prior testing.
- [ ] By assuming it works if the original function works.

> **Explanation:** To ensure reliability, decorators should be tested with various function signatures and edge cases to verify their behavior in different scenarios.

### What is a key consideration when implementing a retry decorator?

- [x] The number of retry attempts and the handling of errors.
- [ ] The function's return type.
- [ ] The function's execution time.
- [ ] The function's name.

> **Explanation:** When implementing a retry decorator, it's crucial to define the number of retry attempts and handle errors appropriately to ensure robustness.

### True or False: Function decorators can only be used with synchronous functions.

- [ ] True
- [x] False

> **Explanation:** Function decorators can be used with both synchronous and asynchronous functions, although special care is needed to handle async operations correctly.

{{< /quizdown >}}
