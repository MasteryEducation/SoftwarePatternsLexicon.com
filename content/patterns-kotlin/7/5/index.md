---
canonical: "https://softwarepatternslexicon.com/patterns-kotlin/7/5"
title: "Partial Application and Currying in Kotlin: Mastering Functional Design Patterns"
description: "Explore the concepts of Partial Application and Currying in Kotlin, learn how to create new functions with fixed arguments, and implement currying effectively."
linkTitle: "7.5 Partial Application and Currying"
categories:
- Kotlin
- Functional Programming
- Design Patterns
tags:
- Partial Application
- Currying
- Kotlin
- Functional Design Patterns
- Higher-Order Functions
date: 2024-11-17
type: docs
nav_weight: 7500
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 7.5 Partial Application and Currying

In the realm of functional programming, **Partial Application** and **Currying** are two powerful concepts that allow us to create more flexible and reusable code. By understanding and implementing these patterns in Kotlin, we can enhance our ability to write concise and expressive functions. Let's dive into these concepts and explore how they can be applied in Kotlin.

### Understanding Partial Application

**Partial Application** refers to the process of fixing a few arguments of a function, producing another function of smaller arity. This means that you can take a function that requires multiple arguments and create a new function that requires fewer arguments by pre-filling some of the arguments.

#### Key Concepts

- **Arity**: The number of arguments a function takes.
- **Partially Applied Function**: A function that has some of its arguments fixed.

#### Example of Partial Application

Consider a simple function that adds three numbers:

```kotlin
fun addThreeNumbers(a: Int, b: Int, c: Int): Int {
    return a + b + c
}
```

Using partial application, we can create a new function that fixes the first argument:

```kotlin
fun addTwoNumbers(b: Int, c: Int): Int {
    return addThreeNumbers(5, b, c)
}
```

Here, `addTwoNumbers` is a partially applied version of `addThreeNumbers` with the first argument set to `5`.

#### Implementing Partial Application in Kotlin

To implement partial application in Kotlin, we can use higher-order functions. Let's create a generic function that allows partial application:

```kotlin
fun <A, B, C> partialApplication(a: A, function: (A, B) -> C): (B) -> C {
    return { b: B -> function(a, b) }
}
```

Now, let's see how we can use this function:

```kotlin
fun multiply(x: Int, y: Int): Int = x * y

val multiplyBy2 = partialApplication(2, ::multiply)

fun main() {
    println(multiplyBy2(5)) // Output: 10
}
```

In this example, `multiplyBy2` is a partially applied function that multiplies any given number by `2`.

### Visualizing Partial Application

To better understand how partial application works, let's visualize it using a diagram:

```mermaid
graph TD;
    A[Original Function: addThreeNumbers(a, b, c)] --> B[Partially Applied Function: addTwoNumbers(b, c)]
    B --> C[Result: a + b + c]
    A --> D[Fixed Argument: a = 5]
```

This diagram illustrates how the original function `addThreeNumbers` is transformed into a partially applied function `addTwoNumbers` by fixing the first argument.

### Exploring Currying

**Currying** is the process of transforming a function that takes multiple arguments into a sequence of functions, each taking a single argument. This technique allows us to break down a function into smaller, more manageable pieces.

#### Key Concepts

- **Curried Function**: A function that returns another function as its result.
- **Sequential Application**: Applying arguments one at a time.

#### Example of Currying

Let's revisit the `addThreeNumbers` function and transform it into a curried version:

```kotlin
fun curriedAdd(a: Int): (Int) -> (Int) -> Int {
    return { b: Int -> { c: Int -> a + b + c } }
}
```

Now, we can use this curried function as follows:

```kotlin
val addFive = curriedAdd(5)
val addFiveAndTwo = addFive(2)
val result = addFiveAndTwo(3)

fun main() {
    println(result) // Output: 10
}
```

In this example, `curriedAdd` returns a series of functions, each taking one argument.

#### Implementing Currying in Kotlin

To implement currying in Kotlin, we can create a utility function that transforms a regular function into a curried function:

```kotlin
fun <A, B, C> curry(function: (A, B) -> C): (A) -> (B) -> C {
    return { a: A -> { b: B -> function(a, b) } }
}
```

Let's see how this utility function can be used:

```kotlin
fun subtract(x: Int, y: Int): Int = x - y

val curriedSubtract = curry(::subtract)

fun main() {
    val subtractFrom10 = curriedSubtract(10)
    println(subtractFrom10(3)) // Output: 7
}
```

In this example, `curriedSubtract` is a curried version of the `subtract` function.

### Visualizing Currying

To visualize currying, let's use a diagram:

```mermaid
graph TD;
    A[Original Function: subtract(x, y)] --> B[Curried Function: curriedSubtract(x)]
    B --> C[Function: subtractFrom10(y)]
    C --> D[Result: x - y]
    A --> E[Sequential Application: x = 10, y = 3]
```

This diagram shows how the original function `subtract` is transformed into a curried function `curriedSubtract`, allowing sequential application of arguments.

### Differences Between Partial Application and Currying

While both partial application and currying involve transforming functions, they serve different purposes:

- **Partial Application**: Fixes some arguments, creating a new function with fewer arguments.
- **Currying**: Transforms a function into a series of functions, each taking one argument.

### Practical Applications

Partial application and currying can be used in various scenarios to improve code reusability and readability:

- **Configuration**: Create functions with default configurations that can be overridden.
- **Event Handling**: Partially apply event handlers with specific contexts.
- **Pipelines**: Build data processing pipelines with curried functions.

### Try It Yourself

Experiment with the following exercises to deepen your understanding:

1. **Modify the `partialApplication` function** to support functions with three arguments.
2. **Create a curried version** of a function that calculates the volume of a box given its length, width, and height.
3. **Use partial application** to create a function that formats strings with a specific prefix.

### Design Considerations

When using partial application and currying, consider the following:

- **Readability**: Ensure that the resulting functions are easy to understand.
- **Performance**: Be mindful of the overhead introduced by creating additional functions.
- **Kotlin Features**: Leverage Kotlin's higher-order functions and lambda expressions to simplify implementation.

### Differences and Similarities

Partial application and currying are often confused, but they have distinct characteristics:

- **Partial Application**: Focuses on fixing arguments.
- **Currying**: Focuses on transforming functions into a series of unary functions.

### Conclusion

Partial application and currying are powerful techniques in functional programming that can greatly enhance the flexibility and expressiveness of your code. By mastering these concepts in Kotlin, you can create more modular and reusable functions, leading to cleaner and more maintainable codebases.

Remember, this is just the beginning. As you progress, you'll discover new ways to apply these techniques in your projects. Keep experimenting, stay curious, and enjoy the journey!

## Quiz Time!

{{< quizdown >}}

### What is Partial Application?

- [x] Fixing a few arguments of a function, producing another function of smaller arity.
- [ ] Transforming a function into a series of functions, each taking a single argument.
- [ ] A method of optimizing function calls.
- [ ] A way to handle exceptions in functional programming.

> **Explanation:** Partial application involves fixing some arguments of a function, creating a new function with fewer arguments.

### What is Currying?

- [ ] Fixing a few arguments of a function, producing another function of smaller arity.
- [x] Transforming a function into a series of functions, each taking a single argument.
- [ ] A method of optimizing function calls.
- [ ] A way to handle exceptions in functional programming.

> **Explanation:** Currying transforms a function into a sequence of functions, each taking one argument.

### How can you implement partial application in Kotlin?

- [x] By using higher-order functions to fix some arguments of a function.
- [ ] By using reflection to modify function signatures.
- [ ] By using Kotlin's `inline` functions.
- [ ] By using Kotlin's `data` classes.

> **Explanation:** Partial application can be implemented using higher-order functions to fix some arguments of a function.

### What is the result of applying currying to a function?

- [x] A sequence of functions, each taking a single argument.
- [ ] A single function with all arguments fixed.
- [ ] A function that returns a list of results.
- [ ] A function that handles exceptions.

> **Explanation:** Currying results in a sequence of functions, each taking one argument.

### Which of the following is a key difference between partial application and currying?

- [x] Partial application fixes some arguments, while currying transforms a function into unary functions.
- [ ] Partial application is only applicable in Kotlin, while currying is universal.
- [ ] Currying is a subset of partial application.
- [ ] Partial application requires more memory than currying.

> **Explanation:** Partial application fixes some arguments, while currying transforms a function into a series of unary functions.

### How can currying improve code readability?

- [x] By breaking down complex functions into smaller, more manageable pieces.
- [ ] By reducing the number of lines of code.
- [ ] By eliminating the need for comments.
- [ ] By using reflection to simplify function signatures.

> **Explanation:** Currying improves readability by breaking down complex functions into smaller, more manageable pieces.

### What is the purpose of using higher-order functions in partial application?

- [x] To create new functions with fixed arguments.
- [ ] To optimize performance.
- [ ] To handle exceptions.
- [ ] To simplify class hierarchies.

> **Explanation:** Higher-order functions are used in partial application to create new functions with fixed arguments.

### What is a curried function?

- [x] A function that returns another function as its result.
- [ ] A function that handles exceptions.
- [ ] A function that optimizes performance.
- [ ] A function that fixes some arguments.

> **Explanation:** A curried function returns another function as its result.

### Which Kotlin feature is commonly used to implement currying?

- [x] Higher-order functions and lambda expressions.
- [ ] Reflection and metaprogramming.
- [ ] Inline functions.
- [ ] Data classes.

> **Explanation:** Higher-order functions and lambda expressions are commonly used to implement currying in Kotlin.

### True or False: Currying and partial application are the same concepts.

- [ ] True
- [x] False

> **Explanation:** Currying and partial application are distinct concepts; currying transforms a function into unary functions, while partial application fixes some arguments.

{{< /quizdown >}}
