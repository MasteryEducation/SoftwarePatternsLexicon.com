---
canonical: "https://softwarepatternslexicon.com/patterns-erlang/2/2"
title: "First-Class and Higher-Order Functions in Erlang"
description: "Explore the power of first-class and higher-order functions in Erlang, and learn how they enable functional composition and code abstraction."
linkTitle: "2.2 First-Class and Higher-Order Functions"
categories:
- Functional Programming
- Erlang
- Software Design Patterns
tags:
- Erlang
- Functional Programming
- Higher-Order Functions
- First-Class Functions
- Code Abstraction
date: 2024-11-23
type: docs
nav_weight: 22000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.2 First-Class and Higher-Order Functions

In this section, we delve into the concepts of first-class and higher-order functions in Erlang, exploring how these features empower developers to write more abstract, reusable, and expressive code. By understanding these concepts, you will be able to leverage Erlang's functional programming capabilities to their fullest potential.

### Understanding First-Class Functions

In programming, when we say that functions are "first-class citizens," we mean that they are treated like any other data type. This means functions can be assigned to variables, passed as arguments to other functions, and returned as values from other functions. Erlang, being a functional language, treats functions as first-class citizens, which is a cornerstone of its functional programming paradigm.

#### Key Characteristics of First-Class Functions

1. **Assignability**: Functions can be assigned to variables.
2. **Passability**: Functions can be passed as arguments to other functions.
3. **Returnability**: Functions can be returned from other functions.

Let's illustrate these characteristics with some Erlang code examples.

#### Example: Assigning Functions to Variables

```erlang
% Define a simple function that adds two numbers
Add = fun(X, Y) -> X + Y end.

% Use the function by calling the variable
Result = Add(5, 3).
% Result is now 8
```

In this example, we define an anonymous function using the `fun` keyword and assign it to the variable `Add`. We can then use `Add` as if it were a function itself.

#### Example: Passing Functions as Arguments

```erlang
% Define a function that takes another function as an argument
apply_twice(F, X) ->
    F(F(X)).

% Define a function to be passed
Double = fun(X) -> X * 2 end.

% Apply the function twice
Result = apply_twice(Double, 3).
% Result is now 12
```

Here, `apply_twice` is a higher-order function that takes another function `F` and a value `X`, applying `F` to `X` twice. We pass the `Double` function to `apply_twice`, demonstrating how functions can be passed as arguments.

#### Example: Returning Functions from Other Functions

```erlang
% Define a function that returns another function
make_multiplier(Factor) ->
    fun(X) -> X * Factor end.

% Create a new function that triples a number
Triple = make_multiplier(3).

% Use the new function
Result = Triple(4).
% Result is now 12
```

In this example, `make_multiplier` returns a new function that multiplies its input by a given `Factor`. We use this to create a `Triple` function that multiplies by 3.

### Higher-Order Functions

Higher-order functions are functions that can take other functions as arguments or return them as results. They are a powerful tool for abstraction and code reuse, allowing you to create more generic and flexible code.

#### Benefits of Higher-Order Functions

1. **Code Reuse**: By abstracting common patterns into higher-order functions, you can reuse code across different parts of your application.
2. **Modularity**: Higher-order functions promote modularity by separating concerns and encapsulating behavior.
3. **Expressiveness**: They allow you to express complex operations succinctly and clearly.

#### Common Higher-Order Functions in Erlang

Erlang provides several built-in higher-order functions, particularly in the `lists` module, which are commonly used for list processing.

- **`lists:map/2`**: Applies a function to each element of a list.
- **`lists:filter/2`**: Filters elements of a list based on a predicate function.
- **`lists:foldl/3`**: Accumulates a result by applying a function to each element of a list, starting from the left.

#### Example: Using `lists:map/2`

```erlang
% Define a function to square a number
Square = fun(X) -> X * X end.

% Use lists:map to apply Square to each element of a list
SquaredList = lists:map(Square, [1, 2, 3, 4]).
% SquaredList is now [1, 4, 9, 16]
```

In this example, `lists:map/2` applies the `Square` function to each element of the list `[1, 2, 3, 4]`, resulting in a new list of squared numbers.

#### Example: Using `lists:filter/2`

```erlang
% Define a predicate function to check if a number is even
IsEven = fun(X) -> X rem 2 == 0 end.

% Use lists:filter to select even numbers from a list
EvenNumbers = lists:filter(IsEven, [1, 2, 3, 4, 5, 6]).
% EvenNumbers is now [2, 4, 6]
```

Here, `lists:filter/2` uses the `IsEven` predicate to filter out odd numbers, leaving only the even ones.

#### Example: Using `lists:foldl/3`

```erlang
% Define a function to sum two numbers
Sum = fun(X, Acc) -> X + Acc end.

% Use lists:foldl to sum all elements of a list
Total = lists:foldl(Sum, 0, [1, 2, 3, 4]).
% Total is now 10
```

In this example, `lists:foldl/3` accumulates the sum of the list `[1, 2, 3, 4]`, starting with an initial accumulator value of `0`.

### Visualizing Higher-Order Functions

To better understand how higher-order functions work, let's visualize the process of applying a function to each element of a list using a flowchart.

```mermaid
graph TD;
    A[Start] --> B[Function: Square];
    B --> C[Input List: [1, 2, 3, 4]];
    C --> D[Apply Square to each element];
    D --> E[Output List: [1, 4, 9, 16]];
    E --> F[End];
```

**Figure 1**: This flowchart illustrates the process of using `lists:map/2` to apply the `Square` function to each element of a list.

### Practical Applications of Higher-Order Functions

Higher-order functions are not just theoretical constructs; they have practical applications in real-world programming. Here are a few scenarios where they can be particularly useful:

1. **Data Transformation**: Use higher-order functions to transform data structures, such as converting a list of strings to uppercase.
2. **Event Handling**: Implement event handlers that can be dynamically composed and modified.
3. **Configuration**: Create flexible configuration systems where behavior can be altered by passing different functions.

### Try It Yourself

To solidify your understanding of first-class and higher-order functions, try modifying the examples provided:

- Change the `Square` function to cube each number instead.
- Create a new higher-order function that applies a given function three times to an input.
- Use `lists:foldl/3` to find the product of a list of numbers.

### Key Takeaways

- **First-Class Functions**: Erlang treats functions as first-class citizens, allowing them to be assigned to variables, passed as arguments, and returned from other functions.
- **Higher-Order Functions**: These functions take other functions as arguments or return them as results, enabling powerful abstractions and code reuse.
- **Practical Use**: Higher-order functions are widely used in Erlang for list processing, event handling, and more.

### References and Further Reading

- [Erlang Documentation on Functions](https://www.erlang.org/doc/reference_manual/functions.html)
- [Learn You Some Erlang for Great Good!](http://learnyousomeerlang.com/)
- [Erlang Programming: A Concurrent Approach to Software Development](https://www.oreilly.com/library/view/erlang-programming/9780596518189/)

Remember, mastering first-class and higher-order functions is a journey. As you continue to explore Erlang, you'll find these concepts invaluable in writing clean, efficient, and maintainable code. Keep experimenting, stay curious, and enjoy the journey!

## Quiz: First-Class and Higher-Order Functions

{{< quizdown >}}

### What does it mean for functions to be first-class citizens in Erlang?

- [x] Functions can be assigned to variables, passed as arguments, and returned from other functions.
- [ ] Functions can only be used within the module they are defined.
- [ ] Functions cannot be passed as arguments.
- [ ] Functions must always return a value.

> **Explanation:** First-class functions can be treated like any other data type, allowing them to be assigned, passed, and returned.

### Which of the following is a higher-order function?

- [x] A function that takes another function as an argument.
- [ ] A function that only returns integers.
- [ ] A function that does not take any arguments.
- [ ] A function that only operates on strings.

> **Explanation:** Higher-order functions can take other functions as arguments or return them as results.

### What is the result of applying `lists:map/2` with a function that doubles its input to the list `[1, 2, 3]`?

- [x] [2, 4, 6]
- [ ] [1, 2, 3]
- [ ] [3, 6, 9]
- [ ] [1, 4, 9]

> **Explanation:** `lists:map/2` applies the doubling function to each element, resulting in `[2, 4, 6]`.

### How does `lists:filter/2` work?

- [x] It filters elements of a list based on a predicate function.
- [ ] It sorts the list in ascending order.
- [ ] It removes duplicate elements from the list.
- [ ] It concatenates two lists.

> **Explanation:** `lists:filter/2` uses a predicate function to determine which elements to keep.

### What is the purpose of `lists:foldl/3`?

- [x] To accumulate a result by applying a function to each element of a list, starting from the left.
- [ ] To reverse the order of elements in a list.
- [ ] To find the maximum element in a list.
- [ ] To split a list into two parts.

> **Explanation:** `lists:foldl/3` is used for accumulation, applying a function to each element with an accumulator.

### Can functions be returned from other functions in Erlang?

- [x] Yes
- [ ] No

> **Explanation:** In Erlang, functions can be returned from other functions, demonstrating their first-class nature.

### What is a practical use of higher-order functions?

- [x] Data transformation and event handling.
- [ ] Only for mathematical calculations.
- [ ] For defining constants.
- [ ] For error handling only.

> **Explanation:** Higher-order functions are versatile and can be used for data transformation, event handling, and more.

### Which module in Erlang provides common higher-order functions for list processing?

- [x] `lists`
- [ ] `math`
- [ ] `string`
- [ ] `io`

> **Explanation:** The `lists` module provides higher-order functions like `map`, `filter`, and `foldl`.

### What is the benefit of using higher-order functions?

- [x] They promote code reuse and modularity.
- [ ] They make code harder to read.
- [ ] They are only useful in small programs.
- [ ] They are slower than regular functions.

> **Explanation:** Higher-order functions enhance code reuse and modularity, making code more maintainable.

### True or False: In Erlang, functions can only be passed as arguments but not returned from other functions.

- [ ] True
- [x] False

> **Explanation:** Functions in Erlang can both be passed as arguments and returned from other functions, showcasing their first-class status.

{{< /quizdown >}}
