---
canonical: "https://softwarepatternslexicon.com/patterns-erlang/7/7"
title: "Higher-Order Functions in Erlang: Practical Applications and Patterns"
description: "Explore the practical applications of higher-order functions in Erlang, including examples of common functions like lists:map/2 and lists:filter/2, and how they simplify code through abstraction and reuse."
linkTitle: "7.7 Higher-Order Functions in Practice"
categories:
- Functional Programming
- Erlang Design Patterns
- Code Abstraction
tags:
- Higher-Order Functions
- Erlang
- Functional Programming
- Code Reuse
- Abstraction
date: 2024-11-23
type: docs
nav_weight: 77000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 7.7 Higher-Order Functions in Practice

Higher-order functions are a cornerstone of functional programming, and Erlang is no exception. They allow us to write more abstract, reusable, and concise code by enabling functions to be passed as arguments, returned as values, and stored in data structures. In this section, we will delve into the practical applications of higher-order functions in Erlang, exploring common functions like `lists:map/2` and `lists:filter/2`, as well as custom implementations. We will also discuss how higher-order functions can simplify code and encourage abstraction and code reuse.

### Understanding Higher-Order Functions

A higher-order function is a function that takes one or more functions as arguments or returns a function as a result. This concept allows for a high level of abstraction and code reuse, making it easier to manage complex logic and operations.

#### Key Characteristics

- **Function as Argument**: Higher-order functions can accept other functions as parameters.
- **Function as Return Value**: They can return functions as results.
- **Function Composition**: They enable the composition of functions to build complex operations from simpler ones.

### Common Higher-Order Functions in Erlang

Erlang provides several built-in higher-order functions, particularly in the `lists` module, which are widely used for list processing. Let's explore some of these functions and their practical applications.

#### `lists:map/2`

The `lists:map/2` function applies a given function to each element of a list, returning a new list with the results. This is particularly useful for transforming data.

```erlang
% Define a function to double a number
double(X) -> X * 2.

% Use lists:map/2 to apply the double function to each element
DoubledList = lists:map(fun double/1, [1, 2, 3, 4, 5]),
io:format("Doubled List: ~p~n", [DoubledList]).
```

**Output:**

```
Doubled List: [2,4,6,8,10]
```

#### `lists:filter/2`

The `lists:filter/2` function selects elements from a list that satisfy a given predicate function.

```erlang
% Define a predicate function to check if a number is even
is_even(X) -> X rem 2 == 0.

% Use lists:filter/2 to filter even numbers
EvenNumbers = lists:filter(fun is_even/1, [1, 2, 3, 4, 5]),
io:format("Even Numbers: ~p~n", [EvenNumbers]).
```

**Output:**

```
Even Numbers: [2,4]
```

#### `lists:foldl/3` and `lists:foldr/3`

These functions are used to accumulate a result by iterating over a list. `lists:foldl/3` processes the list from left to right, while `lists:foldr/3` processes it from right to left.

```erlang
% Define a function to sum two numbers
sum(X, Y) -> X + Y.

% Use lists:foldl/3 to sum all elements in the list
Total = lists:foldl(fun sum/2, 0, [1, 2, 3, 4, 5]),
io:format("Total Sum: ~p~n", [Total]).
```

**Output:**

```
Total Sum: 15
```

### Custom Higher-Order Functions

In addition to using built-in higher-order functions, you can define your own to encapsulate common patterns and logic.

#### Example: Custom Map Function

```erlang
% Define a custom map function
custom_map(Fun, [H|T]) -> [Fun(H) | custom_map(Fun, T)];
custom_map(_, []) -> [].

% Use the custom map function
SquaredList = custom_map(fun(X) -> X * X end, [1, 2, 3, 4, 5]),
io:format("Squared List: ~p~n", [SquaredList]).
```

**Output:**

```
Squared List: [1,4,9,16,25]
```

### Simplifying Code with Higher-Order Functions

Higher-order functions can significantly simplify code by abstracting repetitive patterns and logic. This leads to cleaner, more maintainable code.

#### Example: Abstracting Conditional Logic

Consider a scenario where you need to apply different operations based on conditions. Higher-order functions can help abstract this logic.

```erlang
% Define operations
add_one(X) -> X + 1.
multiply_by_two(X) -> X * 2.

% Define a function to apply an operation based on a condition
apply_operation(Condition, X) ->
    Operation = case Condition of
        true -> fun add_one/1;
        false -> fun multiply_by_two/1
    end,
    Operation(X).

% Use the function
Result1 = apply_operation(true, 5),
Result2 = apply_operation(false, 5),
io:format("Results: ~p, ~p~n", [Result1, Result2]).
```

**Output:**

```
Results: 6, 10
```

### Encouraging Abstraction and Code Reuse

Higher-order functions encourage abstraction by allowing you to define generic operations that can be reused across different contexts.

#### Example: Reusable Sorting Function

```erlang
% Define a generic sorting function
sort_list(List, CompareFun) ->
    lists:sort(CompareFun, List).

% Use the sorting function with different comparison functions
Ascending = sort_list([5, 3, 1, 4, 2], fun(X, Y) -> X =< Y end),
Descending = sort_list([5, 3, 1, 4, 2], fun(X, Y) -> X >= Y end),
io:format("Ascending: ~p~nDescending: ~p~n", [Ascending, Descending]).
```

**Output:**

```
Ascending: [1,2,3,4,5]
Descending: [5,4,3,2,1]
```

### Visualizing Higher-Order Function Flow

To better understand how higher-order functions work, let's visualize the flow of data through a simple map operation.

```mermaid
graph TD;
    A[Input List: [1, 2, 3]] -->|map/2| B[Function: double/1];
    B --> C[Output List: [2, 4, 6]];
```

**Diagram Description:** This flowchart illustrates how the `map/2` function applies the `double/1` function to each element of the input list, resulting in the output list.

### Try It Yourself

Experiment with the examples provided by modifying the functions or the data they operate on. For instance, try creating a custom filter function or implementing a higher-order function that combines multiple operations.

### References and Further Reading

- [Erlang Documentation on Higher-Order Functions](https://www.erlang.org/doc/programming_examples/higher_order.html)
- [Functional Programming Concepts](https://en.wikipedia.org/wiki/Functional_programming)

### Knowledge Check

- What are higher-order functions, and why are they useful?
- How can you use `lists:map/2` to transform a list?
- What is the difference between `lists:foldl/3` and `lists:foldr/3`?
- How can higher-order functions help in code abstraction and reuse?

### Embrace the Journey

Remember, mastering higher-order functions is a journey. As you practice and experiment, you'll discover new ways to simplify and enhance your code. Keep exploring, stay curious, and enjoy the process of learning and applying functional programming concepts in Erlang!

## Quiz: Higher-Order Functions in Practice

{{< quizdown >}}

### What is a higher-order function?

- [x] A function that takes other functions as arguments or returns a function as a result.
- [ ] A function that only operates on numbers.
- [ ] A function that is always recursive.
- [ ] A function that cannot be used with lists.

> **Explanation:** Higher-order functions are those that can take other functions as arguments or return them as results, enabling powerful abstractions and code reuse.

### Which Erlang module provides common higher-order functions for list processing?

- [x] lists
- [ ] string
- [ ] io
- [ ] file

> **Explanation:** The `lists` module in Erlang provides common higher-order functions like `map/2`, `filter/2`, and `foldl/3` for list processing.

### What does `lists:map/2` do?

- [x] Applies a given function to each element of a list and returns a new list with the results.
- [ ] Filters elements from a list based on a predicate.
- [ ] Accumulates a result by iterating over a list.
- [ ] Sorts a list in ascending order.

> **Explanation:** `lists:map/2` applies a specified function to each element of a list, producing a new list with the transformed elements.

### How does `lists:filter/2` work?

- [x] It selects elements from a list that satisfy a given predicate function.
- [ ] It applies a function to each element of a list.
- [ ] It sorts a list based on a comparison function.
- [ ] It reverses the order of elements in a list.

> **Explanation:** `lists:filter/2` filters elements from a list that meet the criteria defined by a predicate function.

### What is the purpose of `lists:foldl/3`?

- [x] To accumulate a result by iterating over a list from left to right.
- [ ] To apply a function to each element of a list.
- [ ] To filter elements from a list.
- [ ] To reverse a list.

> **Explanation:** `lists:foldl/3` is used to accumulate a result by processing a list from left to right, applying a function to each element and an accumulator.

### Can higher-order functions return other functions?

- [x] Yes
- [ ] No

> **Explanation:** Higher-order functions can return other functions, allowing for dynamic function creation and composition.

### What is an advantage of using higher-order functions?

- [x] They allow for code abstraction and reuse.
- [ ] They make code longer and more complex.
- [ ] They are only useful for mathematical operations.
- [ ] They are slower than regular functions.

> **Explanation:** Higher-order functions enable code abstraction and reuse, making code more concise and maintainable.

### How can higher-order functions simplify conditional logic?

- [x] By abstracting the logic into reusable functions.
- [ ] By making conditions more complex.
- [ ] By eliminating the need for conditions.
- [ ] By increasing the number of conditional statements.

> **Explanation:** Higher-order functions can encapsulate conditional logic into reusable functions, simplifying the code structure.

### What is a common use case for higher-order functions?

- [x] Transforming and filtering data in lists.
- [ ] Writing low-level system code.
- [ ] Managing file I/O operations.
- [ ] Handling network protocols.

> **Explanation:** Higher-order functions are commonly used for transforming and filtering data in lists, among other applications.

### True or False: Higher-order functions can only be used with lists.

- [ ] True
- [x] False

> **Explanation:** Higher-order functions can be used with various data structures and contexts, not just lists.

{{< /quizdown >}}
