---
linkTitle: "Function Wrapping"
title: "Function Wrapping: Enhancing Functions with Additional Behavior"
description: "A comprehensive guide on the Function Wrapping design pattern in functional programming, focusing on enhancing functions by wrapping additional behavior."
categories:
- functional-programming
- design-patterns
tags:
- function-wrapping
- decorators
- higher-order-functions
- functional-programming-patterns
- software-design
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/miscellaneous-patterns/decorative-patterns/function-wrapping"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


Function wrapping is a design pattern in functional programming where existing functions are enhanced by wrapping them with additional behavior. This pattern is incredibly effective for introducing cross-cutting concerns without modifying the original function, thereby promoting code reuse and separation of concerns.

## Overview

In essence, function wrapping involves creating higher-order functions (HOFs) that take a function as an argument and return a new function with additional behavior. This method is analogous to the Decorator Pattern found in object-oriented programming but is more naturally and expressively implemented in functional languages.

## Core Concepts

### Higher-Order Functions (HOFs)
A HOF is any function that takes one or more functions as arguments and/or returns a function as its result. In the context of function wrapping, HOFs are pivotal as they allow for the composition of additional behaviors around existing functions.

### Closures
Closures are functions that capture and enclose variables from their containing environment. They are crucial for encapsulating the additional behavior along with the original function, enabling stateful decorations if needed.

## Key Benefits

- **Separation of Concerns**: Allows the core functionality of the function to remain clean while adding orthogonal concerns such as logging, monitoring, etc.
- **Modularity**: Facilitates code reuse by modularly composing behaviors.
- **Testing and Maintenance**: Simplifies testing and maintenance since the core logic and additional behaviors are decoupled.

## Implementation

### Basic Example in JavaScript

Using JavaScript, we can implement function wrapping to demonstrate a logging wrapper:

```javascript
function logWrapper(fn) {
    return function(...args) {
        console.log(`Arguments: ${args}`);
        const result = fn(...args);
        console.log(`Result: ${result}`);
        return result;
    };
}

function add(a, b) {
    return a + b;
}

const wrappedAdd = logWrapper(add);
wrappedAdd(2, 3); // Logs arguments and result
```

### Complex Example in Haskell

In Haskell, we can use higher-order functions to wrap additional behaviors around pure functions:

```haskell
-- Define a logging wrapper
logWrapper :: (Show a, Show b) => (a -> b) -> (a -> b)
logWrapper fn = \x -> 
                let result = fn x
                in trace ("Input: " ++ show x ++ " Output: " ++ show result) result

-- Example function
multiplyByTwo :: Int -> Int
multiplyByTwo x = x * 2

-- Wrapped function
main :: IO ()
main = print $ logWrapper multiplyByTwo 10
```

### Design Considerations

1. **Performance**: Wrapping functions can introduce overhead, especially if the wrapped functionality is complex or if the wrapping function is applied frequently.
2. **Readability**: Massive layers of wrapping can sometimes lead to less readable and maintainable code.

## Related Patterns

### Decorator Pattern
While primarily an object-oriented design pattern, the Decorator Pattern shares conceptual similarities with function wrapping. Both aim to add behavior to entities without altering their core.

### Middleware Pattern (in Web Frameworks)
This pattern, common in web frameworks like Express.js, is essentially a form of wrapping where requests pass through a chain of middleware functions, each adding behavior before handing off control.

### Interceptor Pattern
Similar to function wrapping, interceptors allow pre-processing and post-processing. Common in event-driven or AOP systems.

## Additional Resources

1. **Books**
   - "Functional Programming in Scala" by Paul Chiusano and Runar Bjarnason - Discusses higher-order functions in depth.
   - "JavaScript: The Good Parts" by Douglas Crockford - Covers functional aspects and higher-order functions in JavaScript.

2. **Online Articles**
   - [Eloquent JavaScript](https://eloquentjavascript.net/), Chapter 5: Higher-Order Functions.

3. **Libraries**
   - **Lodash** - A JavaScript library that provides utilities for function manipulation, including decorators.

## Summary

Function wrapping is a versatile and powerful design pattern that enhances the functionality of existing functions through higher-order functions. It embodies the principles of modularity, separation of concerns, and code reuse effectively. Whether implemented in dynamically-typed languages like JavaScript, or statically-typed ones like Haskell, function wrapping remains a fundamental technique in the functional programming toolkit.

By understanding and leveraging function wrapping, developers can create more maintainable, testable, and extensible codebases.
