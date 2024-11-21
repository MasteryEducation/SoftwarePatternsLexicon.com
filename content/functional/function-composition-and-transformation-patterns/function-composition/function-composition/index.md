---
linkTitle: "Function Composition"
title: "Function Composition: Combining Functions to Form New Functions"
description: "Function Composition involves combining two or more functions to form a new function. It is a fundamental concept in functional programming which enhances reusability and modularity."
categories:
- Functional Programming
- Design Patterns
tags:
- Function Composition
- Higher-Order Functions
- Functional Programming Principles
- Reusability
- Modularity
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/function-composition-and-transformation-patterns/function-composition/function-composition"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction
Function Composition is a core concept in functional programming where two or more functions are combined to produce a new function. By chaining functions together, you can simplify complex operations and create more reusable and modular code. Function Composition enhances code readability, maintainability, and expressiveness.

## Definitions and Examples

### Mathematical Background
In mathematics, function composition is defined as follows. Given two functions \\( f: X \to Y \\) and \\( g: Y \to Z \\), the composition \\( g \circ f : X \to Z \\) is defined by:

{{< katex >}} (g \circ f)(x) = g(f(x)) {{< /katex >}}

where \\( x \\) is an element of set \\( X \\).

### Programming Example
In many functional programming languages, function composition is a basic operation. For instance, let's observe examples in Haskell and JavaScript.

#### Haskell
In Haskell, function composition is done using the `.` operator.

```haskell
-- Define two simple functions
f :: Int -> Int
f x = x + 1

g :: Int -> Int
g x = x * 2

-- Compose the functions
composedFunction :: Int -> Int
composedFunction = g . f

-- Example usage
result = composedFunction 3 -- Output will be (3 + 1) * 2 which is 8
```

#### JavaScript
In JavaScript, function composition can be achieved by chaining functions or using utility functions like `compose` from libraries like lodash or Ramda.

```javascript
const f = x => x + 1;
const g = x => x * 2;

// Composing manually
const composedFunction = x => g(f(x));

console.log(composedFunction(3)); // Output will be 8

// Using Ramda's compose function
const { compose } = require('ramda');
const composedWithRamda = compose(g, f);

console.log(composedWithRamda(3)); // Output will be 8
```

## Properties and Benefits
### Properties
* **Associativity:** Function composition is associative. \\( (f \circ g) \circ h = f \circ (g \circ h) \\).
* **Identity Function:** An identity function is a function that does nothing to its argument. For a function \\( f \\), composing it with an identity function \\( id \\) will yield \\( f \\). Formally, \\( f \circ id = id \circ f = f \\).

### Benefits
* **Abstraction:** Higher-level abstractions by composing smaller functions.
* **Reusability:** Modular and reusable code blocks.
* **Readability:** Clear, concise, and readable code.
* **Testability:** Easier to test smaller, focused functions.

## Related Design Patterns
### Higher-Order Functions
Higher-order functions are functions that take other functions as arguments or return them as results. Function Composition itself is an example of higher-order functions.
  
### Monads
Monads provide a way to chain operations together using a context (e.g., Maybe, Either, or IO monads in Haskell). They rely heavily on functions and composition to manage side effects.

## Additional Resources
1. [Haskell Documentation](https://www.haskell.org/documentation/)
2. [Ramda Documentation](https://ramdajs.com/docs/)
3. [Function Composition in JavaScript](https://jrsinclair.com/articles/2019/compose-function-magic-magical-composition/)
4. Book: *Functional Programming in JavaScript* by Luis Atencio.

## Summary
Function Composition is an essential concept in functional programming that allows developers to combine smaller functions into more complex operations. It enhances code reusability, readability, and maintainability. By leveraging this pattern, along with related patterns such as higher-order functions and monads, developers can write more robust and elegant codebases.

Mastering Function Composition enables leveraging the full potential of functional programming principles, leading to high-quality and efficient software solutions.
