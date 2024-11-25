---
linkTitle: "Prisms"
title: "Prisms: Focus on a Part of a Data Structure That Can Be Absent"
description: "In-depth exploration of Prisms, a powerful functional programming design pattern used to focus on optional parts of data structures. Learn about its principles, applications, related patterns, and advanced usage."
categories:
- Functional Programming
- Design Patterns
tags:
- Functional Programming
- Design Patterns
- Prisms
- Optics
- Data Structures
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/functional-collections-and-structures/specialized-structures/prisms"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Prisms: Focus on a Part of a Data Structure That Can Be Absent

Prisms are a powerful concept in functional programming, particularly in the realm of *optics*. They allow us to focus on and manipulate parts of data structures that can be optionally present—essentially, they provide a way to work with data structures that may contain absent elements. By mastering prisms, you'll gain greater control over complex data transformations that involve optional data.

### What Are Prisms?

In layman's terms, a prism is an abstraction that lets you look into a data structure to see if a specific part exists and to interact with it if it does. Prisms provide a lens into data structures that may not always contain the value you're interested in, enabling safe read and write operations on optional data.

### Formal Definition

Formally, a prism in functional programming is a type of optic used to navigate through sum types, particularly those that have an optional component. They generalize the concept of optional values seen in algebraic data types.

A prism can be represented using the following two primary operations:

1. **Review**: This operation constructs a value of the data type from a part.
2. **Preview**: This operation looks into the data type, checking if the part is present and extracting it if available.

### Implementing Prisms in Haskell

In Haskell, prisms are part of the larger category of lens libraries such as `lens`, `optics`, or `microlens`. Here’s an example implementation using the `lens` library:

```haskell
{-# LANGUAGE TemplateHaskell #-}

import Control.Lens

data MyDataType = MyVariant Int | AnotherVariant String
makePrisms ''MyDataType

-- Using the prism
example :: Maybe Int
example = MyVariant 42 ^? _MyVariant  -- This results in Just 42

-- Using review to construct data
exampleConstruct :: MyDataType
exampleConstruct = _MyVariant # 42  -- This constructs MyVariant 42
```

### The Role of Prisms

#### Safe Data Access

Prisms are invaluable for safely accessing and modifying data when it might not be present. They extend the concept of lenses, which focus on a definite part of a data structure, and are particularly useful for dealing with sum types, i.e., types with multiple constructs or branches.

#### Pattern Matching

Using prisms, you can concisely and effectively pattern match against different variants of a type. This is hugely beneficial when dealing with complex data structures, as it simplifies branching logic.

### Related Patterns and Concepts

Prisms are not isolated; they interact with various other design patterns and concepts in functional programming:

- **Lenses**: While lenses deal with product types by focusing on definite parts of data structures, prisms handle sum types with optional parts.
- **Optional (Maybe)**: The `Maybe` type in Haskell represents an optional value. Prisms can be viewed as an advanced tool for working with `Maybe` or similar types.
- **Traversal**: While a lens or prism targets a single focus, traversals can potentially target multiple parts. Prisms can be composed with traversals for intricate data access patterns.

### Additional Resources

To deepen your understanding of prisms, consider the following resources:

- [Lens Library Documentation](https://hackage.haskell.org/package/lens)
- [Optics in Haskell by Vlad Ciobanu](http://comonad.com/reader/2012/practical-lens/)
- [Functional Programming Patterns in Scala and Clojure](https://www.edx.org/course/functional-programming-principles-in-scala)

### Summary

Prisms provide a robust and elegant way to handle optional data in functional programming. By delivering tools to focus on parts of data structures that may or may not be present, prisms extend your ability to write concise and safe data manipulation code. Leveraging prisms properly can lead to better code maintainability and cleaner logic, especially when dealing with complex types.

Mastering prisms alongside lenses and other optic constructs will significantly enhance your functional programming toolkit, allowing for more expressive and powerful data transformations.


