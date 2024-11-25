---
linkTitle: "Lenses"
title: "Lenses: Abstractions for Accessing and Updating Nested Data Structures Immutably"
description: "A detailed exploration of the Lenses design pattern in functional programming, which allows for elegant and efficient access and modification of nested data structures immutably."
categories:
- Functional Programming
- Design Patterns
tags:
- Functional Programming
- Lenses
- Immutability
- Data Structures
- Abstractions
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/functional-collections-and-structures/specialized-structures/lenses"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


In functional programming, immutability and data transformation are fundamental principles. However, working with nested data structures often requires deep and frequent updates, which can lead to verbose and inefficient code. Lenses provide an elegant and powerful way to manage this complexity by offering a composable and immutable abstraction for accessing and updating deep structures.

## What Are Lenses?

A **Lens** is a first-class abstraction representing a getter and a setter for a particular field within a data structure. This allows for the modular and reusable encapsulation of access and modification logic, promoting code reusability and functional composition.

### Basic Anatomy of a Lens
Lenses are typically comprised of:

1. **Getter**: A function that extracts the value from a specific part of the data structure.
2. **Setter**: A function that updates the value at a specific part of the data structure while preserving immutability.

In functional programming, lenses are typically composed using pure functions, ensuring that the original data structures remain unmodified.

### Formal Definition
In Haskell-like pseudocode, a Lens `Lens S A` can be defined as:

```haskell
type Lens S A = Functor f => (A -> f A) -> S -> f S
```
1. `S` represents the original data type.
2. `A` represents the target element within the data type `S`.

## Key Operations with Lenses

### Accessing Data
Using the getter part of the lens:
```haskell
view :: Lens S A -> S -> A
view lens s = getConst (lens Const s)
```

### Updating Data
Using the setter part of the lens:
```haskell
set :: Lens S A -> A -> S -> S
set lens a s = runIdentity (lens (\_ -> Identity a) s)
```

### Modifying Data (Functional Update)
```haskell
over :: Lens S A -> (A -> A) -> S -> S
over lens f s = runIdentity (lens (Identity . f) s)
```

Additionally, compositions such as hierarchical access and cumulative transformations enhance the expressiveness and flexibility provided by lenses.

## Examples

### Getter Example
To illustrate, consider a simple data structure:

```haskell
data Address = Address { city :: String, zipCode :: String }
data Person  = Person { name :: String, address :: Address }
```

A lens to access the `city` field of a `Person`'s `Address` can be defined as:

```haskell
addressLens :: Lens' Person Address
addressLens f person = fmap (\newAddress -> person { address = newAddress }) (f (address person))

cityLens :: Lens' Address String
cityLens f address = fmap (\newCity -> address { city = newCity }) (f (city address))

personCityLens = addressLens . cityLens
```

To get the `city` of a `Person`:
```haskell
view personCityLens john
```

### Setter Example
To update the `city` of a `Person`:
```haskell
set personCityLens "NewCity" john
```

### Modifier Example
To transform the `city` of a `Person`:
```haskell
over personCityLens (map toUpper) john
```

## Related Design Patterns

### Prism
A **Prism** is a similar abstraction to a Lens but focuses on sum types or variants, allowing construction and deconstruction of these types. It's essential for managing data types with multiple constructors robustly.

### Traversal
A **Traversal** generalizes lenses by allowing operations on multiple parts within a data structure simultaneously. It effectively extends lenses to collection-like operations, such as applying a transformation to all elements within a list.

### Optional
An **Optional** is a combination of lens and traversal, enabling operations on potentially nonexistent elements within a data structure, accommodating partiality or optional presence flexibly.

## Additional Resources
1. [Learn You a Haskell – Lenses](http://learnyouahaskell.com/lenses-functors-and-myopia)
2. [The Essence of the Iterator Pattern](http://comonad.com/reader/2009/iterator/)
3. [Functional Programming in Scala – Lenses](https://leanpub.com/fpinscala/read#leanpub-auto-lenses)

## Summary

Lenses are a powerful and expressive design pattern in functional programming, offering a highly composable and reusable framework for accessing and updating nested data structures. By separating the concerns of data access and transformation, lenses enable the creation of more readable, maintainable, and composable code, fostering immutability and functional purity.

Understanding lenses can significantly improve how one models complex data interactions, enhancing both code quality and developer productivity. As you continue to leverage lenses and related abstractions like prisms, traversals, and optionals, you will unlock a versatile toolkit tailored for the functional programming paradigm.
