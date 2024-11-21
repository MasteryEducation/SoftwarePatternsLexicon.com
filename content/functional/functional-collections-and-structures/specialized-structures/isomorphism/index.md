---
linkTitle: "Isomorphism"
title: "Isomorphism: A Reversible Transformation Between Two Structures"
description: "Understanding isomorphisms in the context of functional programming allows developers to transform data structures and types in a reversible manner, ensuring that the essential structure and properties are preserved."
categories:
- Functional Programming
- Design Patterns
tags:
- functional programming
- design patterns
- isomorphism
- types
- data structures
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/functional-collections-and-structures/specialized-structures/isomorphism"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction
In the realm of functional programming, **isomorphism** refers to a reversible transformation between two structures, ensuring that data can be transformed back and forth without loss of information. Understanding and leveraging isomorphisms can lead to more robust, flexible, and maintainable software design by preserving the essential properties and behaviors of the transformed structures or types.

## Core Concepts

### Definition and Properties
An isomorphism between two structures \\( A \\) and \\( B \\) involves a pair of functions:

- \\( f: A \rightarrow B \\)
- \\( g: B \rightarrow A \\)

These functions must satisfy the following conditions:

1. **Reversibility**: Applying one function after the other returns the original value.
   {{< katex >}}
   g(f(a)) = a \quad \forall a \in A
   {{< /katex >}}
   {{< katex >}}
   f(g(b)) = b \quad \forall b \in B
   {{< /katex >}}

2. **Bijectiveness**: Each element of \\( A \\) is mapped to a unique element of \\( B \\), and vice versa, implying no data is lost during transformation.

This concept ensures that transformations between data structures or types retain their integrity.

## Examples

### From Pairs to Lists
Consider the example of transforming a pair \\( (a, b) \\) into a list \\( [a, b] \\):

```haskell
toList :: (a, b) -> [a, b]
toList (x, y) = [x, y]

fromList :: [a, b] -> (a, b)
fromList [x, y] = (x, y)
```

These functions form an isomorphism as they satisfy the requirements of reversibility and bijectiveness.

### From JSON to Haskell Data Types
Another practical example is transforming between JSON objects and Haskell data types. Using the Aeson library in Haskell:

```haskell
{-# LANGUAGE DeriveGeneric #-}

import Data.Aeson
import GHC.Generics

data User = User { name :: String, age :: Int } deriving (Generic, Show)

instance ToJSON User
instance FromJSON User

-- Example of encoding and decoding
jsonEncode :: User -> ByteString
jsonEncode user = encode user

jsonDecode :: ByteString -> Maybe User
jsonDecode json = decode json
```

Here, `ToJSON` and `FromJSON` instances ensure that Haskell data types can be serialized to JSON and deserialized back to their original forms, maintaining an isomorphism between these representations.

## Related Design Patterns

### Functors and Natural Transformations
* **Functors**: Mappings between categories that preserve the categorical structure. In functional programming, they allow us to apply a function over a wrapped value (e.g., in a list or maybe context).
* **Natural Transformations**: Mappings between functors that preserve the structure of the functor. They can be thought of as a higher-level form of isomorphisms where transformations apply uniformly across functors.

### Lenses
Lenses provide a composable way to access and update nested data structures. Lenses can be seen as partial isomorphisms that focus on specific parts of data structures, enabling elegant and efficient manipulation.

```haskell
{-# LANGUAGE TemplateHaskell #-}

import Control.Lens

data Person = Person { _name :: String, _address :: Address} deriving (Show)
data Address = Address { _city :: String, _postcode :: String } deriving (Show)

makeLenses ''Person
makeLenses ''Address

-- Example of using Lenses
changeCity :: Person -> String -> Person
changeCity person newCity = person & address . city .~ newCity
```

In this example, lenses provide a reversible transformation (access and update) for nested fields within the data structure.

## Additional Resources
- [Category Theory for Programmers by Bartosz Milewski](https://www.goodreads.com/book/show/26284985-category-theory-for-programmers)
- [Learn You a Haskell for Great Good! by Miran Lipovaca](http://learnyouahaskell.com/)
- [Haskell Programming from First Principles by Christopher Allen and Julie Moronuki](http://haskellbook.com/)

## Summary
Isomorphisms play a crucial role in functional programming by enabling the reversible transformation of data structures and types. They ensure that transformations are lossless and bijective, thereby retaining the original data integrity. Understanding isomorphisms and related patterns like functors, natural transformations, and lenses can significantly enhance the flexibility and maintainability of your code.

Utilizing isomorphisms can simplify data handling and promote a more declarative and composable codebase, which are the hallmarks of functional programming.
