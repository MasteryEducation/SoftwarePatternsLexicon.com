---
linkTitle: "Catamorphism"
title: "Catamorphism: Generalizing Folds to Merge Different Representations"
description: "Catamorphisms provide a general abstraction for folding structures, merging various representations into reduced forms."
categories:
- Functional Programming
- Design Patterns
tags:
- catamorphism
- fold
- recursion schemes
- functional programming
- Haskell
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/advanced-patterns/functional-abstractions/catamorphism"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


**Catamorphisms** are a core concept in functional programming that generalize the idea of folding (also known as reduces) data structures into a single value. In essence, catamorphisms provide a way to systematically deconstruct complex data structures like lists, trees, or even more general algebraic data types, transforming them step by step into a simpler representation.

## Formal Definition

In category theory, a catamorphism for a functor \\( F \\) and an algebra \\( (A, \alpha) \\) is a unique homomorphism from the initial \\( F \\)-algebra to \\( (A, \alpha) \\).

### Mathematical Notation and Explanation

The catamorphism is often denoted as:

{{< katex >}} \mathrm{cata}_{(A, \alpha)} = \left| \begin{array}{c}
    X_{} \\ 
    F (X) \xrightarrow{\alpha} A \\ 
    \xi  \\ 
    \end{array} \right| {{< /katex >}}

#### Definitions:

- \\( F \\): A functor that describes the shape of the data structure.
- \\( X \\): The initial algebra, typically a data structure like a tree or list.
- \\( A \\): The resultant type.
- \\( \alpha \\): The algebra, a function \\( F(A) \rightarrow A \\) used to combine values.

## Example in Haskell

Consider the simple folding (reducing) of a list to its sum, which is a specific form of catamorphism applied to lists:

```haskell
-- List data type
data List a = Nil | Cons a (List a)

-- Fold for List (catamorphism)
foldr :: (a -> b -> b) -> b -> List a -> b
foldr _ acc Nil = acc
foldr f acc (Cons x xs) = f x (foldr f acc xs)

-- Sum example using foldr
sumList :: List Integer -> Integer
sumList = foldr (+) 0
```

In this example:
- The list structure `List a` is the initial algebra.
- The function `(+)` and `0` define the algebra for summing the list elements.

## Using Catamorphisms with Trees

Now, let’s illustrate a more complex example using binary trees.

### Binary Tree Structure

```haskell
-- Binary Tree data type
data Tree a = Leaf a | Node (Tree a) a (Tree a)

-- Catamorphism (fold function) for Tree
cataTree :: (a -> b) -> (b -> a -> b -> b) -> Tree a -> b
cataTree leafFunc nodeFunc = go
  where
    go (Leaf a) = leafFunc a
    go (Node left a right) = nodeFunc (go left) a (go right)
    
-- Usage example: Sum of all nodes in the tree
sumTree :: Num a => Tree a -> a
sumTree = cataTree id (\leftSum x rightSum -> leftSum + x + rightSum)
```

## Related Design Patterns

Catamorphisms are part of a broader family of recursion schemes in functional programming. These include:
- **Anamorphisms**: Generalize unfolds or constructions of data structures.
- **Hylomorphisms**: A combination of anamorphisms and catamorphisms, essentially a build followed by a fold.
- **Paramorphisms**: A generalization of catamorphisms that also gets access to the structure being deconstructed.

## Additional Resources

To deepen your understanding of catamorphisms and related concepts, consider the following resources:
- "Functional Programming with Bananas, Lenses, Envelopes and Barbed Wire" by Erik Meijer, Maarten Fokkinga, and Ross Paterson.
- "Theorem-Proving in Lean" by Leonardo de Moura et al.
- "Category Theory for Programmers" by Bartosz Milewski.

## Summary

Catamorphisms are a powerful pattern in functional programming, allowing for elegant and generalized reduction operations on complex data structures. By abstracting the process of folding, they enable more reusable and declarative code. Understanding catamorphisms and their relationships with other recursion schemes can greatly enhance your functional programming toolkit.
