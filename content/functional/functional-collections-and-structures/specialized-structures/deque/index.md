---
linkTitle: "Deque"
title: "Deque: Immutable Double-Ended Queue"
description: "A comprehensive exploration of the Deque design pattern in functional programming, emphasizing its immutability and double-ended interface."
categories:
- Functional Programming
- Design Patterns
tags:
- Functional Programming
- Design Patterns
- Immutability
- Data Structures
- Deque
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/functional-collections-and-structures/specialized-structures/deque"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Deque: Immutable Double-Ended Queue

In functional programming, a Deque (double-ended queue) is a versatile data structure that allows for efficient insertion and removal operations from both ends — the front (head) and the back (tail). The distinguishing aspect of a functional Deque is its immutability. This means any operation on the Deque results in a new Deque instance rather than modifying the original.

### Characteristics of Immutable Deque

1. **Immutability**: Immutable collections do not change after creation. Any operation that would mutate the collection, such as adding or removing elements, instead returns a new collection with the operation performed.
2. **Double-Ended Accessibility**: Unlike single-ended queues like stacks (last-in, first-out) or queues (first-in, first-out), deques allow insertion and deletion from both ends.

### Operations on Deque

The typical operations on a Deque include:

- **`empty`**: Creates an empty Deque.
- **`isEmpty`**: Checks if the Deque is empty.
- **`cons`**: Adds an element to the front of the Deque.
- **`snoc`**: Adds an element to the rear of the Deque.
- **`head`**: Retrieves the front element of the Deque.
- **`tail`**: Returns a new Deque without the front element.
- **`last`**: Retrieves the rear element of the Deque.
- **`init`**: Returns a new Deque without the rear element.

### Implementation Example in Haskell

Here is a simple implementation of an immutable Deque in Haskell:

```haskell
data Deque a = Deque [a] [a]
  deriving Show

empty :: Deque a
empty = Deque [] []

isEmpty :: Deque a -> Bool
isEmpty (Deque [] []) = True
isEmpty _             = False

cons :: a -> Deque a -> Deque a
cons x (Deque f r) = Deque (x : f) r

snoc :: a -> Deque a -> Deque a
snoc x (Deque f r) = Deque f (x : r)

head :: Deque a -> Maybe a
head (Deque [] [])  = Nothing
head (Deque [] r)   = Just (last r)
head (Deque (x:_) _) = Just x

tail :: Deque a -> Deque a
tail (Deque [] [])   = empty
tail (Deque [] [x])  = empty
tail (Deque [] (x:r)) = Deque (reverse r) []
tail (Deque (x:f) r) = Deque f r

last :: Deque a -> Maybe a
last (Deque [] [])  = Nothing
last (Deque f [])   = Just (last f)
last (Deque _ (x:_)) = Just x

init :: Deque a -> Deque a
init (Deque [] [])   = empty
init (Deque [x] [])  = empty
init (Deque f [])    = Deque (init f) []
init (Deque f (x:r)) = Deque f r
```

This implementation uses two lists to store elements, and reverses them when necessary to maintain performance characteristics for double-ended access.

### Comparison with Other Patterns

- **Functional List**: A singly-linked list supports only one-ended operations efficiently.
- **Queue**: A typical functional queue supports only enqueuing at the rear and dequeuing from the front.
- **Stack**: LIFO structure which supports only cons and tail operations efficiently.

### Related Design Patterns

1. **Persistent Data Structures**: The concept of immutability in Deque aligns with persistent data structures, which offer versions of themselves with each modification.
2. **Iterator Pattern**: Iterating over Deque can be related to the iterator pattern, though ensuring immutability.
3. **Builder Pattern**: A Deque builder can help construct a Deque by aggregating cons and snoc operations efficiently before finalizing the immutable structure.

### Additional Resources

- [Okasaki, Chris. "Purely Functional Data Structures"](https://www.cambridge.org/core/books/purely-functional-data-structures/DB550571DB016E20126BAF11DF85D7CE): Offers a deep dive into various functional data structures, including Deque.
- Haskell Documentation: [Data.Sequence in Haskell](https://hackage.haskell.org/package/containers/docs/Data-Sequence.html): Explore another immutable double-ended sequence.
- Scala documentation: [Scala Collections](https://docs.scala-lang.org/overviews/collections/introduction.html): Scala’s immutable collections library, including Queues and Lists.

## Summary

This article explored the Deque design pattern focusing on its application in an immutable context within functional programming. We covered basic operations, compared Deque to similar data structures, and highlighted related design patterns. Immutable Deque offers advantages in functional paradigms, ensuring safe, predictable state transformations while maintaining efficient access on both ends.

_ptr_alpha "Enjoy working with immutability and efficient data structures."_
