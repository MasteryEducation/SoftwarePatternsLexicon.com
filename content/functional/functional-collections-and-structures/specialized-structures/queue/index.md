---
linkTitle: "Queue"
title: "Queue: Immutable First-In-First-Out Collection"
description: "A comprehensive guide to the Queue design pattern in Functional Programming, an immutable collection that strictly follows the first-in-first-out (FIFO) principle."
categories:
- Functional Programming
- Data Structures
tags:
- Queue
- Immutable
- Collection
- Functional Programming
- FIFO
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/functional-collections-and-structures/specialized-structures/queue"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

The **Queue** is a fundamental data structure in computer science that operates strictly under the First-In-First-Out (FIFO) principle. This means that the first element added to the queue will be the first one to be removed. In the context of functional programming, a queue is implemented as an immutable collection, which provides several advantages such as simpler state management and easier reasoning about the code.

## Characteristics of an Immutable Queue

An immutable queue in functional programming has the following characteristics:

1. **First-In-First-Out (FIFO) Principle**: The order of operations maintains that the first element added to the queue is the first one to be removed.
2. **Immutability**: The queue cannot be modified once it's created; instead, any operation that transforms the queue returns a new queue.
3. **Persistent Data Structure**: Modifications to the queue result in a new version of the queue while sharing the unchanged parts of the old version, making them memory efficient.

## Basic Operations

An immutable queue typically supports the following operations:

1. **Enqueue**: Adding an element to the end of the queue.
2. **Dequeue**: Removing an element from the front of the queue.
3. **Peek**: Viewing the element at the front of the queue without removing it.
4. **IsEmpty**: Checking if the queue is empty.

Here is a basic implementation of an immutable queue in a functional programming language like Haskell or Scala:

### Haskell Example

```haskell
data Queue a = Queue [a] [a]

-- Create an empty queue
emptyQueue :: Queue a
emptyQueue = Queue [] []

-- Enqueue an element
enqueue :: a -> Queue a -> Queue a
enqueue x (Queue ins outs) = Queue (x:ins) outs

-- Dequeue an element
dequeue :: Queue a -> (Maybe a, Queue a)
dequeue (Queue [] []) = (Nothing, emptyQueue)
dequeue (Queue ins []) = dequeue (Queue [] (reverse ins))
dequeue (Queue ins (y:ys)) = (Just y, Queue ins ys)

-- Peek the front element
peek :: Queue a -> Maybe a
peek (Queue _ []) = Nothing
peek (Queue _ (y:_)) = Just y

-- Check if the queue is empty
isEmpty :: Queue a -> Bool
isEmpty (Queue [] []) = True
isEmpty _ = False
```

### Scala Example

```scala
case class Queue[+A](
  in: List[A] = List.empty[A], 
  out: List[A] = List.empty[A]
) {
  // Enqueue an element
  def enqueue[B >: A](element: B): Queue[B] = 
    Queue(element :: in, out)

  // Dequeue an element
  def dequeue: (Option[A], Queue[A]) = out match {
    case Nil if in.isEmpty => (None, Queue(in, out))
    case Nil => Queue(Nil, in.reverse).dequeue
    case y :: ys => (Some(y), Queue(in, ys))
  }

  // Peek the front element
  def peek: Option[A] = out.headOption

  // Check if the queue is empty
  def isEmpty: Boolean = in.isEmpty && out.isEmpty
}
```

## Related Design Patterns

### Stack

- **Description**: A stack is another fundamental data structure that follows the Last-In-First-Out (LIFO) principle. Unlike queues, the most recently added element is the first to be removed.
- **Use Case**: Useful for tasks that require reversing the order of inputs, such as parsing expressions in compilers.

### Persistent Data Structures

- **Description**: Persistent data structures maintain previous versions of themselves when modified, making them particularly well-suited for immutable contexts.
- **Use Case**: Ideal for applications that require undo/redo functionality or versioned data access.

### Lazy Evaluation

- **Description**: Lazy evaluation delays the computation of expressions until their value is actually needed, which can lead to performance improvements.
- **Use Case**: Applicable in functional programming languages to optimize operations on large data structures or infinite sequences.

## Additional Resources

- [Purely Functional Data Structures by Chris Okasaki](https://www.cs.cmu.edu/~rwh/theses/okasaki.pdf)
- [Functional Programming in Scala](https://www.manning.com/books/functional-programming-in-scala)
- [Haskell Language Documentation](https://www.haskell.org/documentation/)

## Summary

The Queue design pattern is a crucial tool in the functional programming toolkit, providing a simple yet powerful way to manage collections in a first-in-first-out manner while preserving immutability. Its primary operations—enqueue, dequeue, peek, and isEmpty—are straightforward yet enable complex behaviors and efficient performance. Utilizing queues within the principles of functional programming leads to more maintainable, predictable, and reusable code.

By exploring related design patterns like stacks and persistent data structures, and utilizing additional resources for further study, one can gain a deep understanding of how to effectively leverage immutable queues in functional programming to solve real-world problems efficiently and elegantly.
