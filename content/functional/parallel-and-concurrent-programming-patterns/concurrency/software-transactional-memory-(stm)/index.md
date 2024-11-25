---
linkTitle: "Software Transactional Memory"
title: "Software Transactional Memory: Handling Concurrency with Transactions"
description: "A deep dive into Software Transactional Memory, a functional programming design pattern for managing concurrency by using transactions to manage memory changes."
categories:
- Functional Programming
- Concurrency
tags:
- STM
- Concurrency
- Functional Programming
- Transactions
- Haskell
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/parallel-and-concurrent-programming-patterns/concurrency/software-transactional-memory-(stm)"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction to Software Transactional Memory (STM)

Software Transactional Memory (STM) is a concurrency control mechanism analogous to database transactions for controlling access to shared memory in concurrent computing. STM simplifies concurrent programming by allowing multiple threads to operate on shared data structures while ensuring consistency and isolation, much like transactions in a database management system (DBMS). STM is particularly suited for functional programming as it aligns well with the principles of immutability and statelessness.

## Core Concepts

### Transactions
Transactions in STM are atomic units of work. Each transaction succeeds or fails as a single unit. If a transaction fails, all of its side effects are rolled back.

### Atomicity
STM ensures that all operations within a transaction are executed atomically. This means that each transaction appears to be instantaneous — no intermediate states are visible to other threads.

### Consistency
STM maintains consistent state across transactions. If multiple transactions are attempting to modify the same shared state, STM ensures that only one of them can commit the changes, maintaining consistency.

### Isolation
Transactions are isolated from each other, meaning that the intermediate states of a transaction are invisible to other transactions. Only the final committed state can be observed.

### Durability (Optional)
While durability is crucial in database systems (ensuring committed changes are not lost), it is generally not a core requirement of STM systems, as data is usually volatile within application memory.

## How STM Works

STM uses two main strategies: optimistic and pessimistic concurrency control.

### Optimistic Concurrency Control
Optimistic STM allows transactions to execute without locks, validating them at commit time to ensure no conflicts with other transactions.

### Pessimistic Concurrency Control
Pessimistic STM uses locks to ensure that transactions do not violate consistency, locking resources they are going to use during the transaction's execution.

### Example in Haskell

Below is a simple example in Haskell demonstrating the use of STM:

```haskell
import Control.Concurrent.STM
import Control.Monad (forM_)

-- Define the transactional variables (TVars)
type Account = TVar Int

-- Transfer function
transfer :: Account -> Account -> Int -> STM ()
transfer fromAcc toAcc amount = do
    fromBalance <- readTVar fromAcc
    toBalance <- readTVar toAcc
    writeTVar fromAcc (fromBalance - amount)
    writeTVar toAcc (toBalance + amount)

main :: IO ()
main = do
    -- Create two accounts
    accountA <- atomically $ newTVar 1000
    accountB <- atomically $ newTVar 2000

    -- Perform the transfer inside an atomically block
    atomically $ transfer accountA accountB 100
    
    -- Print the result
    blnA <- atomically $ readTVar accountA
    blnB <- atomically $ readTVar accountB
    putStrLn $ "Account A Balance: " ++ show blnA
    putStrLn $ "Account B Balance: " ++ show blnB
```

In this example, the `transfer` function performs a simple transaction: moving an amount from `accountA` to `accountB`. The `atomically` function ensures the transaction executes atomically.

## Related Design Patterns

### Actor Model
The Actor Model is another concurrency abstraction where "actors" are the fundamental units of computation. Each actor has a mailbox to which other actors can send messages. It's a powerful model for distributed systems but can be overkill for simpler shared-memory concurrency use cases that STM elegantly handles.

### Futures and Promises
Futures and Promises provide an alternative approach to handling concurrency by allowing tasks to be scheduled to run asynchronously, with results available at some point in the future. This model, however, can complicate inter-thread communication which STM simplifies.

### Lock-Free Data Structures
Lock-free data structures offer another concurrency mechanism where multiple threads can interact with data without conventional locking mechanisms. STM provides transactional guarantees that can be easier to reason about compared to low-level lock-free designs.

## Additional Resources

1. **Books**:
    - *"Parallel and Concurrent Programming in Haskell"* by Simon Marlow
    - *"Concurrent Programming in Java™: Design Principles and Patterns"* by Doug Lea

2. **Research Papers**:
    - *"Transactional memory: architectural support for lock-free data structures"* by Maurice Herlihy and J. Eliot B. Moss
    - *"Composable Memory Transactions"* by Tim Harris, Simon Marlow, Simon Peyton Jones, and Maurice Herlihy

3. **Online Tutorials**:
    - [Haskell STM Documentation](https://hackage.haskell.org/package/stm)
    - [Learn You a Haskell for Great Good!](http://learnyouahaskell.com)

## Summary

Software Transactional Memory (STM) is a powerful tool for managing concurrency in functional programming. By using transactional semantics, STM ensures that shared memory operations remain consistent, atomic, and isolated, making it easier to reason about concurrency. This pattern aligns well with functional programming paradigms and simplifies the construction of robust, concurrent systems.

STM is not just limited to Haskell; concepts and libraries exist for other languages that support functional patterns, providing a broad utility for software developers facing complex concurrency challenges.


