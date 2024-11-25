---
linkTitle: "Vars and Refs"
title: "Vars and Refs: Handling Mutable State Immutably"
description: "Exploring how Vars and Refs enable mutable state handling in an immutable fashion within Functional Programming paradigms."
categories:
- Functional Programming
- Design Patterns
tags:
- Vars and Refs
- Mutable State
- Immutability
- Concurrency
- Functional Programming Patterns
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/miscellaneous-patterns/state-management/vars-and-refs"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


In functional programming, the primary tenet is immutability, which suggests that variables once defined should never change. However, real-world applications inevitably encounter scenarios where maintaining a mutable state is necessary. The design pattern encompassing Vars and Refs provides a sophisticated means to manage mutable states while preserving the benefits of functional paradigms.

## Introduction

Vars and Refs are constructs used to embody mutable state securely and consistently, ensuring that state transitions are handled in a controlled manner. By leveraging these constructs, we can maintain program predictability, facilitate concurrency, and ensure data integrity.

### Var

A `Var` is a type of reference that points to a value that can be changed. They are akin to variables found in imperative languages, but with a crucial difference — changes to the value of a Var are managed within the functional context to ensure consistency.

### Ref

A `Ref` offers a higher level of abstraction over a `Var`. Typically, a Ref is used in conjunction with software transactional memory (STM) systems to ensure that multiple changes can be made atomically, preventing race conditions and ensuring consistency across concurrent operations.

## Detailed Mechanisms

### Vars Implementation in Clojure

In Clojure, `Vars` are implemented in a way that allows threads to have isolated changes to a variable:

```clojure
(def ^:dynamic *x* 10)

(defn update-var []
  (binding [*x* 20]
    (println *x*)))

(println *x*) ; prints 10
(update-var)  ; prints 20
(println *x*) ; prints 10
```

This demonstrates dynamic scoping where the binding affects only the current thread's scope.

### Refs Implementation in Clojure

Clojure's `Refs` provide a way to manage state changes across multiple threads using STM. Here’s an example with `Refs`:

```clojure
(def balance (ref 100))

(defn update-balance [amt]
  (dosync
    (alter balance + amt)))

(println @balance) ; prints 100
(update-balance 50)
(println @balance) ; prints 150
```

In this example, the `dosync` block ensures that operations within it are atomic and isolated.

## Advantages

1. **Concurrency Control**: Vars and Refs offer fine-grained control over concurrent state modifications, utilizing managed reference changing and STM capabilities.
   
2. **Data Integrity**: With atomic operations and managed state transitions, Vars and Refs ensure that state remains consistent across various operations.

3. **Isolation**: Dynamic bindings in Vars isolate state changes to specific contexts, allowing seamless data mutations in isolation.

## Related Design Patterns

1. **Monads**: Monads encapsulate computations and side effects, providing a method to handle state changes in pure functions.
   
2. **Actor Model**: The Actor Model in functional programming manages state internally and communicates through message passing to handle concurrency robustly.

3. **Event Sourcing**: This pattern stores all changes to the state as a sequence of events, providing a model to reconstruct state immutably.

## Additional Resources

- ["Programming Clojure" by Alex Miller, Stuart Halloway, Aaron Bedra](https://pragprog.com/book/shcloj3/programming-clojure)
- [Clojure documentation on Vars](https://clojure.org/reference/vars)
- [Rich Hickey's talk on Value and Identity](https://www.youtube.com/watch?v=HMZoBCo1sWw)

## Summary

The Vars and Refs pattern adeptly balances the necessity for mutable state with the core principles of functional programming. By leveraging managed references and STM, this pattern enables mutable state handling in an ordered, predictable, and concurrency-friendly manner, retaining data integrity and consistency.

While other patterns such as Monads, the Actor Model, and Event Sourcing also provide mechanisms to deal with state changes, Vars and Refs uniquely enable mutable state management within the functional programming paradigm. Understanding and effectively utilising these patterns allows developers to build applications that are both robust and concurrent, all while following immutability principles.
