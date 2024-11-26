---
canonical: "https://softwarepatternslexicon.com/patterns-erlang/30/1"
title: "Glossary of Terms: Erlang Design Patterns and Functional Programming"
description: "Explore a comprehensive glossary of terms related to Erlang, OTP, design patterns, and functional programming, providing clear and concise definitions for expert developers."
linkTitle: "30.1 Glossary of Terms"
categories:
- Erlang
- Functional Programming
- Design Patterns
tags:
- Erlang
- OTP
- Design Patterns
- Functional Programming
- Concurrency
date: 2024-11-23
type: docs
nav_weight: 301000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 30.1 Glossary of Terms

This glossary provides definitions of key technical terms and concepts used throughout the guide. It serves as a quick reference for readers to understand the terminology related to Erlang, OTP (Open Telecom Platform), design patterns, and functional programming. Each term is explained clearly and concisely, with cross-references to relevant sections in the guide where applicable.

### A

- **Actor Model**: A conceptual model for concurrent computation in which "actors" are the universal primitives of concurrent computation. In Erlang, processes are actors that communicate via message passing. See [Section 4.1: The Actor Model and Erlang Processes](#4.1-the-actor-model-and-erlang-processes).

- **Anonymous Function**: A function defined without a name, often used for short-lived operations. In Erlang, they are created using the `fun` keyword. See [Section 2.5: Anonymous Functions and Closures](#2.5-anonymous-functions-and-closures).

```erlang
% Example of an anonymous function
Square = fun(X) -> X * X end.
```

- **Applicative Order**: An evaluation strategy where the arguments to a function are evaluated before the function itself is applied. This is the default strategy in Erlang.

### B

- **BEAM**: The virtual machine at the core of the Erlang runtime system, responsible for executing compiled Erlang code. It supports concurrency, distribution, and fault tolerance.

- **Behaviour**: A design pattern in Erlang used to define a set of functions that a module must implement. Common behaviours include `gen_server`, `gen_statem`, and `supervisor`. See [Section 6.2: OTP Behaviours](#6.2-otp-behaviours).

- **Bitstring**: A data type in Erlang used to handle sequences of bits. It is often used for binary data manipulation. See [Section 3.1.3: Binaries and Bitstrings](#3.1.3-binaries-and-bitstrings).

### C

- **Closure**: A function along with a referencing environment for the non-local variables of that function. Closures allow functions to capture variables from their surrounding scope. See [Section 2.5: Anonymous Functions and Closures](#2.5-anonymous-functions-and-closures).

- **Concurrency**: The ability of a system to handle multiple tasks simultaneously. Erlang's concurrency model is based on lightweight processes and message passing. See [Section 4: Concurrency in Erlang](#4-concurrency-in-erlang).

- **Currying**: The process of transforming a function that takes multiple arguments into a sequence of functions, each with a single argument. See [Section 11.2: Currying and Partial Application](#11.2-currying-and-partial-application).

### D

- **Dialyzer**: A static analysis tool for Erlang that identifies software discrepancies such as type errors, unreachable code, and unnecessary tests. See [Section 3.9: Typespecs and Dialyzer for Static Analysis](#3.9-typespecs-and-dialyzer-for-static-analysis).

- **Distributed Erlang**: A feature of Erlang that allows multiple Erlang runtime systems to communicate and work together as a single system. See [Section 5: Distributed Programming in Erlang](#5-distributed-programming-in-erlang).

- **Dynamic Typing**: A type system where type checking is performed at runtime, allowing for more flexibility in code but requiring careful runtime error handling.

### E

- **ETS (Erlang Term Storage)**: An in-memory storage system for Erlang terms, providing fast access to large amounts of data. See [Section 7.13: ETS: In-Memory Storage Patterns](#7.13-ets-in-memory-storage-patterns).

- **EUnit**: A unit testing framework for Erlang, used to write and execute tests to ensure code correctness. See [Section 13.13: Testing with EUnit and Common Test](#13.13-testing-with-eunit-and-common-test).

- **Erlang Shell (REPL)**: An interactive command-line interface for executing Erlang expressions and testing code snippets. See [Section 3.6: Effective Use of the Erlang Shell (REPL)](#3.6-effective-use-of-the-erlang-shell-repl).

### F

- **Factory Pattern**: A creational design pattern used to create objects without specifying the exact class of object that will be created. In Erlang, this can be implemented using functions and modules. See [Section 8.2: Factory Pattern with Functions and Modules](#8.2-factory-pattern-with-functions-and-modules).

- **Fault Tolerance**: The ability of a system to continue operating properly in the event of the failure of some of its components. Erlang's "let it crash" philosophy supports building fault-tolerant systems. See [Section 2.8: The "Let It Crash" Philosophy](#2.8-the-let-it-crash-philosophy).

- **First-Class Function**: A function that can be treated like any other variable, passed as an argument, returned from a function, and assigned to a variable. See [Section 2.2: First-Class and Higher-Order Functions](#2.2-first-class-and-higher-order-functions).

### G

- **Gen_Server**: A generic server behaviour in OTP that abstracts the common patterns of a server process, simplifying the implementation of concurrent servers. See [Section 6.4: Implementing Servers with `gen_server`](#6.4-implementing-servers-with-gen_server).

- **Guard**: A boolean expression used in pattern matching to add additional constraints. Guards enhance pattern matching by allowing more complex conditions. See [Section 2.3: Pattern Matching and Guards](#2.3-pattern-matching-and-guards).

### H

- **Higher-Order Function**: A function that takes other functions as arguments or returns a function as a result. Higher-order functions are a key feature of functional programming. See [Section 2.2: First-Class and Higher-Order Functions](#2.2-first-class-and-higher-order-functions).

- **Hot Code Upgrade**: The ability to update a running system with new code without stopping it. Erlang supports hot code upgrades through its release handling mechanisms. See [Section 6.10: Hot Code Upgrades and `release_handler`](#6.10-hot-code-upgrades-and-release_handler).

### I

- **Immutability**: A property of data that prevents it from being modified after creation. In Erlang, all data structures are immutable, which simplifies reasoning about code and enhances concurrency. See [Section 2.1: Immutability and Pure Functions](#2.1-immutability-and-pure-functions).

- **Interpreter Pattern**: A design pattern used to evaluate sentences in a language. In Erlang, this can be implemented using custom parsers. See [Section 10.8: Interpreter Pattern using Custom Parsers](#10.8-interpreter-pattern-using-custom-parsers).

### J

- **JSON (JavaScript Object Notation)**: A lightweight data interchange format that is easy for humans to read and write and easy for machines to parse and generate. Erlang provides libraries for JSON serialization and deserialization. See [Section 13.6: Data Serialization and Deserialization (JSON, BSON)](#13.6-data-serialization-and-deserialization-json-bson).

### K

- **Keyword List**: A list of tuples where the first element is an atom, often used for passing options to functions in Erlang.

### L

- **Lazy Evaluation**: An evaluation strategy that delays the computation of expressions until their values are needed. Erlang does not natively support lazy evaluation, but patterns can be implemented to achieve similar behavior. See [Section 11.4: Lazy Evaluation Patterns](#11.4-lazy-evaluation-patterns).

- **List Comprehension**: A concise way to create lists in Erlang, using a syntax similar to set notation in mathematics. See [Section 3.4: List Comprehensions and Generators](#3.4-list-comprehensions-and-generators).

```erlang
% Example of a list comprehension
Squares = [X*X || X <- lists:seq(1, 10)].
```

### M

- **Macro**: A compile-time construct that allows code to be generated programmatically. In Erlang, macros are defined using the `-define` directive. See [Section 2.9: Introduction to Macros and Parse Transformations](#2.9-introduction-to-macros-and-parse-transformations).

- **Message Passing**: The primary method of communication between processes in Erlang, where processes send and receive messages asynchronously. See [Section 4.2: Message Passing and Process Communication](#4.2-message-passing-and-process-communication).

- **Mnesia**: A distributed database management system built into Erlang, designed for high availability and fault tolerance. See [Section 13.2: Distributed Databases with Mnesia](#13.2-distributed-databases-with-mnesia).

### N

- **Node**: An instance of the Erlang runtime system that can communicate with other nodes in a distributed system. See [Section 5.2: Node Communication and Connectivity](#5.2-node-communication-and-connectivity).

- **NIF (Native Implemented Function)**: A mechanism for writing Erlang functions in C for performance-critical operations. See [Section 14.2: Using Ports, NIFs, and C Nodes for Native Integration](#14.2-using-ports-nifs-and-c-nodes-for-native-integration).

### O

- **OTP (Open Telecom Platform)**: A set of Erlang libraries and design principles for building applications. It includes tools for building concurrent, fault-tolerant, and distributed systems. See [Section 6: OTP Design Principles and Patterns](#6-otp-design-principles-and-patterns).

- **Observer Pattern**: A behavioral design pattern where an object, known as the subject, maintains a list of its dependents, called observers, and notifies them of any state changes. See [Section 10.2: Observer Pattern with Publish/Subscribe](#10.2-observer-pattern-with-publish-subscribe).

### P

- **Pattern Matching**: A mechanism for checking a value against a pattern. It is a fundamental feature in Erlang, used in variable binding, function clauses, and case expressions. See [Section 2.3: Pattern Matching and Guards](#2.3-pattern-matching-and-guards).

- **Process**: The basic unit of concurrency in Erlang, lightweight and isolated, communicating with other processes via message passing. See [Section 4.1: The Actor Model and Erlang Processes](#4.1-the-actor-model-and-erlang-processes).

- **Prototype Pattern**: A creational design pattern used to create objects based on a template of an existing object through cloning. In Erlang, this can be achieved through process cloning. See [Section 8.5: Prototype Pattern through Process Cloning](#8.5-prototype-pattern-through-process-cloning).

### Q

- **Quorum**: A minimum number of votes that must be obtained to make a decision in distributed systems, often used in consensus algorithms.

### R

- **Recursion**: A technique where a function calls itself to solve a problem. Erlang relies heavily on recursion due to its immutable data structures. See [Section 2.4: Recursion and Tail Call Optimization](#2.4-recursion-and-tail-call-optimization).

- **Release Handling**: The process of managing application versions, including upgrades and downgrades, in a running system. See [Section 6.10: Hot Code Upgrades and `release_handler`](#6.10-hot-code-upgrades-and-release_handler).

- **Registry Pattern**: A design pattern used to store and retrieve objects by name. In Erlang, this can be implemented using processes. See [Section 8.7: Registry Pattern with Erlang Processes](#8.7-registry-pattern-with-erlang-processes).

### S

- **Supervisor**: An OTP behaviour used to monitor and manage other processes, restarting them if they fail. See [Section 6.3: Designing Fault-Tolerant Systems with Supervisors](#6.3-designing-fault-tolerant-systems-with-supervisors).

- **Synchronous Messaging**: A communication method where the sender waits for a response from the receiver before continuing execution. See [Section 4.3: Synchronous vs. Asynchronous Messaging](#4.3-synchronous-vs-asynchronous-messaging).

- **State Pattern**: A behavioral design pattern that allows an object to alter its behavior when its internal state changes. In Erlang, this can be implemented using `gen_statem`. See [Section 10.6: State Pattern with `gen_statem`](#10.6-state-pattern-with-gen_statem).

### T

- **Tail Call Optimization**: An optimization technique where the last call in a function is optimized to avoid adding a new stack frame, allowing for efficient recursion. See [Section 2.4: Recursion and Tail Call Optimization](#2.4-recursion-and-tail-call-optimization).

- **Tuple**: A fixed-size collection of values in Erlang, used to group related data. Tuples are immutable and can contain elements of different types. See [Section 3.1.1: Lists, Tuples, and Arrays](#3.1.1-lists-tuples-and-arrays).

### U

- **Uptime**: The amount of time a system has been running without interruption. Erlang systems are designed for high uptime through fault tolerance and hot code upgrades.

### V

- **Variable Binding**: The process of associating a variable with a value. In Erlang, once a variable is bound to a value, it cannot be changed.

- **Visitor Pattern**: A behavioral design pattern that allows adding new operations to existing object structures without modifying them. In Erlang, this can be implemented using behaviours. See [Section 10.10: Visitor Pattern via Behaviours](#10.10-visitor-pattern-via-behaviours).

### W

- **Worker**: A process that performs a specific task, often managed by a supervisor in an OTP application.

### X

- **XML (Extensible Markup Language)**: A markup language used for encoding documents in a format that is both human-readable and machine-readable. Erlang provides libraries for XML parsing and generation.

### Y

- **Yield**: The act of a process voluntarily giving up control to allow other processes to execute. In Erlang, processes are scheduled by the BEAM VM, and explicit yielding is not typically required.

### Z

- **Zero-Downtime Deployment**: A deployment strategy that allows updates to be made to a system without interrupting service availability. Erlang supports zero-downtime deployments through hot code upgrades. See [Section 22.4: Zero-Downtime Deployments and Rolling Upgrades](#22.4-zero-downtime-deployments-and-rolling-upgrades).

---

## Quiz: Glossary of Terms

{{< quizdown >}}

### What is the primary method of communication between processes in Erlang?

- [x] Message Passing
- [ ] Shared Memory
- [ ] Sockets
- [ ] Remote Procedure Calls

> **Explanation:** Erlang processes communicate through message passing, which is asynchronous and does not require shared memory.

### Which Erlang feature allows for updating a running system with new code without stopping it?

- [x] Hot Code Upgrade
- [ ] Cold Restart
- [ ] Code Reload
- [ ] System Reboot

> **Explanation:** Hot code upgrades in Erlang allow for updating code in a running system without downtime.

### What is the purpose of the `gen_server` behaviour in Erlang?

- [x] To simplify the implementation of concurrent servers
- [ ] To manage database connections
- [ ] To handle HTTP requests
- [ ] To perform mathematical calculations

> **Explanation:** The `gen_server` behaviour abstracts common patterns of a server process, making it easier to implement concurrent servers.

### What does ETS stand for in Erlang?

- [x] Erlang Term Storage
- [ ] Erlang Transaction System
- [ ] Erlang Testing Suite
- [ ] Erlang Telecommunication Service

> **Explanation:** ETS stands for Erlang Term Storage, which is used for in-memory storage of Erlang terms.

### Which pattern is used to create objects based on a template of an existing object?

- [x] Prototype Pattern
- [ ] Singleton Pattern
- [ ] Factory Pattern
- [ ] Builder Pattern

> **Explanation:** The Prototype Pattern is used to create objects by cloning an existing object.

### What is the BEAM in Erlang?

- [x] The virtual machine that executes Erlang code
- [ ] A type of data structure
- [ ] A concurrency model
- [ ] A testing framework

> **Explanation:** BEAM is the virtual machine at the core of the Erlang runtime system, responsible for executing compiled Erlang code.

### What is the "let it crash" philosophy in Erlang?

- [x] A fault-tolerance approach where processes are allowed to fail and restart
- [ ] A debugging technique
- [ ] A performance optimization strategy
- [ ] A design pattern for concurrency

> **Explanation:** The "let it crash" philosophy in Erlang embraces process failures and relies on supervisors to restart failed processes, enhancing fault tolerance.

### What is a closure in Erlang?

- [x] A function with its referencing environment
- [ ] A type of data structure
- [ ] A concurrency model
- [ ] A testing framework

> **Explanation:** A closure is a function along with its referencing environment, allowing it to capture variables from its surrounding scope.

### What is the primary purpose of the OTP framework in Erlang?

- [x] To provide libraries and design principles for building concurrent, fault-tolerant systems
- [ ] To manage database connections
- [ ] To handle HTTP requests
- [ ] To perform mathematical calculations

> **Explanation:** OTP provides a set of libraries and design principles for building robust, concurrent, and fault-tolerant systems in Erlang.

### Erlang processes are considered lightweight because they:

- [x] Have low memory overhead and fast context switching
- [ ] Share memory with other processes
- [ ] Run on separate physical machines
- [ ] Require complex synchronization mechanisms

> **Explanation:** Erlang processes are lightweight due to their low memory overhead and efficient context switching, allowing for high concurrency.

{{< /quizdown >}}

Remember, this glossary is just the beginning. As you progress through the guide, you'll encounter these terms in context, deepening your understanding of Erlang's powerful features and design patterns. Keep experimenting, stay curious, and enjoy the journey!
