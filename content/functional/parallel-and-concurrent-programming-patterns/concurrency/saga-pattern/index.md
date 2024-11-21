---
linkTitle: "Saga Pattern"
title: "Saga Pattern: Coordination of Long-Running Business Transactions"
description: "An approach to managing distributed transactions by breaking them into a sequence of smaller, independent operations, each with its own compensating transaction."
categories:
- Design Patterns
- Functional Programming
tags:
- Saga Pattern
- Distributed Systems
- Compensating Transactions
- Business Logic
- Functional Programming
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/parallel-and-concurrent-programming-patterns/concurrency/saga-pattern"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

The Saga Pattern is a design strategy used to manage complex, long-running business processes and distributed transactions by splitting them into a series of smaller, isolated operations. Each operation is designed to allow a system to fail and recover gracefully without losing consistency by employing compensation logic or rollbacks.

In distributed systems, achieving global ACID (Atomicity, Consistency, Isolation, Durability) properties can be challenging, primarily due to the time frames and independence of involved services. The Saga Pattern circumvents this by defining a sequence of transactions where each transaction updates the system. If a step fails, compensating transactions (steps) will undo the subsequent steps maintaining the system's consistency.

## Core Concepts

### Transactions and Compensation

- **Transactions**: A unit of work that modifies the state of a service and should be completed successfully or rolled back to maintain consistency.
  
- **Compensating Transactions**: These are necessary actions designed to undo the effects of a previously completed transaction when a failure occurs later in the saga.

### Saga Execution Types

1. **Choreography**: In this approach, each service in the saga is aware of its role and maintains autonomy by deciding when to execute its transactions and compensations by listening to and emitting events.
   
2. **Orchestration**: This utilizes a central coordinator that sequences and controls the saga's execution. The orchestrator decides the order of operations and compensations.
   ```mermaid
   sequenceDiagram
       participant Coordinator
       participant Service1
       participant Service2
       participant Service3

       Coordinator->>Service1: Execute Transaction T1
       Service1-->>Coordinator: T1 Done
       Coordinator->>Service2: Execute Transaction T2
       Service2-->>Coordinator: T2 Done
       Coordinator->>Service3: Execute Transaction T3
       Service3-->>Coordinator: T3 Done
       Service3-->>Coordinator: Error/Failure
       Coordinator->>Service2: Execute Compensation C2
       Service2-->>Coordinator: C2 Completed
       Coordinator->>Service1: Execute Compensation C1
       Service1-->>Coordinator: C1 Completed
   ```

### Example Scenario

Consider an e-commerce application where a multi-step process is required to complete an order:

1. **Reserve Inventory**: Ensure that the items are available and reserve them for the order.
2. **Process Payment**: Charge the customer for the order.
3. **Arrange Shipment**: Notify the shipping service to deliver the items.

If creating the shipment fails, previous steps need to be undone:

- **Arrange Shipment** fails → Compensate by **Refunding Payment**.
- **Refund Payment** succeeds → Compensate by **Releasing Inventory Reservation**.

```katex
\begin{array}{ll}
\text{Saga Steps:} & \text{Compensating Transactions:} \\
1. \text{Reserve Inventory} & 1. \text{Release Inventory Reservation} \\
2. \text{Process Payment} & 2. \text{Refund Payment} \\
3. \text{Arrange Shipment} & \text{No compensation needed since shipment failed} \\
\end{array}
```

## Related Design Patterns

### Command Query Responsibility Segregation (CQRS)

CQRS separates the write model (commands) from the read model (queries), allowing independent scaling, optimization, and tackling eventual consistency challenges, which is often seen in conjunction with sagas.

### Event Sourcing

Instead of storing just the current state, event sourcing logs as sequence of state-changing events. This fits well with the Saga Pattern's need for historical tracking of compensation transactions.

### Transactional Outbox Pattern

To ensure exact-once message delivery, the transactional outbox pattern helps encapsulate side-effects within a reliable, self-contained, transactional unit, thus fitting well within an orchestrated saga.

## Additional Resources

- "Designing Data-Intensive Applications" by Martin Kleppmann.
- [Microservices Patterns](https://microservices.io/) by Chris Richardson.
- [The Saga Pattern](https://www.baeldung.com/saga-pattern) on Baeldung.
- [Saga Pattern: Handling Distributed Transactions](https://dzone.com/articles/saga-pattern-implementing-business-transactions-usi) on DZone.

## Summary

The Saga Pattern is crucial for managing consistency and recovery in long-running and distributed business transactions. By choreographing or orchestrating smaller, well-defined units of work, it deals effectively with system failure and offers a robust alternative to traditional, rigid ACID transactions. Understanding and implementing sagas enable developers to build resilient systems addressing the multi-faceted consistency challenges presented by microservices.

By decomposing business logic into isolated steps and appropriately compensating for failures, the Saga Pattern ensures system robustness and fault tolerance, making it a vital design choice in modern functional programming paradigms.


