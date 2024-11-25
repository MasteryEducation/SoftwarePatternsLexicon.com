---
linkTitle: "Sagas for Distributed Transactions"
title: "Sagas for Distributed Transactions"
category: "7. Polyglot Persistence Patterns"
series: "Data Modeling Design Patterns"
description: "Managing complex transactions across multiple services/databases using a sequence of local transactions with compensation."
categories:
- Transaction Management
- Distributed Systems
- Microservices Architecture
tags:
- Sagas
- Distributed Transactions
- Microservices
- Transaction Management
- Data Consistency
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/102/7/10"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---


## Introduction

In microservices architectures, where services are often decoupled and databases are distributed, achieving consistent data across multiple services during complex transactions can be challenging. The Saga Pattern offers a solution by breaking down a distributed transaction into a sequence of simpler, local transactions. Each step in the saga is executed independently, and if any step fails, compensatory transactions are executed to undo changes and maintain consistency.

## Problem Statement

Traditional two-phase commit (2PC) protocols become less feasible in distributed systems mainly due to their blocking nature and potential for performance bottlenecks. As a result, in distributed and loosely-coupled systems, we need an alternative approach that ensures data consistency without relying on lock-based coordination among multiple services. This is where the Saga Pattern comes into play.

## Pattern Overview

The Saga Pattern ensures data consistency in distributed systems by splitting a transaction into a sequence of smaller, independent transactions. These local transactions are coordinated in a defined sequence. The pattern defines two types of saga execution:

1. **Success Path**: If all steps succeed, the saga completes normally, and the system transitions to the new consistent state.
2. **Compensation Path**: If a step fails, previously completed steps are compensated using predefined compensatory transactions that revert the state to maintain consistency.

### Types of Sagas

- **Choreography-based Sagas**: Each service involved in the saga independently emits an event upon completing its transaction, which triggers the next step. This approach favors decentralized control but can lead to complex event flows.
- **Orchestration-based Sagas**: A central orchestrator (also known as a saga coordinator) dictates the sequence of transaction steps and their compensation logic. This approach provides centralized control, simplifying the flow but introducing a single point of control.

## Example Use Case

Consider an order processing system with multiple services:

1. **Order Service**: Creates the order.
2. **Payment Service**: Processes payment.
3. **Inventory Service**: Allocates inventory.
4. **Shipping Service**: Arranges shipping.

These services have individual databases, and each transaction occurs locally. If the Payment Service fails, the Order Service initiates a compensatory transaction to cancel the order.

### Example Code (Pseudocode)

```java
saga {
  createOrder();
  try {
    processPayment();
    allocateInventory();
    arrangeShipping();
  } catch (Exception e) {
    compensate(orderId);
  }
}

compensate(int orderId) {
  cancelShipping(orderId);
  deallocateInventory(orderId);
  refundPayment(orderId);
  cancelOrder(orderId);
}
```

## Related Patterns

- **Command Query Responsibility Segregation (CQRS)**: Complement Sagas by separating reads from writes, ensuring read operations access eventual consistent state.
- **Event Sourcing**: Records every state change as an event, allowing recovery with saga compensation logic in failure scenarios.

## Best Practices

- Always define compensation transactions for every successful segment of the saga to ensure system recoverability.
- Handle idempotency in service operations to manage duplicate messages.
- Use logging and monitoring to detect and manage failures quickly.
- Aim to minimize the time window for incomplete sagas to reduce resource locks and inconsistent states.

## Additional Resources

- "Building Microservices" by Sam Newman
- "Enterprise Integration Patterns: Designing, Building, and Deploying Messaging Solutions" by Gregor Hohpe and Bobby Woolf
- Saga Pattern documentation on various cloud provider platforms

## Summary

The Saga Pattern is a vital approach for managing distributed transactions in a microservices architecture. By breaking down complex transactions into a sequence of local transactions, sagas allow systems to maintain consistency without the drawbacks of traditional transactions. While offering flexibility, the pattern necessitates careful planning around compensation logic and service orchestration to manage robust workflows across distributed systems.
