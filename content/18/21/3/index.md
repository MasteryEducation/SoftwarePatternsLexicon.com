---
linkTitle: "Circuit Breaker Pattern"
title: "Circuit Breaker Pattern: Preventing Cascading Failures"
category: "Resiliency and Fault Tolerance in Cloud"
series: "Cloud Computing: Essential Patterns & Practices"
description: "The Circuit Breaker Pattern is designed to prevent cascading failures and enhance system resiliency by temporarily halting calls to failing services."
categories:
- resiliency
- fault tolerance
- cloud patterns
tags:
- circuit breaker
- fault tolerance
- resiliency
- design pattern
- cloud computing
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/18/21/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

The Circuit Breaker Pattern is a crucial design pattern for achieving resiliency and fault tolerance in cloud-based architectures. It prevents systems from making repeated calls to a known failing service, thus averting potential cascading failures throughout the system. This pattern derives its name from electrical circuit breakers which halt the flow of electricity to prevent damage during overloads.

## Detailed Explanation

### Architectural Overview

The Circuit Breaker Pattern acts as a proxy for service calls. It maintains the current state of the external service:

1. **Closed**: Calls are allowed to pass through normally.
2. **Open**: Calls are blocked, and a fallback response is returned immediately.
3. **Half-Open**: Some calls are allowed to test if the service is back to its normal state.

Transition between these states involves tracking the number of failures and successes. When call failures reach a specified threshold, the circuit trips to an "Open" state. After a predetermined timeout, the circuit transitions to "Half-Open" to test the stability of the system before returning to "Closed."

### Implementation Strategy

1. **Failure Detection**: Monitor a series of request/response interactions.
2. **Threshold Configuration**: Establish a failure threshold to trigger the state transition.
3. **Timeout for Open State**: Set a duration after which the system will attempt to transition to a Half-Open state to test service recovery.
4. **Fallback Mechanism**: Define alternative strategies to handle requests during the Open state, such as cached data responses or error messaging.

### Example Code

Here's a sample implementation of the Circuit Breaker in Java using a specific framework like Resilience4j:

```java
CircuitBreakerConfig config = CircuitBreakerConfig.custom()
    .failureRateThreshold(50)
    .waitDurationInOpenState(Duration.ofSeconds(30))
    .permittedNumberOfCallsInHalfOpenState(3)
    .build();

CircuitBreaker circuitBreaker = CircuitBreaker.of("serviceName", config);

Supplier<String> decoratedSupplier = CircuitBreaker.decorateSupplier(circuitBreaker, this::callExternalService);

Try<String> result = Try.ofSupplier(decoratedSupplier)
    .recover(throwable -> "Fallback response");
```

### Related Patterns

- **Retry Pattern**: Complements Circuit Breaker by re-attempting failed operations at defined intervals.
- **Timeout Pattern**: Defines maximum duration for service requests to complete, crucial for avoiding indefinite waits during failures.
- **Bulkhead Pattern**: Isolates components to prevent a failure in one service from cascading across the entire system.

### Best Practices

- Determine appropriate thresholds and timeouts based on application behavior and SLAs.
- Implement logging and monitoring around Circuit Breaker events for insight into service performance and fault trends.
- Use in conjunction with other fault-tolerance patterns to enhance system resiliency.

### Additional Resources

- [Resilience4j Documentation](https://resilience4j.readme.io/docs/circuitbreaker)
- [Netflix Hystrix: Circuit Breaker Further Reading](https://github.com/Netflix/Hystrix/wiki)

## Final Summary

The Circuit Breaker Pattern is instrumental in creating robust cloud applications by managing interaction failures gracefully and preventing failures from cascading throughout the system. By implementing a stateful mechanism that monitors and governs service calls based on their success or failure rates, the Circuit Breaker Pattern enhances overall system reliability and user experience.

By adhering to this pattern, architects and developers ensure their systems can withstand and recover from unexpected disruptions, ultimately leading to more resilient, reliable cloud applications.
