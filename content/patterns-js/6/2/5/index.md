---
linkTitle: "6.2.5 Circuit Breaker"
title: "Circuit Breaker Pattern in JavaScript and TypeScript: Ensuring Resilient Applications"
description: "Explore the Circuit Breaker pattern in JavaScript and TypeScript to enhance application resilience by preventing repeated failures and managing service dependencies effectively."
categories:
- Software Design Patterns
- Enterprise Integration Patterns
- JavaScript
tags:
- Circuit Breaker
- Resilience
- JavaScript
- TypeScript
- Node.js
date: 2024-10-25
type: docs
nav_weight: 625000
canonical: "https://softwarepatternslexicon.com/patterns-js/6/2/5"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 6.2.5 Circuit Breaker

In modern software development, especially in distributed systems, ensuring application resilience is crucial. The Circuit Breaker pattern is a vital tool in achieving this resilience by preventing an application from repeatedly attempting operations that are likely to fail. This article delves into the Circuit Breaker pattern, its implementation in JavaScript and TypeScript, and its importance in maintaining robust applications.

### Understand the Concept

The Circuit Breaker pattern is inspired by electrical circuit breakers, which prevent electrical overloads by interrupting the flow of electricity. Similarly, in software, a Circuit Breaker monitors the success and failure rates of operations, such as network calls, and interrupts the flow of requests to prevent further failures when a threshold is reached.

#### Key Components of the Circuit Breaker Pattern

1. **Closed State:** The circuit is closed, and requests are allowed to pass through. The system monitors the success and failure rates.
2. **Open State:** The circuit opens when failures exceed a predefined threshold, blocking further requests for a specified period.
3. **Half-Open State:** After the open state, the circuit transitions to half-open, allowing a limited number of test requests to determine if the issue has been resolved.

### Implementation Steps

#### 1. Implement Circuit Breaker Logic

You can implement the Circuit Breaker pattern using libraries like `opossum` in Node.js or by writing custom logic. The library simplifies the process by providing built-in mechanisms to monitor and manage circuit states.

#### 2. Define Thresholds

Set thresholds for failure rates that will trigger the circuit to open. These thresholds can be based on the number of consecutive failures or a percentage of failed requests over a time period.

#### 3. Handle State Transitions

Manage the transitions between the Closed, Open, and Half-Open states. This involves defining the conditions for each transition and implementing logic to handle these changes.

#### 4. Fallback Mechanisms

When the circuit is open, provide alternative responses or behaviors, such as returning cached data or default values, to maintain a level of service availability.

### Code Examples

Let's explore how to implement a Circuit Breaker using the `opossum` library in Node.js.

```javascript
const CircuitBreaker = require('opossum');

// Define a function that makes a network request
async function fetchData() {
  // Simulate a network call
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      // Randomly succeed or fail
      Math.random() > 0.5 ? resolve('Data fetched successfully') : reject('Network error');
    }, 1000);
  });
}

// Create a circuit breaker for the fetchData function
const breaker = new CircuitBreaker(fetchData, {
  timeout: 3000, // If the request takes longer than 3 seconds, consider it a failure
  errorThresholdPercentage: 50, // Open the circuit if 50% of requests fail
  resetTimeout: 5000 // After 5 seconds, try again
});

// Listen for circuit breaker events
breaker.on('open', () => console.log('Circuit is open'));
breaker.on('halfOpen', () => console.log('Circuit is half-open'));
breaker.on('close', () => console.log('Circuit is closed'));

// Use the circuit breaker to make a request
breaker.fire()
  .then(console.log)
  .catch(console.error);
```

### Use Cases

The Circuit Breaker pattern is particularly useful in scenarios where services depend on external APIs or microservices. It helps prevent cascading failures by stopping requests to a failing service, allowing it time to recover.

#### Real-World Scenario

Consider an e-commerce application that relies on a payment gateway. If the payment gateway experiences downtime, the Circuit Breaker pattern can prevent the application from continuously attempting failed transactions, thus maintaining overall system stability.

### Practice

To practice implementing a Circuit Breaker, try applying it to an external API call in your application. Monitor the behavior of the circuit under different conditions and adjust the thresholds to suit your application's needs.

### Considerations

- **Monitoring and Logging:** Implement logging for circuit breaker events to gain insights into system behavior and identify potential issues.
- **Testing:** Test your system's behavior when the circuit is open to ensure that fallback mechanisms provide adequate service levels.

### Advantages and Disadvantages

#### Advantages

- **Improved Resilience:** Prevents repeated failures and allows systems to recover gracefully.
- **Enhanced Stability:** Protects services from cascading failures in distributed systems.
- **Operational Awareness:** Provides insights into system health through monitoring and logging.

#### Disadvantages

- **Complexity:** Adds complexity to the system, requiring careful configuration and monitoring.
- **Latency:** May introduce latency due to state transitions and fallback mechanisms.

### Best Practices

- **Set Appropriate Thresholds:** Carefully define failure thresholds based on your application's tolerance for errors.
- **Monitor Performance:** Continuously monitor circuit breaker events and adjust configurations as needed.
- **Implement Fallbacks:** Ensure that fallback mechanisms provide meaningful responses to maintain user experience.

### Conclusion

The Circuit Breaker pattern is a powerful tool for enhancing the resilience of applications, especially in distributed systems. By preventing repeated failures and managing service dependencies effectively, it helps maintain system stability and improve user experience. Implementing this pattern in JavaScript and TypeScript, particularly with libraries like `opossum`, can significantly contribute to building robust applications.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of the Circuit Breaker pattern?

- [x] To prevent an application from repeatedly trying to execute an operation that's likely to fail
- [ ] To increase the speed of network requests
- [ ] To manage user authentication
- [ ] To optimize database queries

> **Explanation:** The Circuit Breaker pattern is designed to detect failures and prevent an application from repeatedly attempting operations that are likely to fail, thus enhancing resilience.

### Which library is commonly used in Node.js to implement the Circuit Breaker pattern?

- [x] opossum
- [ ] express
- [ ] lodash
- [ ] axios

> **Explanation:** The `opossum` library is specifically designed for implementing the Circuit Breaker pattern in Node.js applications.

### What are the three states of a Circuit Breaker?

- [x] Closed, Open, Half-Open
- [ ] Start, Stop, Pause
- [ ] Active, Inactive, Pending
- [ ] On, Off, Standby

> **Explanation:** The Circuit Breaker pattern involves three states: Closed (normal operation), Open (requests are blocked), and Half-Open (limited requests are allowed to test recovery).

### What happens when the Circuit Breaker is in the Open state?

- [x] Requests are blocked
- [ ] Requests are allowed
- [ ] Requests are queued
- [ ] Requests are logged

> **Explanation:** In the Open state, the Circuit Breaker blocks requests to prevent further failures and allow the system to recover.

### What is a common fallback mechanism when the Circuit Breaker is open?

- [x] Returning cached data
- [ ] Increasing request frequency
- [ ] Disabling the service
- [ ] Redirecting to a different API

> **Explanation:** A common fallback mechanism is to return cached data or default values to maintain service availability when the Circuit Breaker is open.

### How does the Circuit Breaker pattern enhance system resilience?

- [x] By preventing repeated failures and allowing systems to recover gracefully
- [ ] By increasing the number of concurrent requests
- [ ] By reducing the need for error handling
- [ ] By simplifying the codebase

> **Explanation:** The Circuit Breaker pattern enhances resilience by stopping repeated failures and allowing the system time to recover, thus maintaining stability.

### What should be monitored to gain insights into system behavior with Circuit Breakers?

- [x] Circuit breaker events
- [ ] User login attempts
- [ ] Database query times
- [ ] File system access

> **Explanation:** Monitoring circuit breaker events provides valuable insights into system health and helps identify potential issues.

### What is the role of the Half-Open state in a Circuit Breaker?

- [x] To test if the issue has been resolved by allowing limited requests
- [ ] To block all requests indefinitely
- [ ] To reset the system to its initial state
- [ ] To log all incoming requests

> **Explanation:** The Half-Open state allows a limited number of test requests to determine if the underlying issue has been resolved before fully closing the circuit.

### Why is it important to set appropriate failure thresholds in a Circuit Breaker?

- [x] To ensure the circuit opens only when necessary
- [ ] To increase the number of successful requests
- [ ] To reduce the complexity of the system
- [ ] To enhance user experience

> **Explanation:** Setting appropriate failure thresholds ensures that the circuit opens only when necessary, preventing unnecessary interruptions and maintaining system stability.

### True or False: The Circuit Breaker pattern can introduce latency due to state transitions and fallback mechanisms.

- [x] True
- [ ] False

> **Explanation:** The Circuit Breaker pattern can introduce latency as it involves state transitions and fallback mechanisms, which may add some overhead to the system.

{{< /quizdown >}}
