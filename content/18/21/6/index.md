---
linkTitle: "Retries with Exponential Backoff"
title: "Retries with Exponential Backoff: Resiliency and Fault Tolerance in Cloud"
category: "Resiliency and Fault Tolerance in Cloud"
series: "Cloud Computing: Essential Patterns & Practices"
description: "Learn about the Retries with Exponential Backoff pattern, a strategy that involves retrying failed operations with progressively increasing wait times. This approach helps in avoiding overwhelming a failing service and stabilizes system performance in cloud environments."
categories:
- Cloud Patterns
- Fault Tolerance
- Resiliency
tags:
- Cloud Computing
- Fault Tolerance
- Resiliency
- Exponential Backoff
- Retry Patterns
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/18/21/6"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Overview

The **Retries with Exponential Backoff** pattern is a retry strategy designed to handle transient failures in distributed systems and cloud services. By retrying failed operations with exponentially increasing wait times, this pattern minimizes the risk of overwhelming a struggling service. This behavior is particularly beneficial when dealing with rate-limited APIs or services experiencing temporary overloads.

## Explanation of the Pattern

In cloud-based applications, transient failures such as temporary network issues or service overloading are common. A straightforward retry strategy where requests are sent again immediately might aggravate the issue by flooding the service with requests. Instead, **Retries with Exponential Backoff** provides a disciplined approach by introducing progressively longer wait periods between retries. This allows the service time to recover and stabilize.

### Exponential Backoff Algorithm

The exponential backoff algorithm uses a simple formula:

{{< katex >}}
\text{Backoff Time} = \text{Base Delay} \times (2^{\text{retry attempt}})
{{< /katex >}}

- **Base Delay**: The initial delay before the first retry.
- **Retry Attempt**: The current attempt number.

For instance, if the base delay is 100 milliseconds and it needs three retries, the wait times would be 100ms, 200ms, 400ms, and so on.

### Jitter

Adding jitter—randomness to the backoff duration—can further improve the system's ability to deal with failures by dispersing requests across time, reducing the likelihood of a large number of retries causing another round of congestion.

### Example Code

Here's a basic implementation in Java, using Exponential Backoff with Jitter:

```java
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class ExponentialBackoffRetry {

    private static final int MAX_RETRY_ATTEMPTS = 5;
    private static final long BASE_DELAY = 100; // in milliseconds
    private static final Random RANDOM = new Random();

    public static void main(String[] args) {
        ExponentialBackoffRetry retryExample = new ExponentialBackoffRetry();
        retryExample.performActionWithRetry();
    }

    public void performActionWithRetry() {
        int attempt = 0;
        while (attempt < MAX_RETRY_ATTEMPTS) {
            try {
                // Perform the operation
                performOperation();
                return; // Exit if successful
            } catch (Exception e) {
                attempt++;
                long backoffTime = calculateExponentialBackoff(attempt);
                System.out.println("Retrying in " + backoffTime + "ms... (" + attempt + " attempt)");
                try {
                    TimeUnit.MILLISECONDS.sleep(backoffTime);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("Thread interrupted during backoff", ie);
                }
            }
        }
        System.out.println("Max retry attempts reached.");
    }

    private void performOperation() throws Exception {
        // Simulate an operation that can fail
        if (RANDOM.nextInt(10) < 8) { // 80% chance to fail
            throw new Exception("Transient failure.");
        }
        System.out.println("Operation succeeded.");
    }

    private long calculateExponentialBackoff(int attempt) {
        long exponentialTime = BASE_DELAY * (1L << attempt);
        // Add jitter
        return exponentialTime / 2 + RANDOM.nextInt((int) exponentialTime / 2);
    }
}
```

### Advantages

- **Prevents Overload**: Mitigates the risk of overwhelming the service by spreading out retries.
- **Improves Stability**: Allows time for transient problems to resolve before another attempt.
- **Graceful Error Handling**: Provides a systematic way to handle failures, improving user experience.

### Best Practices

- **Set a Maximum Retry Limit**: Avoid infinite retries by setting a finite number of attempts.
- **Monitor and Log Retries**: Collect retry attempts and failures for better observability.
- **Adjust Base Delay and Max Attempts**: Tune these parameters based on the specific application requirements and SLAs.
- **Consider Jitter**: This can prevent synchronized retry spikes across multiple clients.

### Related Patterns

- **Circuit Breaker**: Stops the execution of a function when failure is probable.
- **Bulkhead**: Isolates failures in parts of a system to prevent cascading effects.

## Additional Resources

- AWS Architecture Blog on retry strategies
- Google's Cloud APIs docs on handling retries
- Martin Fowler's article on patterns of reliability

## Summary

The **Retries with Exponential Backoff** pattern is a crucial retry strategy in cloud environments, where managing transient failures effectively can lead to improved system resilience and user experience. By gradually increasing wait times and dispersing retries with jitter, applications can reduce the load on services and avoid cascading failures. This pattern, when combined with others like Circuit Breaker and Bulkhead, can fortify cloud applications against transient and partial failures while providing responsive and reliable user interactions.
