---
canonical: "https://softwarepatternslexicon.com/patterns-java/10/3/3/3"

title: "Semaphore in Java: Mastering Concurrency Control"
description: "Explore Java's Semaphore for effective concurrency control, managing permits, and ensuring resource access regulation."
linkTitle: "10.3.3.3 Semaphore"
tags:
- "Java"
- "Concurrency"
- "Semaphore"
- "Synchronization"
- "Multithreading"
- "Java.util.concurrent"
- "Resource Management"
- "Concurrency Patterns"
date: 2024-11-25
type: docs
nav_weight: 103330
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 10.3.3.3 Semaphore

### Introduction to Semaphores

In the realm of concurrent programming, managing access to shared resources is crucial to ensure data integrity and application stability. A **Semaphore** is a synchronization construct that controls access to a shared resource by maintaining a set of permits. It is a part of the `java.util.concurrent` package, which provides high-level concurrency utilities in Java.

Semaphores can be visualized as counters that regulate the number of threads that can access a particular resource simultaneously. When a thread wants to access the resource, it must acquire a permit from the semaphore. If no permits are available, the thread is blocked until one becomes available. Once the thread is done with the resource, it releases the permit back to the semaphore.

### Historical Context

The concept of semaphores was introduced by Edsger Dijkstra in the 1960s as a means to solve synchronization problems in operating systems. Initially, semaphores were used to manage access to critical sections and prevent race conditions. Over time, they have evolved to become a fundamental tool in concurrent programming, especially in environments where resource management is critical.

### Semaphore in Java

Java's `Semaphore` class provides a robust mechanism for managing concurrent access to resources. It supports two main operations:

- **Acquire**: A thread requests a permit to access the resource. If a permit is available, the semaphore decrements the count and allows the thread to proceed. If no permits are available, the thread is blocked until one becomes available.
- **Release**: A thread returns a permit to the semaphore, incrementing the count and potentially unblocking a waiting thread.

#### Key Features of Java's Semaphore

- **Fairness**: Semaphores can be configured to be fair or non-fair. A fair semaphore ensures that threads acquire permits in the order they requested them, using a first-in-first-out (FIFO) queue. A non-fair semaphore does not guarantee this order, which can lead to higher throughput but may cause starvation.
- **Permits**: The number of permits determines how many threads can access the resource concurrently. This can be set during the semaphore's initialization.

### Practical Applications

Semaphores are particularly useful in scenarios where you need to limit the number of concurrent accesses to a resource. Common use cases include:

- **Limiting Concurrent Database Connections**: In a web application, you might want to limit the number of simultaneous connections to a database to prevent overloading it.
- **Controlling Access to a Pool of Resources**: For example, managing a pool of network connections or threads.
- **Rate Limiting**: Ensuring that a certain operation is performed only a specified number of times per second.

### Semaphore Usage in Java

Let's explore how to use semaphores in Java with practical examples.

#### Basic Semaphore Example

```java
import java.util.concurrent.Semaphore;

public class SemaphoreExample {
    private static final int MAX_PERMITS = 3;
    private final Semaphore semaphore = new Semaphore(MAX_PERMITS);

    public void accessResource() {
        try {
            // Acquire a permit
            semaphore.acquire();
            System.out.println(Thread.currentThread().getName() + " acquired a permit.");

            // Simulate resource access
            Thread.sleep(1000);

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            // Release the permit
            semaphore.release();
            System.out.println(Thread.currentThread().getName() + " released a permit.");
        }
    }

    public static void main(String[] args) {
        SemaphoreExample example = new SemaphoreExample();

        // Create multiple threads to access the resource
        for (int i = 0; i < 10; i++) {
            new Thread(example::accessResource).start();
        }
    }
}
```

**Explanation**: In this example, a semaphore with three permits is created. Ten threads attempt to access the resource, but only three can do so concurrently. The rest must wait until a permit is released.

#### Fair vs. Non-Fair Semaphore

By default, semaphores in Java are non-fair. To create a fair semaphore, you can pass `true` as the second argument to the constructor.

```java
Semaphore fairSemaphore = new Semaphore(MAX_PERMITS, true);
```

**Fair Semaphore Example**: 

```java
import java.util.concurrent.Semaphore;

public class FairSemaphoreExample {
    private static final int MAX_PERMITS = 3;
    private final Semaphore semaphore = new Semaphore(MAX_PERMITS, true);

    public void accessResource() {
        try {
            semaphore.acquire();
            System.out.println(Thread.currentThread().getName() + " acquired a permit.");

            Thread.sleep(1000);

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            semaphore.release();
            System.out.println(Thread.currentThread().getName() + " released a permit.");
        }
    }

    public static void main(String[] args) {
        FairSemaphoreExample example = new FairSemaphoreExample();

        for (int i = 0; i < 10; i++) {
            new Thread(example::accessResource).start();
        }
    }
}
```

**Explanation**: In this fair semaphore example, threads acquire permits in the order they requested them, ensuring fairness.

### Advanced Semaphore Usage

#### Semaphore with Timeout

Sometimes, a thread may need to acquire a permit within a certain timeframe. Java's `Semaphore` class provides a method `tryAcquire` that allows specifying a timeout.

```java
boolean acquired = semaphore.tryAcquire(500, TimeUnit.MILLISECONDS);
if (acquired) {
    try {
        // Access the resource
    } finally {
        semaphore.release();
    }
} else {
    System.out.println("Could not acquire a permit within the timeout.");
}
```

**Explanation**: This code attempts to acquire a permit within 500 milliseconds. If successful, the thread proceeds; otherwise, it handles the timeout scenario.

#### Semaphore for Resource Pools

Semaphores are ideal for managing a pool of resources, such as a connection pool. Here's a simplified example:

```java
import java.util.concurrent.Semaphore;
import java.util.ArrayList;
import java.util.List;

public class ConnectionPool {
    private final List<Connection> pool;
    private final Semaphore semaphore;

    public ConnectionPool(int size) {
        pool = new ArrayList<>(size);
        semaphore = new Semaphore(size);
        for (int i = 0; i < size; i++) {
            pool.add(new Connection("Connection-" + i));
        }
    }

    public Connection acquireConnection() throws InterruptedException {
        semaphore.acquire();
        return getConnectionFromPool();
    }

    public void releaseConnection(Connection connection) {
        returnConnectionToPool(connection);
        semaphore.release();
    }

    private synchronized Connection getConnectionFromPool() {
        return pool.remove(pool.size() - 1);
    }

    private synchronized void returnConnectionToPool(Connection connection) {
        pool.add(connection);
    }

    public static void main(String[] args) {
        ConnectionPool pool = new ConnectionPool(3);

        for (int i = 0; i < 10; i++) {
            new Thread(() -> {
                try {
                    Connection connection = pool.acquireConnection();
                    System.out.println(Thread.currentThread().getName() + " acquired " + connection.getName());
                    Thread.sleep(1000);
                    pool.releaseConnection(connection);
                    System.out.println(Thread.currentThread().getName() + " released " + connection.getName());
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }).start();
        }
    }
}

class Connection {
    private final String name;

    public Connection(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
```

**Explanation**: This example demonstrates a connection pool managed by a semaphore. The pool allows only a limited number of connections to be acquired simultaneously.

### Best Practices and Considerations

- **Fairness vs. Throughput**: Choose fair semaphores when fairness is critical, but be aware that they may reduce throughput due to the overhead of maintaining a queue.
- **Avoid Deadlocks**: Ensure that permits are always released, even in the event of an exception. Use `try-finally` blocks to guarantee this.
- **Resource Management**: Use semaphores to manage finite resources effectively, preventing resource exhaustion and ensuring stability.

### Common Pitfalls

- **Permit Leaks**: Failing to release a permit can lead to deadlocks or resource starvation. Always ensure permits are released.
- **Overuse of Fairness**: While fairness is important, overusing fair semaphores can lead to performance bottlenecks. Evaluate the trade-offs carefully.
- **Incorrect Initialization**: Initializing a semaphore with an incorrect number of permits can lead to unexpected behavior. Ensure the permit count matches the available resources.

### Conclusion

Semaphores are a powerful tool for managing concurrency in Java applications. By controlling access to shared resources, they help prevent race conditions and ensure application stability. Understanding how to use semaphores effectively is essential for any Java developer working with multithreaded applications.

### Related Patterns

- **[10.3.3.1 CountDownLatch]({{< ref "/patterns-java/10/3/3/1" >}} "CountDownLatch")**: Similar to semaphores, but used for waiting for a set of operations to complete.
- **[10.3.3.2 CyclicBarrier]({{< ref "/patterns-java/10/3/3/2" >}} "CyclicBarrier")**: Used to synchronize threads at a common barrier point.

### Known Uses

- **Java's Executors Framework**: Uses semaphores internally to manage thread pools.
- **Database Connection Pools**: Many database connection pool implementations use semaphores to limit concurrent connections.

### Further Reading

- [Java Documentation on Semaphore](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/concurrent/Semaphore.html)
- [Concurrency in Practice](https://www.amazon.com/Java-Concurrency-Practice-Brian-Goetz/dp/0321349601)

## Test Your Knowledge: Java Semaphore Concurrency Quiz

{{< quizdown >}}

### What is the primary purpose of a semaphore in Java?

- [x] To control access to a shared resource by managing permits.
- [ ] To execute threads in a specific order.
- [ ] To increase the speed of thread execution.
- [ ] To manage memory allocation.

> **Explanation:** A semaphore controls access to a shared resource by managing a set of permits, allowing only a certain number of threads to access the resource concurrently.

### How does a fair semaphore differ from a non-fair semaphore?

- [x] A fair semaphore ensures threads acquire permits in the order they requested them.
- [ ] A fair semaphore allows more threads to access the resource.
- [ ] A fair semaphore is faster than a non-fair semaphore.
- [ ] A fair semaphore does not use permits.

> **Explanation:** A fair semaphore uses a FIFO queue to ensure that threads acquire permits in the order they requested them, whereas a non-fair semaphore does not guarantee this order.

### Which method is used to acquire a permit from a semaphore?

- [x] acquire()
- [ ] release()
- [ ] lock()
- [ ] wait()

> **Explanation:** The `acquire()` method is used to request a permit from the semaphore, blocking the thread if no permits are available.

### What happens if a thread tries to acquire a permit when none are available?

- [x] The thread is blocked until a permit becomes available.
- [ ] The thread is terminated.
- [ ] The thread continues without a permit.
- [ ] The thread throws an exception.

> **Explanation:** If no permits are available, the thread is blocked until a permit is released by another thread.

### Why might you use a semaphore with a timeout?

- [x] To prevent a thread from waiting indefinitely for a permit.
- [ ] To increase the number of permits dynamically.
- [x] To handle scenarios where waiting too long is undesirable.
- [ ] To ensure fairness among threads.

> **Explanation:** Using a timeout with a semaphore allows a thread to stop waiting for a permit after a certain period, which is useful in scenarios where indefinite waiting is not acceptable.

### What is a common use case for semaphores?

- [x] Limiting concurrent database connections.
- [ ] Sorting data in a collection.
- [ ] Managing memory allocation.
- [ ] Executing tasks in parallel.

> **Explanation:** Semaphores are commonly used to limit the number of concurrent connections to a database or other shared resources.

### How can you ensure that a permit is always released in Java?

- [x] Use a try-finally block to release the permit.
- [ ] Use a synchronized block.
- [x] Use a try-catch block to handle exceptions.
- [ ] Use a lock object.

> **Explanation:** A try-finally block ensures that the release of a permit occurs even if an exception is thrown, preventing deadlocks.

### What is a potential drawback of using a fair semaphore?

- [x] It may reduce throughput due to the overhead of maintaining a queue.
- [ ] It can cause memory leaks.
- [ ] It allows too many threads to access the resource.
- [ ] It does not guarantee thread safety.

> **Explanation:** Fair semaphores can reduce throughput because they maintain a queue to ensure fairness, which adds overhead.

### What is the effect of a permit leak in a semaphore?

- [x] It can lead to deadlocks or resource starvation.
- [ ] It increases the number of permits.
- [ ] It improves performance.
- [ ] It allows more threads to access the resource.

> **Explanation:** A permit leak occurs when a permit is not released, leading to deadlocks or resource starvation as other threads cannot acquire permits.

### True or False: Semaphores are only useful in single-threaded applications.

- [ ] True
- [x] False

> **Explanation:** Semaphores are primarily used in multithreaded applications to manage concurrent access to shared resources.

{{< /quizdown >}}

By understanding and effectively utilizing semaphores, Java developers can enhance the concurrency control in their applications, ensuring efficient and safe access to shared resources.
