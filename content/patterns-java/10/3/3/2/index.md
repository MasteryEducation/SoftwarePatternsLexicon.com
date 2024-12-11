---
canonical: "https://softwarepatternslexicon.com/patterns-java/10/3/3/2"

title: "CyclicBarrier in Java Concurrency: Synchronization and Parallelism"
description: "Explore the CyclicBarrier in Java, a powerful synchronization aid that allows threads to wait for each other at a common barrier point. Learn how to implement and utilize CyclicBarrier for efficient multithreading."
linkTitle: "10.3.3.2 CyclicBarrier"
tags:
- "Java"
- "Concurrency"
- "CyclicBarrier"
- "Synchronization"
- "Multithreading"
- "Parallelism"
- "Java Concurrency"
- "Thread Synchronization"
date: 2024-11-25
type: docs
nav_weight: 103320
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 10.3.3.2 CyclicBarrier

### Introduction

In the realm of Java concurrency, the `CyclicBarrier` is a powerful synchronization aid that enables a set of threads to wait for each other to reach a common barrier point. This is particularly useful in scenarios where multiple threads must perform tasks in phases, and each phase cannot proceed until all threads have completed the current phase. This section delves into the functionality, practical applications, and implementation of `CyclicBarrier`, providing a comprehensive understanding for experienced Java developers and software architects.

### Understanding CyclicBarrier

#### Functionality

The `CyclicBarrier` class, part of the `java.util.concurrent` package, allows a fixed number of threads to wait for each other at a barrier point. Once all threads have reached this point, the barrier is broken, and the threads can proceed. Unlike `CountDownLatch`, which is a one-time use synchronizer, `CyclicBarrier` can be reused after the barrier is reached, making it ideal for iterative tasks.

#### Key Concepts

- **Barrier Action**: An optional `Runnable` task that can be executed once all threads reach the barrier.
- **Reusability**: After the barrier is tripped, it can be reset and reused, allowing for cyclic operations.
- **Parties**: The number of threads that must reach the barrier before it is tripped.

### Practical Application

Consider a scenario where multiple threads are performing complex calculations in phases. Each thread must complete its current phase before any can proceed to the next. `CyclicBarrier` ensures that all threads synchronize at the end of each phase, maintaining data consistency and coordination.

### Example: Using CyclicBarrier

Let's explore a practical example where multiple threads perform tasks and wait at a barrier before moving to the next phase.

```java
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class CyclicBarrierExample {

    private static final int NUMBER_OF_THREADS = 3;
    private static final CyclicBarrier barrier = new CyclicBarrier(NUMBER_OF_THREADS, new BarrierAction());

    public static void main(String[] args) {
        for (int i = 0; i < NUMBER_OF_THREADS; i++) {
            new Thread(new Task(i)).start();
        }
    }

    static class Task implements Runnable {
        private final int id;

        Task(int id) {
            this.id = id;
        }

        @Override
        public void run() {
            try {
                System.out.println("Thread " + id + " is performing task.");
                Thread.sleep(1000); // Simulate task execution
                System.out.println("Thread " + id + " waiting at barrier.");
                barrier.await();
                System.out.println("Thread " + id + " passed the barrier.");
            } catch (InterruptedException | BrokenBarrierException e) {
                e.printStackTrace();
            }
        }
    }

    static class BarrierAction implements Runnable {
        @Override
        public void run() {
            System.out.println("All threads have reached the barrier. Barrier action executed.");
        }
    }
}
```

#### Explanation

- **Task Execution**: Each thread performs a task and waits at the barrier.
- **Barrier Action**: Once all threads reach the barrier, the `BarrierAction` is executed.
- **Reusability**: The barrier can be reused for subsequent phases.

### Reusability of CyclicBarrier

A significant advantage of `CyclicBarrier` is its ability to be reused. After all threads have reached the barrier and the barrier action is executed, the barrier is reset, allowing threads to synchronize again in future iterations. This makes `CyclicBarrier` ideal for iterative algorithms and simulations.

### Differences from CountDownLatch

While both `CyclicBarrier` and `CountDownLatch` are used for synchronization, they serve different purposes:

- **Reusability**: `CyclicBarrier` can be reused after the barrier is tripped, whereas `CountDownLatch` cannot.
- **Barrier Action**: `CyclicBarrier` supports an optional barrier action, executed once all threads reach the barrier.
- **Use Case**: `CyclicBarrier` is suitable for scenarios requiring repeated synchronization, while `CountDownLatch` is ideal for one-time events.

### Best Practices

- **Exception Handling**: Always handle `InterruptedException` and `BrokenBarrierException` to ensure robust thread execution.
- **Thread Count**: Ensure the number of threads matches the number of parties specified in the `CyclicBarrier`.
- **Barrier Action**: Utilize the barrier action for tasks that should execute once all threads reach the barrier.

### Common Pitfalls

- **Deadlock**: Ensure all threads reach the barrier to avoid deadlock situations.
- **Thread Interruption**: Handle thread interruptions gracefully to prevent unexpected behavior.
- **Resource Management**: Manage resources efficiently to avoid bottlenecks when threads are waiting at the barrier.

### Conclusion

The `CyclicBarrier` is a versatile and powerful tool in Java concurrency, enabling efficient synchronization of threads in iterative tasks. By understanding its functionality, practical applications, and differences from other synchronizers, developers can leverage `CyclicBarrier` to enhance the performance and reliability of multithreaded applications.

### Further Reading

For more information on Java concurrency and synchronization, refer to the [Oracle Java Documentation](https://docs.oracle.com/en/java/) and explore related sections in this guide, such as [10.3.3.1 CountDownLatch]({{< ref "/patterns-java/10/3/3/1" >}} "CountDownLatch").

---

## Test Your Knowledge: CyclicBarrier in Java Concurrency Quiz

{{< quizdown >}}

### What is the primary purpose of a CyclicBarrier in Java?

- [x] To allow a set of threads to wait for each other at a common barrier point.
- [ ] To execute tasks asynchronously without synchronization.
- [ ] To manage thread pools efficiently.
- [ ] To provide a one-time synchronization point for threads.

> **Explanation:** The CyclicBarrier allows a group of threads to wait for each other at a common barrier point, making it ideal for tasks that require synchronization in phases.

### How does CyclicBarrier differ from CountDownLatch?

- [x] CyclicBarrier can be reused after the barrier is reached.
- [ ] CyclicBarrier is used for one-time synchronization.
- [ ] CyclicBarrier does not support barrier actions.
- [ ] CyclicBarrier is less efficient than CountDownLatch.

> **Explanation:** Unlike CountDownLatch, CyclicBarrier can be reused after the barrier is reached, making it suitable for iterative tasks.

### What happens when all threads reach the CyclicBarrier?

- [x] The barrier is tripped, and an optional barrier action is executed.
- [ ] The threads are terminated.
- [ ] The threads are put to sleep indefinitely.
- [ ] The barrier is reset without executing any action.

> **Explanation:** When all threads reach the CyclicBarrier, the barrier is tripped, and an optional barrier action can be executed before the threads proceed.

### Can CyclicBarrier be used for tasks that require repeated synchronization?

- [x] Yes
- [ ] No

> **Explanation:** CyclicBarrier is designed for tasks that require repeated synchronization, as it can be reset and reused after each barrier is reached.

### What exception must be handled when using CyclicBarrier?

- [x] InterruptedException
- [x] BrokenBarrierException
- [ ] IOException
- [ ] NullPointerException

> **Explanation:** InterruptedException and BrokenBarrierException must be handled to ensure robust thread execution when using CyclicBarrier.

### What is a barrier action in CyclicBarrier?

- [x] An optional task executed once all threads reach the barrier.
- [ ] A mandatory task executed before threads reach the barrier.
- [ ] A task that interrupts all threads at the barrier.
- [ ] A task that resets the barrier without execution.

> **Explanation:** A barrier action is an optional task that is executed once all threads reach the barrier, providing an opportunity to perform additional operations.

### How can deadlock be avoided when using CyclicBarrier?

- [x] Ensure all threads reach the barrier.
- [x] Handle exceptions properly.
- [ ] Use fewer threads than the number of parties.
- [ ] Avoid using barrier actions.

> **Explanation:** To avoid deadlock, ensure all threads reach the barrier and handle exceptions properly to maintain synchronization.

### What is the role of the 'parties' parameter in CyclicBarrier?

- [x] It specifies the number of threads that must reach the barrier.
- [ ] It determines the priority of threads at the barrier.
- [ ] It limits the number of barrier actions.
- [ ] It sets the maximum execution time for threads.

> **Explanation:** The 'parties' parameter specifies the number of threads that must reach the barrier before it is tripped, ensuring synchronization.

### Can CyclicBarrier be used with a single thread?

- [ ] Yes
- [x] No

> **Explanation:** CyclicBarrier requires multiple threads to be effective, as it is designed for synchronization among a group of threads.

### True or False: CyclicBarrier is part of the java.util.concurrent package.

- [x] True
- [ ] False

> **Explanation:** CyclicBarrier is indeed part of the java.util.concurrent package, providing synchronization utilities for concurrent programming.

{{< /quizdown >}}

---
