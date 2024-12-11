---
canonical: "https://softwarepatternslexicon.com/patterns-java/10/3/1"

title: "Understanding the `volatile` Keyword and Atomic Variables in Java Concurrency"
description: "Explore the role of the `volatile` keyword in Java for ensuring visibility of variable changes across threads and learn about atomic variables for lock-free synchronization."
linkTitle: "10.3.1 The `volatile` Keyword and Atomic Variables"
tags:
- "Java"
- "Concurrency"
- "Volatile"
- "Atomic Variables"
- "Synchronization"
- "Multithreading"
- "CAS"
- "Java Concurrency"
date: 2024-11-25
type: docs
nav_weight: 103100
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 10.3.1 The `volatile` Keyword and Atomic Variables

In the realm of Java concurrency, understanding how to manage shared data between threads is crucial for building robust and efficient applications. This section delves into two critical components of Java's concurrency model: the `volatile` keyword and atomic variables. These tools help developers ensure that changes to variables are visible across threads and provide mechanisms for lock-free synchronization.

### Understanding the `volatile` Keyword

#### Semantics of `volatile`

The `volatile` keyword in Java is a modifier that can be applied to variables to ensure that changes made to them are visible to all threads. When a variable is declared as `volatile`, it guarantees that any thread reading the variable will see the most recently written value by any other thread. This is achieved by preventing the compiler from caching the variable in registers or other CPU caches, ensuring that every read and write operation is performed directly on the main memory.

#### When to Use `volatile`

The `volatile` keyword is appropriate in scenarios where:

- **Visibility is Required**: You need to ensure that changes to a variable are immediately visible to other threads.
- **Atomicity is Not Required**: The operations on the variable are simple reads and writes, not compound actions like incrementing or decrementing.

However, `volatile` does not provide atomicity. For example, incrementing a `volatile` integer is not atomic because it involves multiple operations: reading the current value, incrementing it, and writing it back.

#### Example: Ensuring Visibility with `volatile`

Consider a simple example where a flag is used to control the execution of a thread:

```java
public class VolatileExample {
    private volatile boolean running = true;

    public void start() {
        new Thread(() -> {
            while (running) {
                // Perform some work
            }
            System.out.println("Thread stopped.");
        }).start();
    }

    public void stop() {
        running = false;
    }

    public static void main(String[] args) throws InterruptedException {
        VolatileExample example = new VolatileExample();
        example.start();
        Thread.sleep(1000);
        example.stop();
    }
}
```

In this example, the `running` variable is declared as `volatile`, ensuring that when the `stop()` method sets it to `false`, the change is immediately visible to the thread running the loop.

### Introducing Atomic Variables

#### The `java.util.concurrent.atomic` Package

Java provides the `java.util.concurrent.atomic` package, which includes classes like `AtomicInteger`, `AtomicBoolean`, and `AtomicReference`. These classes offer atomic operations on single variables without the need for explicit synchronization, making them ideal for scenarios where you need to perform compound actions atomically.

#### Atomic Variables and Their Operations

Atomic variables provide methods that perform atomic operations, such as:

- **get()**: Retrieves the current value.
- **set()**: Sets to a new value.
- **compareAndSet(expectedValue, newValue)**: Atomically sets the value to `newValue` if the current value equals `expectedValue`.

These operations are implemented using low-level atomic hardware instructions, ensuring that they are performed atomically without locking.

#### Example: Using `AtomicInteger`

Here's an example demonstrating the use of `AtomicInteger`:

```java
import java.util.concurrent.atomic.AtomicInteger;

public class AtomicExample {
    private AtomicInteger counter = new AtomicInteger(0);

    public void increment() {
        counter.incrementAndGet();
    }

    public int getCounter() {
        return counter.get();
    }

    public static void main(String[] args) {
        AtomicExample example = new AtomicExample();
        for (int i = 0; i < 1000; i++) {
            new Thread(example::increment).start();
        }
        System.out.println("Final counter value: " + example.getCounter());
    }
}
```

In this example, `increment()` uses `incrementAndGet()`, which atomically increments the counter, ensuring thread safety without explicit synchronization.

### Compare-and-Swap (CAS) Operations

#### Significance of CAS

Compare-and-swap (CAS) is a fundamental operation used in non-blocking algorithms. It involves three operands: a memory location, an expected old value, and a new value. CAS atomically updates the memory location to the new value if the current value matches the expected old value.

CAS is significant because it allows for lock-free synchronization, reducing the overhead and potential contention associated with locks. This makes CAS-based algorithms highly efficient, especially in high-concurrency environments.

#### Example: CAS in Action

Consider a scenario where CAS is used to implement a simple spinlock:

```java
import java.util.concurrent.atomic.AtomicBoolean;

public class SpinLock {
    private final AtomicBoolean lock = new AtomicBoolean(false);

    public void lock() {
        while (!lock.compareAndSet(false, true)) {
            // Spin-wait
        }
    }

    public void unlock() {
        lock.set(false);
    }
}
```

In this example, `compareAndSet(false, true)` attempts to acquire the lock by setting it to `true` if it is currently `false`. This operation is atomic, ensuring that only one thread can acquire the lock at a time.

### Scenarios for Using `volatile` and Atomic Variables

#### Appropriate Use Cases

- **`volatile`**: Use when you need to ensure visibility of changes to a variable across threads, and the operations on the variable are simple reads and writes.
- **Atomic Variables**: Use when you need to perform compound actions atomically on a single variable without the overhead of locks.

#### Best Practices

- **Avoid Overuse**: Use `volatile` and atomic variables judiciously. Overusing them can lead to complex and hard-to-maintain code.
- **Understand Limitations**: Recognize that `volatile` does not provide atomicity and atomic variables are limited to single-variable operations.

### Conclusion

The `volatile` keyword and atomic variables are powerful tools in Java's concurrency toolkit. They provide mechanisms for ensuring visibility and atomicity without the need for explicit synchronization, making them ideal for certain scenarios in concurrent programming. By understanding their semantics and appropriate use cases, developers can build more efficient and robust multithreaded applications.

---

## Test Your Knowledge: Java Concurrency with `volatile` and Atomic Variables

{{< quizdown >}}

### What does the `volatile` keyword ensure in Java?

- [x] Visibility of changes to variables across threads
- [ ] Atomicity of operations on variables
- [ ] Synchronization of method execution
- [ ] Prevention of deadlocks

> **Explanation:** The `volatile` keyword ensures that changes to a variable are visible to all threads, but it does not provide atomicity.

### Which package provides atomic variables in Java?

- [x] `java.util.concurrent.atomic`
- [ ] `java.util.concurrent`
- [ ] `java.lang`
- [ ] `java.util`

> **Explanation:** The `java.util.concurrent.atomic` package provides classes for atomic variables like `AtomicInteger` and `AtomicBoolean`.

### What is the primary benefit of using atomic variables?

- [x] They provide atomic operations without explicit synchronization.
- [ ] They are faster than primitive types.
- [ ] They consume less memory.
- [ ] They automatically handle exceptions.

> **Explanation:** Atomic variables provide atomic operations on single variables without the need for explicit synchronization, making them efficient for certain concurrent scenarios.

### What operation does `compareAndSet` perform?

- [x] Atomically sets a value if the current value matches the expected value.
- [ ] Sets a value without checking the current value.
- [ ] Compares two values and returns a boolean.
- [ ] Swaps two values in memory.

> **Explanation:** `compareAndSet` is an atomic operation that sets a new value if the current value matches the expected value, ensuring atomicity.

### In which scenario is `volatile` most appropriate?

- [x] When visibility of changes is required, but atomicity is not.
- [ ] When atomicity of operations is required.
- [ ] When multiple variables need to be synchronized.
- [ ] When thread execution order needs to be controlled.

> **Explanation:** `volatile` is used when you need to ensure visibility of changes to a variable across threads, but the operations are simple reads and writes.

### What is a common use case for atomic variables?

- [x] Performing atomic increments on counters
- [ ] Synchronizing access to a collection
- [ ] Managing thread execution order
- [ ] Handling exceptions in multithreaded code

> **Explanation:** Atomic variables are commonly used for atomic operations like incrementing counters without explicit synchronization.

### How does CAS contribute to non-blocking algorithms?

- [x] It allows atomic updates without locks.
- [ ] It prevents race conditions by locking variables.
- [ ] It ensures thread execution order.
- [ ] It reduces memory usage.

> **Explanation:** CAS (Compare-and-Swap) allows for atomic updates without the need for locks, making it a key component of non-blocking algorithms.

### What is a limitation of the `volatile` keyword?

- [x] It does not provide atomicity.
- [ ] It consumes more memory.
- [ ] It is slower than synchronized methods.
- [ ] It cannot be used with primitive types.

> **Explanation:** The `volatile` keyword ensures visibility but does not provide atomicity for compound operations.

### Which method is used to atomically increment an `AtomicInteger`?

- [x] `incrementAndGet()`
- [ ] `addAndGet()`
- [ ] `getAndIncrement()`
- [ ] `set()`

> **Explanation:** `incrementAndGet()` is used to atomically increment the value of an `AtomicInteger`.

### True or False: `volatile` can be used to ensure atomicity of compound operations.

- [ ] True
- [x] False

> **Explanation:** False. The `volatile` keyword ensures visibility but does not provide atomicity for compound operations like incrementing.

{{< /quizdown >}}

By mastering the use of `volatile` and atomic variables, Java developers can effectively manage concurrency in their applications, ensuring both performance and correctness.
