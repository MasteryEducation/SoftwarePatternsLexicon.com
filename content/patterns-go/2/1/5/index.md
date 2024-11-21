---
linkTitle: "2.1.5 Singleton"
title: "Singleton Pattern in Go: Ensuring Single Instance with Thread Safety"
description: "Explore the Singleton design pattern in Go, its implementation, use cases, and best practices for ensuring a single instance with thread safety."
categories:
- Design Patterns
- Go Programming
- Software Architecture
tags:
- Singleton
- Go
- Creational Patterns
- Concurrency
- Thread Safety
date: 2024-10-25
type: docs
nav_weight: 215000
canonical: "https://softwarepatternslexicon.com/patterns-go/2/1/5"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.1.5 Singleton

The Singleton pattern is a creational design pattern that ensures a class has only one instance and provides a global point of access to it. This pattern is particularly useful in scenarios where a single instance of a class is required to coordinate actions across the system, such as managing configurations, logging, or connection pooling.

### Purpose of the Singleton Pattern

- **Ensure a Single Instance:** The primary goal of the Singleton pattern is to restrict the instantiation of a class to a single object. This is crucial when exactly one object is needed to coordinate actions across the system.
- **Global Access Point:** It provides a global access point to the instance, allowing it to be accessed from anywhere in the application.
- **Control Access to Shared Resources:** By ensuring a single instance, the Singleton pattern helps control access to shared resources, preventing conflicts and ensuring consistency.

### Implementation Steps

Implementing the Singleton pattern in Go involves several key steps to ensure thread safety and proper initialization:

1. **Declare a Package-Level Variable:** Create a package-level variable to hold the singleton instance. This variable will be accessed by the function that returns the singleton.

2. **Create a Function to Return the Singleton:** Implement a function that returns the singleton instance. This function should initialize the instance if it hasn't been created yet.

3. **Ensure Thread-Safe Initialization:** Use `sync.Once` to ensure that the singleton instance is initialized only once, even in concurrent environments. This prevents race conditions and ensures thread safety.

### Go-Specific Tips

- **Utilize Package-Level Variables:** In Go, package-level variables are a simple way to manage the singleton instance. They provide a natural scope for the singleton and are easily accessible within the package.

- **Cautious Use of `init()` Functions:** While `init()` functions can be used for initialization, it's often better to use explicit initialization to avoid hidden dependencies and improve code clarity.

- **Concurrency Considerations:** Go's concurrency model requires careful handling of shared resources. Use synchronization primitives like `sync.Once` to ensure thread-safe initialization of the singleton instance.

### Example: Singleton Logger

Let's implement a singleton logger in Go that writes to a single log file. This example demonstrates how to safely access the singleton from multiple goroutines.

```go
package logger

import (
	"log"
	"os"
	"sync"
)

type Logger struct {
	file *os.File
	*log.Logger
}

var (
	instance *Logger
	once     sync.Once
)

// GetInstance returns the singleton instance of Logger.
func GetInstance() *Logger {
	once.Do(func() {
		file, err := os.OpenFile("app.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
		if err != nil {
			log.Fatalf("Failed to open log file: %v", err)
		}
		instance = &Logger{
			file:   file,
			Logger: log.New(file, "APP_LOG: ", log.Ldate|log.Ltime|log.Lshortfile),
		}
	})
	return instance
}

// Close closes the log file.
func (l *Logger) Close() {
	l.file.Close()
}
```

### Accessing the Singleton from Multiple Goroutines

Here's how you can use the singleton logger from multiple goroutines safely:

```go
package main

import (
	"sync"
	"your_project/logger"
)

func main() {
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		log := logger.GetInstance()
		log.Println("Log message from goroutine 1")
	}()

	go func() {
		defer wg.Done()
		log := logger.GetInstance()
		log.Println("Log message from goroutine 2")
	}()

	wg.Wait()
}
```

### When to Use the Singleton Pattern

- **Single Instance Requirement:** Use the Singleton pattern when you need only one instance of a class across the application, such as for configuration management or logging.
- **Shared Resource Management:** It's ideal for managing shared resources like database connections or thread pools, where multiple instances could lead to conflicts or resource exhaustion.

### Advantages and Disadvantages

**Advantages:**

- **Controlled Access:** Provides controlled access to a single instance, ensuring consistency and preventing conflicts.
- **Global Access:** Offers a global point of access, simplifying the use of the instance across different parts of the application.

**Disadvantages:**

- **Global State:** Singleton can introduce global state into an application, which can make testing and debugging more challenging.
- **Hidden Dependencies:** It can lead to hidden dependencies, making the code harder to understand and maintain.

### Best Practices

- **Lazy Initialization:** Use lazy initialization to create the singleton instance only when it's needed, optimizing resource usage.
- **Thread Safety:** Ensure thread safety using synchronization primitives like `sync.Once` to prevent race conditions.
- **Avoid Overuse:** Use the Singleton pattern judiciously, as it can introduce global state and hidden dependencies.

### Conclusion

The Singleton pattern is a powerful tool for ensuring a single instance of a class and providing a global access point. In Go, careful attention to concurrency and initialization ensures that the pattern is implemented safely and effectively. By following best practices and understanding the pattern's advantages and disadvantages, developers can leverage the Singleton pattern to manage shared resources and configurations efficiently.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of the Singleton pattern?

- [x] To ensure a class has only one instance and provide a global access point.
- [ ] To allow multiple instances of a class to be created.
- [ ] To encapsulate a request as an object.
- [ ] To define a family of algorithms.

> **Explanation:** The Singleton pattern ensures a class has only one instance and provides a global access point to it.

### Which Go synchronization primitive is commonly used to ensure thread-safe initialization of a Singleton?

- [x] sync.Once
- [ ] sync.Mutex
- [ ] sync.WaitGroup
- [ ] sync.Cond

> **Explanation:** `sync.Once` is used to ensure that the initialization of the singleton instance occurs only once, even in concurrent environments.

### What is a potential disadvantage of using the Singleton pattern?

- [x] It can introduce global state into an application.
- [ ] It allows multiple instances of a class to be created.
- [ ] It makes code more modular.
- [ ] It simplifies testing.

> **Explanation:** The Singleton pattern can introduce global state, which can complicate testing and debugging.

### In Go, where is the singleton instance typically stored?

- [x] In a package-level variable
- [ ] In a local variable within a function
- [ ] In a global variable outside any package
- [ ] In a constant

> **Explanation:** The singleton instance is typically stored in a package-level variable for easy access within the package.

### When should you use the Singleton pattern?

- [x] When only one instance of a class is needed across the application.
- [ ] When multiple instances of a class are needed.
- [ ] When encapsulating a request as an object.
- [ ] When defining a family of algorithms.

> **Explanation:** The Singleton pattern is used when only one instance of a class is needed across the application.

### What is the role of the `once.Do` function in the Singleton pattern?

- [x] To ensure the singleton instance is initialized only once.
- [ ] To lock the singleton instance for exclusive access.
- [ ] To wait for all goroutines to finish.
- [ ] To signal a condition variable.

> **Explanation:** `once.Do` ensures that the singleton instance is initialized only once, providing thread safety.

### Why is lazy initialization recommended for the Singleton pattern?

- [x] To create the singleton instance only when it's needed, optimizing resource usage.
- [ ] To create the singleton instance at program startup.
- [ ] To allow multiple instances to be created.
- [ ] To simplify the code structure.

> **Explanation:** Lazy initialization creates the singleton instance only when it's needed, optimizing resource usage.

### How can the Singleton pattern affect code testing?

- [x] It can make testing more challenging due to global state.
- [ ] It simplifies testing by providing multiple instances.
- [ ] It has no effect on testing.
- [ ] It makes testing easier by encapsulating requests.

> **Explanation:** The Singleton pattern can make testing more challenging due to the introduction of global state.

### What is a common use case for the Singleton pattern?

- [x] Managing shared resources like configurations or connection pools.
- [ ] Encapsulating a request as an object.
- [ ] Defining a family of algorithms.
- [ ] Allowing multiple instances of a class.

> **Explanation:** The Singleton pattern is commonly used for managing shared resources like configurations or connection pools.

### True or False: The Singleton pattern provides a global access point to the instance.

- [x] True
- [ ] False

> **Explanation:** True. The Singleton pattern provides a global access point to the single instance of the class.

{{< /quizdown >}}
