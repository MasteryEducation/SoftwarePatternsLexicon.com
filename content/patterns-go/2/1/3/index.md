---
linkTitle: "2.1.3 Factory Method"
title: "Factory Method Design Pattern in Go: A Comprehensive Guide"
description: "Explore the Factory Method design pattern in Go, its implementation, use cases, and best practices for creating flexible and scalable software architectures."
categories:
- Design Patterns
- Go Programming
- Software Architecture
tags:
- Factory Method
- Creational Patterns
- Go Language
- Software Design
- Object-Oriented Programming
date: 2024-10-25
type: docs
nav_weight: 213000
canonical: "https://softwarepatternslexicon.com/patterns-go/2/1/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.1.3 Factory Method

The Factory Method is a creational design pattern that provides an interface for creating objects in a superclass but allows subclasses to alter the type of objects that will be created. This pattern is particularly useful in scenarios where a class cannot anticipate the class of objects it needs to create or when it wants to delegate the responsibility of object instantiation to subclasses.

### Understand the Intent

The primary intent of the Factory Method pattern is to define an interface for creating an object, but let subclasses decide which class to instantiate. This approach promotes loose coupling by reducing the dependency of the code on specific classes and allows for more flexible and scalable software design.

### Implementation Steps

To implement the Factory Method pattern in Go, follow these steps:

1. **Define a Creator Interface or Struct with a Factory Method:**
   - The creator interface or struct declares the factory method that returns an object of a product interface.

2. **Implement Concrete Creators:**
   - Concrete creators override the factory method to produce different products. Each concrete creator corresponds to a specific product type.

3. **Define Product Interfaces and Concrete Product Types:**
   - The product interface defines the operations that all concrete products must implement. Concrete products are the different implementations of the product interface.

### Use Cases

The Factory Method pattern is applicable in the following scenarios:

- When a class cannot anticipate the class of objects it needs to create.
- When a class wants its subclasses to specify the objects it creates.
- To delegate the responsibility of object instantiation to subclasses, promoting flexibility and scalability.

### Example in Go

Let's illustrate the Factory Method pattern with an example of a logger that creates different log output objects.

```go
package main

import (
    "fmt"
)

// Logger is the product interface
type Logger interface {
    Log(message string)
}

// ConsoleLogger is a concrete product
type ConsoleLogger struct{}

func (c *ConsoleLogger) Log(message string) {
    fmt.Println("Console log:", message)
}

// FileLogger is another concrete product
type FileLogger struct{}

func (f *FileLogger) Log(message string) {
    fmt.Println("File log:", message)
}

// LoggerFactory is the creator interface
type LoggerFactory interface {
    CreateLogger() Logger
}

// ConsoleLoggerFactory is a concrete creator
type ConsoleLoggerFactory struct{}

func (c *ConsoleLoggerFactory) CreateLogger() Logger {
    return &ConsoleLogger{}
}

// FileLoggerFactory is another concrete creator
type FileLoggerFactory struct{}

func (f *FileLoggerFactory) CreateLogger() Logger {
    return &FileLogger{}
}

func main() {
    var factory LoggerFactory

    // Use ConsoleLoggerFactory to create a ConsoleLogger
    factory = &ConsoleLoggerFactory{}
    logger := factory.CreateLogger()
    logger.Log("Hello, Console!")

    // Use FileLoggerFactory to create a FileLogger
    factory = &FileLoggerFactory{}
    logger = factory.CreateLogger()
    logger.Log("Hello, File!")
}
```

In this example, we define a `Logger` interface with a `Log` method. The `ConsoleLogger` and `FileLogger` structs implement this interface. The `LoggerFactory` interface declares a `CreateLogger` method, which is implemented by `ConsoleLoggerFactory` and `FileLoggerFactory` to create instances of `ConsoleLogger` and `FileLogger`, respectively.

### Best Practices

- **Keep the Creator and Product Interfaces Minimal and Focused:** Ensure that interfaces are concise and only include methods that are essential for the pattern's operation.
- **Centralize Instantiation Logic:** Use the factory method to handle the instantiation logic centrally, promoting consistency and reducing duplication.
- **Encapsulate Object Creation:** By encapsulating object creation, you can easily extend the system with new product types without modifying existing code.

### Advantages and Disadvantages

**Advantages:**

- **Flexibility:** Allows for easy extension of the system with new product types.
- **Decoupling:** Reduces dependency on specific classes, promoting loose coupling.
- **Single Responsibility Principle:** Separates the responsibility of object creation from the main logic.

**Disadvantages:**

- **Complexity:** Can introduce additional complexity due to the increased number of classes and interfaces.
- **Overhead:** May lead to unnecessary overhead if not used judiciously, especially in simple scenarios.

### Comparisons

The Factory Method pattern is often compared with the Abstract Factory pattern. While both patterns deal with object creation, the Factory Method focuses on a single product, whereas the Abstract Factory is concerned with creating families of related products.

### Conclusion

The Factory Method pattern is a powerful tool for creating flexible and scalable software architectures. By delegating object creation to subclasses, it promotes loose coupling and adherence to the Single Responsibility Principle. When implemented correctly, it can significantly enhance the maintainability and extensibility of your Go applications.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Factory Method pattern?

- [x] To define an interface for creating an object but let subclasses decide which class to instantiate.
- [ ] To create a single instance of a class.
- [ ] To provide a simplified interface to a complex subsystem.
- [ ] To compose objects into tree structures.

> **Explanation:** The Factory Method pattern's primary intent is to define an interface for creating an object but allow subclasses to alter the type of objects that will be created.

### Which of the following is a use case for the Factory Method pattern?

- [x] When a class cannot anticipate the class of objects it needs to create.
- [ ] When you need to create a single instance of a class.
- [ ] When you need to provide a simplified interface to a complex subsystem.
- [ ] When you need to compose objects into tree structures.

> **Explanation:** The Factory Method pattern is used when a class cannot anticipate the class of objects it needs to create or wants to delegate the responsibility of object instantiation to subclasses.

### In the provided Go example, what does the `CreateLogger` method do?

- [x] It creates and returns an instance of a Logger.
- [ ] It logs a message to the console.
- [ ] It writes a log message to a file.
- [ ] It initializes the logging system.

> **Explanation:** The `CreateLogger` method in the example is responsible for creating and returning an instance of a Logger.

### What is a disadvantage of the Factory Method pattern?

- [x] It can introduce additional complexity due to the increased number of classes and interfaces.
- [ ] It makes it difficult to extend the system with new product types.
- [ ] It tightly couples the code to specific classes.
- [ ] It violates the Single Responsibility Principle.

> **Explanation:** A disadvantage of the Factory Method pattern is that it can introduce additional complexity due to the increased number of classes and interfaces.

### How does the Factory Method pattern promote flexibility?

- [x] By allowing easy extension of the system with new product types.
- [ ] By tightly coupling the code to specific classes.
- [ ] By reducing the number of classes and interfaces.
- [ ] By providing a single instance of a class.

> **Explanation:** The Factory Method pattern promotes flexibility by allowing easy extension of the system with new product types without modifying existing code.

### Which principle does the Factory Method pattern adhere to?

- [x] Single Responsibility Principle
- [ ] Open/Closed Principle
- [ ] Liskov Substitution Principle
- [ ] Interface Segregation Principle

> **Explanation:** The Factory Method pattern adheres to the Single Responsibility Principle by separating the responsibility of object creation from the main logic.

### What is the role of the `LoggerFactory` interface in the Go example?

- [x] It declares the factory method for creating Logger objects.
- [ ] It implements the logging functionality.
- [ ] It provides a simplified interface to the logging subsystem.
- [ ] It composes Logger objects into tree structures.

> **Explanation:** The `LoggerFactory` interface declares the factory method for creating Logger objects.

### How does the Factory Method pattern reduce dependency on specific classes?

- [x] By defining an interface for creating objects and letting subclasses decide which class to instantiate.
- [ ] By creating a single instance of a class.
- [ ] By providing a simplified interface to a complex subsystem.
- [ ] By composing objects into tree structures.

> **Explanation:** The Factory Method pattern reduces dependency on specific classes by defining an interface for creating objects and letting subclasses decide which class to instantiate.

### What is the difference between the Factory Method and Abstract Factory patterns?

- [x] Factory Method focuses on a single product, while Abstract Factory is concerned with creating families of related products.
- [ ] Factory Method creates a single instance of a class, while Abstract Factory provides a simplified interface to a complex subsystem.
- [ ] Factory Method composes objects into tree structures, while Abstract Factory creates a single instance of a class.
- [ ] Factory Method provides a simplified interface to a complex subsystem, while Abstract Factory focuses on a single product.

> **Explanation:** The Factory Method pattern focuses on a single product, whereas the Abstract Factory pattern is concerned with creating families of related products.

### True or False: The Factory Method pattern can lead to unnecessary overhead if not used judiciously.

- [x] True
- [ ] False

> **Explanation:** True. The Factory Method pattern can lead to unnecessary overhead if not used judiciously, especially in simple scenarios.

{{< /quizdown >}}
