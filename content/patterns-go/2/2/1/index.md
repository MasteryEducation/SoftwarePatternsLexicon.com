---
linkTitle: "2.2.1 Adapter"
title: "Adapter Design Pattern in Go: Bridging Interfaces for Seamless Integration"
description: "Explore the Adapter design pattern in Go, which allows incompatible interfaces to work together seamlessly. Learn implementation steps, use cases, and Go-specific tips for effective integration."
categories:
- Design Patterns
- Go Programming
- Software Architecture
tags:
- Adapter Pattern
- Structural Patterns
- GoF Patterns
- Go Interfaces
- Software Design
date: 2024-10-25
type: docs
nav_weight: 221000
canonical: "https://softwarepatternslexicon.com/patterns-go/2/2/1"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.2.1 Adapter

In the realm of software design, the Adapter pattern is a structural pattern from the classic Gang of Four (GoF) design patterns. It plays a crucial role in enabling classes with incompatible interfaces to work together seamlessly. This pattern is akin to a translator that allows two parties speaking different languages to communicate effectively.

### Understand the Intent

The primary intent of the Adapter pattern is to convert the interface of a class into another interface that clients expect. This conversion allows classes with incompatible interfaces to collaborate without modifying their existing code. By using an adapter, you can integrate third-party libraries or legacy systems into your application without altering their original interfaces.

### Implementation Steps

Implementing the Adapter pattern in Go involves several key steps:

1. **Define the Target Interface:**
   - Identify and define the interface that the client expects. This interface represents the methods that the client will use.

2. **Create the Adapter:**
   - Develop an adapter that implements the target interface. The adapter acts as a bridge between the client and the adaptee (the class with the incompatible interface).

3. **Reference the Adaptee:**
   - Within the adapter, hold a reference to the adaptee. This reference allows the adapter to delegate method calls to the adaptee's methods.

4. **Implement Interface Methods:**
   - Implement the methods of the target interface in the adapter. These methods should translate client calls into calls that the adaptee can understand.

### When to Use

The Adapter pattern is particularly useful in the following scenarios:

- **Interface Mismatch:**
  - When you want to use an existing class, but its interface does not match the requirements of your application.

- **Reusability:**
  - To create reusable classes that can cooperate with unrelated or unforeseen classes, enhancing the flexibility and scalability of your codebase.

### Go-Specific Tips

Go's unique features and idioms can be leveraged to implement the Adapter pattern efficiently:

- **Use Interfaces:**
  - Define the expected methods using Go interfaces. This approach allows for flexible and decoupled designs, making it easier to swap out implementations.

- **Type Embedding:**
  - Leverage Go's type embedding to reduce boilerplate code. By embedding the adaptee type within the adapter, you can directly access its methods, simplifying the implementation.

### Example: Adapting a Third-Party Logging Library

Consider a scenario where you have a third-party logging library with an interface that differs from your application's logging interface. You can use the Adapter pattern to bridge this gap.

#### Step 1: Define the Target Interface

```go
// Logger is the target interface expected by the application.
type Logger interface {
    LogInfo(message string)
    LogError(err error)
}
```

#### Step 2: Create the Adaptee

```go
// ThirdPartyLogger is the existing logging library with a different interface.
type ThirdPartyLogger struct{}

func (l *ThirdPartyLogger) PrintInfo(msg string) {
    fmt.Println("INFO:", msg)
}

func (l *ThirdPartyLogger) PrintError(msg string) {
    fmt.Println("ERROR:", msg)
}
```

#### Step 3: Implement the Adapter

```go
// LoggerAdapter adapts ThirdPartyLogger to the Logger interface.
type LoggerAdapter struct {
    adaptee *ThirdPartyLogger
}

func (a *LoggerAdapter) LogInfo(message string) {
    a.adaptee.PrintInfo(message)
}

func (a *LoggerAdapter) LogError(err error) {
    a.adaptee.PrintError(err.Error())
}
```

#### Step 4: Use the Adapter

```go
func main() {
    thirdPartyLogger := &ThirdPartyLogger{}
    logger := &LoggerAdapter{adaptee: thirdPartyLogger}

    logger.LogInfo("This is an info message.")
    logger.LogError(fmt.Errorf("This is an error message."))
}
```

### Advantages and Disadvantages

#### Advantages

- **Flexibility:**
  - Allows integration of classes with incompatible interfaces without modifying their code.
- **Reusability:**
  - Promotes the reuse of existing code by adapting it to new interfaces.
- **Decoupling:**
  - Decouples the client from the implementation details of the adaptee.

#### Disadvantages

- **Complexity:**
  - Introduces additional layers of abstraction, which can increase complexity.
- **Performance:**
  - May introduce slight overhead due to the additional method calls.

### Best Practices

- **Keep It Simple:**
  - Avoid over-complicating the adapter. It should only translate calls between the client and the adaptee.
- **Use Interfaces Wisely:**
  - Define clear and concise interfaces to ensure that the adapter remains easy to understand and maintain.
- **Document Thoroughly:**
  - Provide documentation to explain the purpose and usage of the adapter, especially if it involves complex translations.

### Comparisons

The Adapter pattern is often compared with the **Facade** pattern. While both patterns provide a simplified interface, the Adapter pattern is specifically used to make two incompatible interfaces work together, whereas the Facade pattern provides a unified interface to a set of interfaces in a subsystem.

### Conclusion

The Adapter pattern is a powerful tool in the software design arsenal, enabling seamless integration of disparate systems and enhancing code reusability. By understanding and applying this pattern, developers can create flexible and maintainable software architectures that accommodate evolving requirements.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Adapter pattern?

- [x] To convert the interface of a class into another interface clients expect.
- [ ] To provide a simplified interface to a set of interfaces in a subsystem.
- [ ] To define a family of algorithms and make them interchangeable.
- [ ] To ensure a class has only one instance and provide a global point of access to it.

> **Explanation:** The Adapter pattern's main goal is to convert the interface of a class into another interface that clients expect, allowing incompatible interfaces to work together.

### When should you use the Adapter pattern?

- [x] When you want to use an existing class, but its interface doesn't match your requirements.
- [ ] When you need to provide a simplified interface to a complex subsystem.
- [ ] When you need to ensure a class has only one instance.
- [ ] When you want to encapsulate a request as an object.

> **Explanation:** The Adapter pattern is used when you want to integrate an existing class with an incompatible interface into your application.

### What Go feature can be leveraged to reduce boilerplate code in the Adapter pattern?

- [x] Type embedding
- [ ] Goroutines
- [ ] Channels
- [ ] Reflection

> **Explanation:** Type embedding in Go can be used to reduce boilerplate code by allowing direct access to the methods of the embedded type.

### Which of the following is a disadvantage of the Adapter pattern?

- [x] It introduces additional layers of abstraction, increasing complexity.
- [ ] It provides a simplified interface to a set of interfaces.
- [ ] It ensures a class has only one instance.
- [ ] It encapsulates a request as an object.

> **Explanation:** The Adapter pattern can increase complexity by introducing additional layers of abstraction.

### In the provided example, what is the role of `LoggerAdapter`?

- [x] It adapts `ThirdPartyLogger` to the `Logger` interface.
- [ ] It provides a simplified interface to `ThirdPartyLogger`.
- [ ] It encapsulates logging requests as objects.
- [ ] It ensures `ThirdPartyLogger` has only one instance.

> **Explanation:** `LoggerAdapter` adapts the `ThirdPartyLogger` to the `Logger` interface, allowing it to be used where the `Logger` interface is expected.

### What is the difference between the Adapter and Facade patterns?

- [x] Adapter makes two incompatible interfaces work together, while Facade provides a unified interface to a subsystem.
- [ ] Adapter provides a unified interface to a subsystem, while Facade makes two incompatible interfaces work together.
- [ ] Adapter and Facade are the same pattern with different names.
- [ ] Adapter is used for concurrency, while Facade is used for simplifying interfaces.

> **Explanation:** The Adapter pattern is used to make two incompatible interfaces work together, whereas the Facade pattern provides a unified interface to a set of interfaces in a subsystem.

### Which of the following best describes the Adapter pattern?

- [x] A structural pattern that allows incompatible interfaces to work together.
- [ ] A creational pattern that ensures a class has only one instance.
- [ ] A behavioral pattern that encapsulates a request as an object.
- [ ] A concurrency pattern that manages goroutines.

> **Explanation:** The Adapter pattern is a structural pattern that enables incompatible interfaces to work together.

### How does the Adapter pattern promote reusability?

- [x] By allowing existing classes to be reused with new interfaces.
- [ ] By ensuring a class has only one instance.
- [ ] By encapsulating requests as objects.
- [ ] By providing a unified interface to a subsystem.

> **Explanation:** The Adapter pattern promotes reusability by allowing existing classes to be used with new interfaces without modifying their code.

### True or False: The Adapter pattern can be used to integrate third-party libraries into an application.

- [x] True
- [ ] False

> **Explanation:** True. The Adapter pattern is often used to integrate third-party libraries with incompatible interfaces into an application.

### What is a key benefit of using interfaces in Go when implementing the Adapter pattern?

- [x] They provide a flexible and decoupled design.
- [ ] They ensure a class has only one instance.
- [ ] They encapsulate requests as objects.
- [ ] They manage goroutines efficiently.

> **Explanation:** Using interfaces in Go allows for a flexible and decoupled design, making it easier to swap out implementations.

{{< /quizdown >}}
