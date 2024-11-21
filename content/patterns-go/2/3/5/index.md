---
linkTitle: "2.3.5 Mediator"
title: "Mediator Pattern in Go: Simplifying Complex Interactions"
description: "Explore the Mediator design pattern in Go, which encapsulates object interactions to promote loose coupling and simplify communication."
categories:
- Design Patterns
- Go Programming
- Software Architecture
tags:
- Mediator Pattern
- GoF Patterns
- Go Language
- Software Design
- Behavioral Patterns
date: 2024-10-25
type: docs
nav_weight: 235000
canonical: "https://softwarepatternslexicon.com/patterns-go/2/3/5"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.3.5 Mediator

In software design, managing complex interactions between objects can become cumbersome and lead to tightly coupled systems. The Mediator pattern offers a solution by encapsulating how a set of objects interact, promoting loose coupling and simplifying communication. This article delves into the Mediator pattern, its implementation in Go, and its practical applications.

### Purpose of the Mediator Pattern

The Mediator pattern is designed to:

- **Encapsulate Interactions:** Define an object that encapsulates how a set of objects interact.
- **Promote Loose Coupling:** Prevent objects from referring to each other explicitly, reducing dependencies and enhancing modularity.

By centralizing communication, the Mediator pattern simplifies the maintenance and evolution of complex systems.

### Implementation Steps

Implementing the Mediator pattern in Go involves several key steps:

#### 1. Define Mediator Interface

The Mediator interface includes methods that facilitate communication between colleague objects. This interface acts as a contract for concrete mediators to implement.

```go
// Mediator interface defines the communication methods between colleagues.
type Mediator interface {
    Notify(sender Colleague, event string)
}
```

#### 2. Implement Concrete Mediator

The Concrete Mediator struct manages and coordinates the interactions between colleague objects. It implements the Mediator interface.

```go
// ConcreteMediator implements the Mediator interface and coordinates communication.
type ConcreteMediator struct {
    airplanes map[string]Colleague
}

func (m *ConcreteMediator) Notify(sender Colleague, event string) {
    // Handle communication logic based on the event and sender.
    for _, airplane := range m.airplanes {
        if airplane != sender {
            airplane.Receive(event)
        }
    }
}
```

#### 3. Define Colleague Interfaces

Colleague interfaces represent the objects that interact through the mediator. These interfaces define methods for sending and receiving messages via the mediator.

```go
// Colleague interface represents an object that interacts through the mediator.
type Colleague interface {
    Send(event string)
    Receive(event string)
}
```

#### 4. Implement Concrete Colleagues

Concrete Colleague structs implement the Colleague interface and use the mediator for communication.

```go
// Airplane represents a concrete colleague that interacts through the mediator.
type Airplane struct {
    name     string
    mediator Mediator
}

func (a *Airplane) Send(event string) {
    fmt.Printf("%s sends event: %s\n", a.name, event)
    a.mediator.Notify(a, event)
}

func (a *Airplane) Receive(event string) {
    fmt.Printf("%s receives event: %s\n", a.name, event)
}
```

### When to Use

The Mediator pattern is particularly useful in scenarios where:

- **Complex Interactions:** There are complex many-to-many interactions among related objects.
- **Simplified Communication:** You need to simplify communication and reduce dependencies between objects.

### Go-Specific Tips

- **Use Channels:** In concurrent contexts, leverage Go's channels for communication to enhance performance and safety.
- **Avoid Complexity:** Be cautious to prevent the mediator from becoming overly complex, as it can become a bottleneck if it handles too much logic.

### Example: Air Traffic Control System

Let's explore a practical example of an air traffic control system using the Mediator pattern. In this system, airplanes (colleagues) communicate with each other through a control tower (mediator) to coordinate takeoffs and landings.

```go
package main

import "fmt"

// Mediator interface defines the communication methods between colleagues.
type Mediator interface {
    Notify(sender Colleague, event string)
}

// ConcreteMediator implements the Mediator interface and coordinates communication.
type ConcreteMediator struct {
    airplanes map[string]Colleague
}

func (m *ConcreteMediator) Notify(sender Colleague, event string) {
    for _, airplane := range m.airplanes {
        if airplane != sender {
            airplane.Receive(event)
        }
    }
}

// Colleague interface represents an object that interacts through the mediator.
type Colleague interface {
    Send(event string)
    Receive(event string)
}

// Airplane represents a concrete colleague that interacts through the mediator.
type Airplane struct {
    name     string
    mediator Mediator
}

func (a *Airplane) Send(event string) {
    fmt.Printf("%s sends event: %s\n", a.name, event)
    a.mediator.Notify(a, event)
}

func (a *Airplane) Receive(event string) {
    fmt.Printf("%s receives event: %s\n", a.name, event)
}

func main() {
    mediator := &ConcreteMediator{airplanes: make(map[string]Colleague)}

    airplane1 := &Airplane{name: "Airplane 1", mediator: mediator}
    airplane2 := &Airplane{name: "Airplane 2", mediator: mediator}

    mediator.airplanes["Airplane 1"] = airplane1
    mediator.airplanes["Airplane 2"] = airplane2

    airplane1.Send("Requesting takeoff")
    airplane2.Send("Requesting landing")
}
```

### Advantages and Disadvantages

**Advantages:**

- **Reduced Complexity:** Simplifies complex interactions by centralizing communication.
- **Loose Coupling:** Promotes loose coupling between objects, enhancing modularity and maintainability.

**Disadvantages:**

- **Mediator Complexity:** The mediator can become complex if it handles too much logic, potentially becoming a bottleneck.
- **Single Point of Failure:** The mediator becomes a single point of failure, which can impact the system if not managed properly.

### Best Practices

- **Keep Mediator Simple:** Ensure the mediator remains simple and focused on coordination rather than business logic.
- **Use Interfaces:** Leverage interfaces to define clear contracts for mediators and colleagues, enhancing flexibility and testability.

### Comparisons

The Mediator pattern is often compared with the Observer pattern. While both patterns facilitate communication between objects, the Mediator pattern centralizes communication through a mediator, whereas the Observer pattern relies on direct notifications between subjects and observers.

### Conclusion

The Mediator pattern is a powerful tool for managing complex interactions between objects in Go applications. By encapsulating communication, it promotes loose coupling and simplifies system architecture. However, care must be taken to prevent the mediator from becoming overly complex. By following best practices and leveraging Go's concurrency features, developers can effectively implement the Mediator pattern to enhance the maintainability and scalability of their applications.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of the Mediator pattern?

- [x] To encapsulate how a set of objects interact and promote loose coupling.
- [ ] To create a single instance of a class.
- [ ] To provide a simplified interface to a complex subsystem.
- [ ] To define a family of algorithms.

> **Explanation:** The Mediator pattern encapsulates interactions between objects and promotes loose coupling by centralizing communication.

### Which of the following is a key component of the Mediator pattern?

- [x] Mediator interface
- [ ] Singleton instance
- [ ] Observer subject
- [ ] Factory method

> **Explanation:** The Mediator interface defines the methods for communication between colleague objects.

### In the Mediator pattern, what role do colleague objects play?

- [x] They interact with each other through the mediator.
- [ ] They directly communicate with each other.
- [ ] They provide a simplified interface to a subsystem.
- [ ] They encapsulate a request as an object.

> **Explanation:** Colleague objects interact with each other through the mediator, not directly.

### When is the Mediator pattern particularly useful?

- [x] When there are complex many-to-many interactions among related objects.
- [ ] When a single instance of a class is needed.
- [ ] When a simplified interface to a complex subsystem is required.
- [ ] When a family of algorithms needs to be defined.

> **Explanation:** The Mediator pattern is useful for managing complex interactions and reducing dependencies between objects.

### What is a potential disadvantage of the Mediator pattern?

- [x] The mediator can become overly complex.
- [ ] It increases coupling between objects.
- [ ] It requires a single instance of a class.
- [ ] It simplifies communication between objects.

> **Explanation:** The mediator can become overly complex if it handles too much logic, which is a potential disadvantage.

### How can Go's channels be used in the Mediator pattern?

- [x] For communication in concurrent contexts.
- [ ] To create a single instance of a class.
- [ ] To define a family of algorithms.
- [ ] To provide a simplified interface to a subsystem.

> **Explanation:** Go's channels can be used for communication in concurrent contexts, enhancing performance and safety.

### What is a best practice when implementing the Mediator pattern?

- [x] Keep the mediator simple and focused on coordination.
- [ ] Use the mediator to handle all business logic.
- [ ] Avoid using interfaces for mediators and colleagues.
- [ ] Ensure the mediator becomes a single point of failure.

> **Explanation:** Keeping the mediator simple and focused on coordination is a best practice to avoid complexity.

### How does the Mediator pattern differ from the Observer pattern?

- [x] The Mediator pattern centralizes communication through a mediator.
- [ ] The Observer pattern centralizes communication through a mediator.
- [ ] The Mediator pattern relies on direct notifications between subjects and observers.
- [ ] The Observer pattern encapsulates how a set of objects interact.

> **Explanation:** The Mediator pattern centralizes communication through a mediator, while the Observer pattern relies on direct notifications.

### What is a key benefit of using the Mediator pattern?

- [x] It reduces complexity by centralizing communication.
- [ ] It increases complexity by adding more objects.
- [ ] It requires direct communication between objects.
- [ ] It simplifies communication by eliminating the need for a mediator.

> **Explanation:** The Mediator pattern reduces complexity by centralizing communication, making interactions easier to manage.

### True or False: The Mediator pattern promotes tight coupling between objects.

- [ ] True
- [x] False

> **Explanation:** False. The Mediator pattern promotes loose coupling by centralizing communication and reducing dependencies between objects.

{{< /quizdown >}}
