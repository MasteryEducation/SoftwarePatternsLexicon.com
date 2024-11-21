---
linkTitle: "2.3.9 Strategy"
title: "Strategy Pattern in Go: Implementing Flexible Algorithms"
description: "Explore the Strategy Pattern in Go, learn how to define interchangeable algorithms, and see practical examples with payment processing systems."
categories:
- Design Patterns
- Go Programming
- Software Architecture
tags:
- Strategy Pattern
- Behavioral Patterns
- GoF Patterns
- Go Language
- Software Design
date: 2024-10-25
type: docs
nav_weight: 239000
canonical: "https://softwarepatternslexicon.com/patterns-go/2/3/9"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.3.9 Strategy

The Strategy Pattern is a behavioral design pattern that enables selecting an algorithm's behavior at runtime. It defines a family of algorithms, encapsulates each one, and makes them interchangeable. This pattern allows the algorithm to vary independently from clients that use it, promoting flexibility and reusability.

### Purpose of the Strategy Pattern

- **Define a Family of Algorithms:** The Strategy Pattern allows you to define a family of algorithms, encapsulate each one, and make them interchangeable. This is particularly useful when you have multiple ways of performing an operation.
- **Independent Variation:** It lets the algorithm vary independently from clients that use it, which means you can add new algorithms without changing the client code.
- **Eliminate Conditional Statements:** By using the Strategy Pattern, you can eliminate complex conditional statements for selecting behaviors, leading to cleaner and more maintainable code.

### Implementation Steps

#### 1. Strategy Interface

The first step in implementing the Strategy Pattern is to define a strategy interface. This interface declares a method that all concrete strategies will implement.

```go
// PaymentStrategy defines the interface for payment strategies
type PaymentStrategy interface {
    Pay(amount float64) string
}
```

#### 2. Concrete Strategies

Next, implement the strategy interface with specific algorithms. Each concrete strategy encapsulates a different algorithm.

```go
// CreditCardStrategy is a concrete strategy for credit card payments
type CreditCardStrategy struct{}

func (c *CreditCardStrategy) Pay(amount float64) string {
    return fmt.Sprintf("Paid %.2f using Credit Card", amount)
}

// PayPalStrategy is a concrete strategy for PayPal payments
type PayPalStrategy struct{}

func (p *PayPalStrategy) Pay(amount float64) string {
    return fmt.Sprintf("Paid %.2f using PayPal", amount)
}

// CryptoStrategy is a concrete strategy for cryptocurrency payments
type CryptoStrategy struct{}

func (c *CryptoStrategy) Pay(amount float64) string {
    return fmt.Sprintf("Paid %.2f using Cryptocurrency", amount)
}
```

#### 3. Context

The context holds a reference to a strategy and allows the strategy to be changed at runtime. It delegates the algorithm execution to the strategy object.

```go
// PaymentContext is the context that uses a PaymentStrategy
type PaymentContext struct {
    strategy PaymentStrategy
}

// SetStrategy allows changing the strategy at runtime
func (p *PaymentContext) SetStrategy(strategy PaymentStrategy) {
    p.strategy = strategy
}

// ExecutePayment executes the payment using the current strategy
func (p *PaymentContext) ExecutePayment(amount float64) string {
    return p.strategy.Pay(amount)
}
```

### When to Use

- **Multiple Ways of Performing an Operation:** Use the Strategy Pattern when you have multiple ways of performing an operation and want to switch between them easily.
- **Eliminate Conditional Statements:** It is particularly useful for eliminating complex conditional statements that select different behaviors.

### Go-Specific Tips

- **Function Types as Strategies:** In Go, you can use function types or first-class functions as strategies for simplicity. This approach can reduce boilerplate code.
- **Stateless Strategies:** Ensure strategies are stateless or manage state appropriately to avoid unexpected behavior in concurrent environments.

### Example: Payment Processing System

Let's look at a practical example of a payment processing system that supports different payment strategies such as credit card, PayPal, and cryptocurrency.

```go
package main

import (
    "fmt"
)

// PaymentStrategy defines the interface for payment strategies
type PaymentStrategy interface {
    Pay(amount float64) string
}

// CreditCardStrategy is a concrete strategy for credit card payments
type CreditCardStrategy struct{}

func (c *CreditCardStrategy) Pay(amount float64) string {
    return fmt.Sprintf("Paid %.2f using Credit Card", amount)
}

// PayPalStrategy is a concrete strategy for PayPal payments
type PayPalStrategy struct{}

func (p *PayPalStrategy) Pay(amount float64) string {
    return fmt.Sprintf("Paid %.2f using PayPal", amount)
}

// CryptoStrategy is a concrete strategy for cryptocurrency payments
type CryptoStrategy struct{}

func (c *CryptoStrategy) Pay(amount float64) string {
    return fmt.Sprintf("Paid %.2f using Cryptocurrency", amount)
}

// PaymentContext is the context that uses a PaymentStrategy
type PaymentContext struct {
    strategy PaymentStrategy
}

// SetStrategy allows changing the strategy at runtime
func (p *PaymentContext) SetStrategy(strategy PaymentStrategy) {
    p.strategy = strategy
}

// ExecutePayment executes the payment using the current strategy
func (p *PaymentContext) ExecutePayment(amount float64) string {
    return p.strategy.Pay(amount)
}

func main() {
    context := &PaymentContext{}

    // Use Credit Card Strategy
    context.SetStrategy(&CreditCardStrategy{})
    fmt.Println(context.ExecutePayment(100.0))

    // Switch to PayPal Strategy
    context.SetStrategy(&PayPalStrategy{})
    fmt.Println(context.ExecutePayment(200.0))

    // Switch to Cryptocurrency Strategy
    context.SetStrategy(&CryptoStrategy{})
    fmt.Println(context.ExecutePayment(300.0))
}
```

### Advantages and Disadvantages

**Advantages:**

- **Flexibility:** Easily switch between different algorithms at runtime.
- **Scalability:** Add new strategies without modifying existing code.
- **Maintainability:** Reduce complex conditional logic, improving code readability.

**Disadvantages:**

- **Overhead:** Introduces additional classes or functions, which may increase complexity.
- **Increased Complexity:** May lead to a higher number of objects in the system.

### Best Practices

- **Use Interfaces Wisely:** Define clear and concise interfaces for strategies to ensure flexibility.
- **Keep Strategies Stateless:** Design strategies to be stateless or manage state carefully to avoid concurrency issues.
- **Leverage Function Types:** Consider using function types for strategies in Go to simplify the implementation.

### Comparisons

The Strategy Pattern is often compared with the State Pattern. While both involve changing behavior, the Strategy Pattern is used for interchangeable algorithms, whereas the State Pattern is used for changing behavior based on an object's state.

### Conclusion

The Strategy Pattern is a powerful tool for designing flexible and maintainable software. By encapsulating algorithms and making them interchangeable, you can create systems that are easy to extend and modify. In Go, leveraging interfaces and function types can simplify the implementation of this pattern, making it a valuable addition to your design toolkit.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of the Strategy Pattern?

- [x] To define a family of algorithms, encapsulate each one, and make them interchangeable.
- [ ] To manage the state of an object.
- [ ] To provide a simplified interface to a complex subsystem.
- [ ] To ensure a class has only one instance.

> **Explanation:** The Strategy Pattern is used to define a family of algorithms, encapsulate each one, and make them interchangeable, allowing the algorithm to vary independently from clients that use it.

### Which component in the Strategy Pattern holds a reference to a strategy?

- [x] Context
- [ ] Strategy Interface
- [ ] Concrete Strategy
- [ ] Client

> **Explanation:** The Context holds a reference to a strategy and allows the strategy to be changed at runtime.

### When should you use the Strategy Pattern?

- [x] When you have multiple ways of performing an operation.
- [ ] When you need to manage the lifecycle of an object.
- [ ] When you want to provide a unified interface to a set of interfaces.
- [ ] When you need to ensure a class has only one instance.

> **Explanation:** The Strategy Pattern is useful when you have multiple ways of performing an operation and want to switch between them easily.

### In Go, what can be used as strategies for simplicity?

- [x] Function types or first-class functions
- [ ] Structs only
- [ ] Methods only
- [ ] Interfaces only

> **Explanation:** In Go, you can use function types or first-class functions as strategies for simplicity, reducing boilerplate code.

### What is a disadvantage of the Strategy Pattern?

- [x] It introduces additional classes or functions, which may increase complexity.
- [ ] It makes the code less flexible.
- [ ] It reduces the number of objects in the system.
- [ ] It makes the code harder to maintain.

> **Explanation:** The Strategy Pattern can introduce additional classes or functions, which may increase complexity.

### How does the Strategy Pattern improve maintainability?

- [x] By reducing complex conditional logic
- [ ] By increasing the number of classes
- [ ] By making the code less flexible
- [ ] By ensuring a class has only one instance

> **Explanation:** The Strategy Pattern improves maintainability by reducing complex conditional logic, leading to cleaner and more readable code.

### What is a best practice when implementing the Strategy Pattern in Go?

- [x] Keep strategies stateless
- [ ] Use global variables for strategies
- [ ] Avoid using interfaces
- [ ] Implement strategies as methods only

> **Explanation:** A best practice when implementing the Strategy Pattern in Go is to keep strategies stateless or manage state carefully to avoid concurrency issues.

### Which pattern is often compared with the Strategy Pattern?

- [x] State Pattern
- [ ] Singleton Pattern
- [ ] Observer Pattern
- [ ] Factory Pattern

> **Explanation:** The Strategy Pattern is often compared with the State Pattern. While both involve changing behavior, the Strategy Pattern is used for interchangeable algorithms, whereas the State Pattern is used for changing behavior based on an object's state.

### What is the role of the Concrete Strategy in the Strategy Pattern?

- [x] It implements the strategy interface with specific algorithms.
- [ ] It holds a reference to a strategy.
- [ ] It defines the method that all strategies will implement.
- [ ] It provides a simplified interface to a complex subsystem.

> **Explanation:** The Concrete Strategy implements the strategy interface with specific algorithms.

### True or False: The Strategy Pattern allows algorithms to vary independently from clients that use them.

- [x] True
- [ ] False

> **Explanation:** True. The Strategy Pattern allows algorithms to vary independently from clients that use them, promoting flexibility and reusability.

{{< /quizdown >}}
