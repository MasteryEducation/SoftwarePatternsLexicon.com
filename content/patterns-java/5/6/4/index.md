---
canonical: "https://softwarepatternslexicon.com/patterns-java/5/6/4"
title: "Implementing Sealed Classes in Java: A Comprehensive Guide"
description: "Explore the implementation of sealed classes in Java, including practical examples, subclass modifiers, and integration with pattern matching."
linkTitle: "5.6.4 Implementing Sealed Classes in Java"
tags:
- "Java"
- "Sealed Classes"
- "Design Patterns"
- "Advanced Java"
- "Pattern Matching"
- "Domain Modeling"
- "Java 17"
- "Object-Oriented Programming"
date: 2024-11-25
type: docs
nav_weight: 56400
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 5.6.4 Implementing Sealed Classes in Java

### Introduction

Sealed classes, introduced in Java 17, represent a significant evolution in the language's type system. They allow developers to define a restricted hierarchy of classes, providing more control over inheritance and enhancing the expressiveness of domain models. This section delves into the practical implementation of sealed classes, exploring their syntax, use cases, and integration with modern Java features like pattern matching.

### Understanding Sealed Classes

Sealed classes are a type of class that restricts which other classes or interfaces may extend or implement them. This feature is particularly useful in scenarios where a fixed set of subclasses is known and desired, allowing for more predictable and maintainable code.

#### Declaring Sealed Classes

To declare a sealed class, use the `sealed` keyword followed by a list of permitted subclasses. This list explicitly defines which classes can extend the sealed class, ensuring that no other classes can do so.

```java
public sealed class Shape permits Circle, Rectangle, Square {
    // Common methods and fields for all shapes
}
```

In this example, `Shape` is a sealed class, and only `Circle`, `Rectangle`, and `Square` are permitted to extend it.

#### Modifiers for Subclasses

Subclasses of a sealed class must specify one of the following modifiers:

- **`final`**: The subclass cannot be extended further.
- **`sealed`**: The subclass can be extended, but only by a specified set of classes.
- **`non-sealed`**: The subclass can be extended by any class, effectively breaking the seal.

```java
public final class Circle extends Shape {
    // Implementation specific to Circle
}

public sealed class Rectangle extends Shape permits FilledRectangle {
    // Implementation specific to Rectangle
}

public non-sealed class Square extends Shape {
    // Implementation specific to Square
}
```

In this example, `Circle` is a final class, `Rectangle` is a sealed class with `FilledRectangle` as its permitted subclass, and `Square` is non-sealed, allowing any class to extend it.

### Integration with Pattern Matching

Sealed classes integrate seamlessly with pattern matching in switch expressions, a feature that enhances readability and reduces boilerplate code.

```java
public double calculateArea(Shape shape) {
    return switch (shape) {
        case Circle c -> Math.PI * c.radius() * c.radius();
        case Rectangle r -> r.length() * r.width();
        case Square s -> s.side() * s.side();
    };
}
```

In this example, the switch expression uses pattern matching to determine the type of `Shape` and calculate its area accordingly. The compiler ensures that all possible subclasses are covered, providing compile-time safety.

### Scenarios for Using Sealed Classes

Sealed classes are particularly beneficial in the following scenarios:

- **Domain Modeling**: When modeling a domain with a fixed set of types, sealed classes provide a clear and concise way to represent these types and their relationships.
- **API Design**: In API design, sealed classes can enforce a controlled extension of types, ensuring that only intended subclasses are used.
- **Pattern Matching**: Sealed classes enhance the power of pattern matching by ensuring exhaustive checks, reducing runtime errors.

### Practical Example: A Sealed Class Hierarchy

Consider a financial application that models different types of accounts. Using sealed classes, you can define a hierarchy that restricts the types of accounts to a known set.

```java
public sealed class Account permits SavingsAccount, CheckingAccount, CreditAccount {
    protected double balance;

    public double getBalance() {
        return balance;
    }
}

public final class SavingsAccount extends Account {
    private double interestRate;

    public double getInterestRate() {
        return interestRate;
    }
}

public final class CheckingAccount extends Account {
    private double overdraftLimit;

    public double getOverdraftLimit() {
        return overdraftLimit;
    }
}

public final class CreditAccount extends Account {
    private double creditLimit;

    public double getCreditLimit() {
        return creditLimit;
    }
}
```

In this example, `Account` is a sealed class with three permitted subclasses: `SavingsAccount`, `CheckingAccount`, and `CreditAccount`. Each subclass has specific fields and methods relevant to its type.

### Benefits and Drawbacks

#### Benefits

- **Enhanced Type Safety**: By restricting subclassing, sealed classes provide better type safety and prevent unintended extensions.
- **Improved Readability**: The explicit declaration of permitted subclasses makes the code more readable and understandable.
- **Exhaustive Pattern Matching**: Sealed classes ensure that all possible subclasses are considered in pattern matching, reducing runtime errors.

#### Drawbacks

- **Limited Flexibility**: The restriction on subclassing can be limiting in scenarios where extensibility is required.
- **Increased Complexity**: Managing a sealed hierarchy can introduce complexity, especially in large systems with many types.

### Best Practices

- **Use Sealed Classes for Fixed Hierarchies**: When the set of subclasses is known and unlikely to change, sealed classes provide a robust solution.
- **Combine with Pattern Matching**: Leverage pattern matching to simplify code and ensure exhaustive checks.
- **Document Subclass Relationships**: Clearly document the relationships between sealed classes and their subclasses to aid understanding and maintenance.

### Conclusion

Sealed classes in Java offer a powerful tool for controlling inheritance and enhancing domain models. By understanding their syntax, use cases, and integration with pattern matching, developers can create more maintainable and predictable code. As with any feature, it's essential to weigh the benefits against the drawbacks and apply sealed classes judiciously in appropriate scenarios.

### Further Reading

- [Java Sealed Classes Documentation](https://docs.oracle.com/en/java/javase/17/language/sealed-classes-and-interfaces.html)
- [Pattern Matching in Java](https://docs.oracle.com/en/java/javase/17/language/pattern-matching.html)

### Exercises

1. Implement a sealed class hierarchy for a transportation system with classes like `Car`, `Bus`, and `Bicycle`.
2. Modify the financial application example to include a new type of account, ensuring that the sealed class hierarchy is updated accordingly.
3. Experiment with pattern matching in switch expressions using sealed classes to handle different types of events in an event-driven system.

### Quiz

## Test Your Knowledge: Sealed Classes in Java Quiz

{{< quizdown >}}

### What is the primary purpose of sealed classes in Java?

- [x] To restrict which classes can extend or implement them.
- [ ] To enhance performance by reducing class loading time.
- [ ] To simplify the syntax of class declarations.
- [ ] To allow dynamic subclassing at runtime.

> **Explanation:** Sealed classes restrict which other classes can extend or implement them, providing more control over inheritance.

### Which keyword is used to declare a sealed class in Java?

- [x] sealed
- [ ] final
- [ ] abstract
- [ ] static

> **Explanation:** The `sealed` keyword is used to declare a sealed class in Java.

### What modifier must a subclass of a sealed class specify?

- [x] final, sealed, or non-sealed
- [ ] public, private, or protected
- [ ] static, abstract, or synchronized
- [ ] volatile, transient, or native

> **Explanation:** Subclasses of a sealed class must specify one of the following modifiers: `final`, `sealed`, or `non-sealed`.

### How do sealed classes enhance pattern matching in switch expressions?

- [x] By ensuring exhaustive checks of all possible subclasses.
- [ ] By allowing dynamic type inference.
- [ ] By reducing the number of case statements required.
- [ ] By enabling runtime type checking.

> **Explanation:** Sealed classes enhance pattern matching by ensuring that all possible subclasses are considered, providing compile-time safety.

### In which version of Java were sealed classes introduced?

- [x] Java 17
- [ ] Java 11
- [ ] Java 8
- [ ] Java 14

> **Explanation:** Sealed classes were introduced in Java 17.

### What is a potential drawback of using sealed classes?

- [x] Limited flexibility in extending classes.
- [ ] Increased runtime performance overhead.
- [ ] Reduced readability of code.
- [ ] Difficulty in integrating with legacy systems.

> **Explanation:** A potential drawback of sealed classes is the limited flexibility in extending classes, as they restrict subclassing.

### Which of the following is NOT a valid subclass modifier for a sealed class?

- [ ] final
- [ ] sealed
- [ ] non-sealed
- [x] abstract

> **Explanation:** `abstract` is not a valid subclass modifier for a sealed class. The valid modifiers are `final`, `sealed`, and `non-sealed`.

### What is a common use case for sealed classes?

- [x] Domain modeling with a fixed set of types.
- [ ] Enhancing multithreading performance.
- [ ] Simplifying database interactions.
- [ ] Improving network communication efficiency.

> **Explanation:** A common use case for sealed classes is domain modeling with a fixed set of types, where the hierarchy is known and controlled.

### How do sealed classes improve API design?

- [x] By enforcing a controlled extension of types.
- [ ] By reducing the number of public methods.
- [ ] By allowing dynamic method dispatch.
- [ ] By simplifying error handling.

> **Explanation:** Sealed classes improve API design by enforcing a controlled extension of types, ensuring that only intended subclasses are used.

### True or False: Sealed classes can be extended by any class.

- [ ] True
- [x] False

> **Explanation:** False. Sealed classes can only be extended by the classes specified in their permits clause.

{{< /quizdown >}}
