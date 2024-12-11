---
canonical: "https://softwarepatternslexicon.com/patterns-java/13/3/1"
title: "Entities and Value Objects in Domain-Driven Design"
description: "Explore the foundational concepts of Entities and Value Objects in Domain-Driven Design, focusing on identity, immutability, and practical implementation in Java."
linkTitle: "13.3.1 Entities and Value Objects"
tags:
- "Java"
- "Domain-Driven Design"
- "Entities"
- "Value Objects"
- "Immutability"
- "Design Patterns"
- "Software Architecture"
- "Best Practices"
date: 2024-11-25
type: docs
nav_weight: 133100
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 13.3.1 Entities and Value Objects

In the realm of Domain-Driven Design (DDD), understanding the distinction between **Entities** and **Value Objects** is crucial for modeling domain concepts effectively. These constructs help developers create a robust and expressive domain model by focusing on identity and immutability. This section delves into the characteristics, implementation, and best practices for using Entities and Value Objects in Java.

### Understanding Entities

#### Characteristics of Entities

Entities are objects that have a distinct identity that persists over time. This identity is crucial because it differentiates one entity from another, even if their attributes are identical. Entities are mutable, meaning their state can change over time, but their identity remains constant.

- **Unique Identity**: Each entity has a unique identifier, often represented by a primary key in a database.
- **Lifecycle**: Entities have a lifecycle that includes creation, modification, and deletion.
- **State Changes**: Entities can undergo state changes while maintaining their identity.

#### When to Use Entities

Entities are suitable when you need to track the lifecycle and identity of a domain object. They are ideal for scenarios where the object's identity is more important than its attributes. Common examples include users, orders, and products in an e-commerce system.

### Implementing Entities in Java

To implement an entity in Java, you typically define a class with a unique identifier and mutable attributes. Here's an example of a simple `User` entity:

```java
public class User {
    private final String id; // Unique identifier
    private String name;
    private String email;

    public User(String id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        User user = (User) o;
        return id.equals(user.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
```

In this example, the `User` class has a unique `id` that serves as its identity. The `equals` and `hashCode` methods are overridden to ensure equality is based on the `id`.

### Understanding Value Objects

#### Characteristics of Value Objects

Value Objects are immutable and defined by their attributes rather than a unique identity. They are used to represent descriptive aspects of the domain with no conceptual identity.

- **Immutability**: Once created, the state of a value object cannot change.
- **Equality Based on Attributes**: Two value objects are equal if all their attributes are equal.
- **No Identity**: Value objects do not have a unique identifier.

#### When to Use Value Objects

Value Objects are ideal for modeling domain concepts that are defined by their attributes rather than identity. They are useful for representing quantities, measurements, or other descriptive elements. Examples include money, dates, and addresses.

### Implementing Value Objects in Java

To implement a value object in Java, define a class with final attributes and no setters. Here's an example of a `Money` value object:

```java
public final class Money {
    private final double amount;
    private final String currency;

    public Money(double amount, String currency) {
        this.amount = amount;
        this.currency = currency;
    }

    public double getAmount() {
        return amount;
    }

    public String getCurrency() {
        return currency;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Money money = (Money) o;
        return Double.compare(money.amount, amount) == 0 &&
               currency.equals(money.currency);
    }

    @Override
    public int hashCode() {
        return Objects.hash(amount, currency);
    }
}
```

In this example, the `Money` class is immutable, with final fields and no setters. Equality is determined by comparing the `amount` and `currency`.

### Best Practices for Designing Entities and Value Objects

#### Designing Entities

- **Use a Unique Identifier**: Ensure each entity has a unique identifier that persists throughout its lifecycle.
- **Encapsulate State Changes**: Use methods to encapsulate state changes and maintain invariants.
- **Implement Equality Based on Identity**: Override `equals` and `hashCode` to use the unique identifier for equality checks.

#### Designing Value Objects

- **Ensure Immutability**: Use final fields and no setters to ensure immutability.
- **Implement Comprehensive Equality**: Override `equals` and `hashCode` to compare all attributes.
- **Use Value Objects for Clarity**: Use value objects to enhance code clarity and prevent logical errors by encapsulating domain concepts.

### Enhancing Code Clarity with Value Objects

Value Objects can significantly enhance code clarity by encapsulating domain concepts and reducing the risk of logical errors. For example, using a `Money` value object instead of a `double` for monetary values can prevent errors related to currency mismatches.

### Practical Applications and Real-World Scenarios

Consider a scenario in an e-commerce application where you need to model a `Product` entity and a `Price` value object. The `Product` entity has a unique identifier and mutable attributes, while the `Price` value object represents the cost of the product in a specific currency.

```java
public class Product {
    private final String id;
    private String name;
    private Money price;

    public Product(String id, String name, Money price) {
        this.id = id;
        this.name = name;
        this.price = price;
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Money getPrice() {
        return price;
    }

    public void setPrice(Money price) {
        this.price = price;
    }
}
```

In this example, the `Product` entity uses the `Money` value object to represent its price, ensuring that monetary values are handled consistently and accurately.

### Conclusion

Entities and Value Objects are fundamental building blocks in Domain-Driven Design, each serving a distinct purpose in modeling domain concepts. By understanding their characteristics and applying best practices, developers can create robust and expressive domain models that enhance code clarity and maintainability.

### Quiz: Test Your Knowledge on Entities and Value Objects

{{< quizdown >}}

### What is the primary characteristic that distinguishes an entity from a value object?

- [x] Unique identity
- [ ] Immutability
- [ ] Attribute-based equality
- [ ] Lack of lifecycle

> **Explanation:** Entities have a unique identity that distinguishes them from other objects, even if their attributes are identical.

### Which of the following is a key feature of value objects?

- [x] Immutability
- [ ] Unique identity
- [ ] Mutable state
- [ ] Lifecycle management

> **Explanation:** Value objects are immutable, meaning their state cannot change once they are created.

### When should you use a value object instead of an entity?

- [x] When the object's identity is not important
- [ ] When the object has a unique identifier
- [ ] When the object has a lifecycle
- [ ] When the object needs to track state changes

> **Explanation:** Value objects are used when the object's identity is not important, and it is defined by its attributes.

### How do you ensure immutability in a Java value object?

- [x] Use final fields and no setters
- [ ] Use mutable fields and setters
- [ ] Use a unique identifier
- [ ] Override equals and hashCode

> **Explanation:** Immutability is ensured by using final fields and not providing setters for the attributes.

### What is the purpose of overriding equals and hashCode in entities?

- [x] To ensure equality is based on identity
- [ ] To ensure equality is based on attributes
- [ ] To manage the object's lifecycle
- [ ] To encapsulate state changes

> **Explanation:** In entities, equals and hashCode are overridden to ensure equality checks are based on the unique identity of the object.

### Which of the following is a benefit of using value objects?

- [x] Enhanced code clarity
- [ ] Unique identity
- [ ] Mutable state
- [ ] Lifecycle management

> **Explanation:** Value objects enhance code clarity by encapsulating domain concepts and reducing logical errors.

### What is a common use case for entities in a domain model?

- [x] Tracking the lifecycle of a user
- [ ] Representing a monetary value
- [ ] Encapsulating a measurement
- [ ] Defining a descriptive aspect

> **Explanation:** Entities are used to track the lifecycle and identity of domain objects, such as users.

### How can value objects prevent logical errors in code?

- [x] By encapsulating domain concepts
- [ ] By providing unique identifiers
- [ ] By allowing state changes
- [ ] By managing lifecycles

> **Explanation:** Value objects encapsulate domain concepts, which helps prevent logical errors by ensuring consistent handling of values.

### What is the role of a unique identifier in an entity?

- [x] To distinguish it from other entities
- [ ] To ensure immutability
- [ ] To define attribute-based equality
- [ ] To encapsulate domain concepts

> **Explanation:** A unique identifier distinguishes an entity from other entities, even if their attributes are identical.

### True or False: Value objects can have mutable state.

- [ ] True
- [x] False

> **Explanation:** Value objects are immutable, meaning their state cannot change once they are created.

{{< /quizdown >}}
