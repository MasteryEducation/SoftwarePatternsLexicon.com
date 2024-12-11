---
canonical: "https://softwarepatternslexicon.com/patterns-java/13/3/5"

title: "Factories in DDD: Mastering Domain-Driven Design with Java"
description: "Explore the role of factories in Domain-Driven Design (DDD) for creating complex domain objects in Java, ensuring correct instantiation while maintaining encapsulation and invariants."
linkTitle: "13.3.5 Factories in DDD"
tags:
- "Java"
- "Domain-Driven Design"
- "Factories"
- "Design Patterns"
- "Software Architecture"
- "Object-Oriented Programming"
- "Best Practices"
- "Advanced Java"
date: 2024-11-25
type: docs
nav_weight: 133500
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 13.3.5 Factories in DDD

In the realm of Domain-Driven Design (DDD), factories play a pivotal role in the creation of complex domain objects. They ensure that domain entities are instantiated correctly while maintaining encapsulation and invariants. This section delves into the intricacies of factories within DDD, providing a comprehensive understanding of their purpose, implementation, and interaction with other domain components.

### Understanding Factories in the Domain Model

**Factories** are design patterns used to encapsulate the instantiation logic of complex objects. In DDD, factories are crucial for creating domain objects that require intricate setup or validation processes. They help maintain the integrity of the domain model by ensuring that objects are created in a valid state, adhering to the business rules and constraints.

#### Purpose of Factories

- **Encapsulation**: Factories encapsulate the creation logic, hiding the complexities involved in constructing domain objects.
- **Consistency**: They ensure that objects are created consistently, adhering to the domain's invariants and business rules.
- **Separation of Concerns**: By delegating the creation logic to factories, the domain model remains focused on business logic rather than object instantiation.

### Factory Methods vs. Factory Objects

In DDD, factories can be implemented as either factory methods or factory objects. Understanding the distinction between these two approaches is essential for selecting the appropriate pattern for your domain model.

#### Factory Methods

A **Factory Method** is a static method that returns an instance of a class. It is typically used when the creation logic is simple and does not require maintaining state between invocations.

```java
public class Order {
    private List<Item> items;
    private Customer customer;

    private Order(List<Item> items, Customer customer) {
        this.items = items;
        this.customer = customer;
    }

    // Factory Method
    public static Order createOrder(List<Item> items, Customer customer) {
        validateItems(items);
        return new Order(items, customer);
    }

    private static void validateItems(List<Item> items) {
        if (items.isEmpty()) {
            throw new IllegalArgumentException("Order must contain at least one item.");
        }
    }
}
```

**Explanation**: In this example, the `createOrder` method acts as a factory method, encapsulating the validation logic and ensuring that an `Order` is created only if it contains at least one item.

#### Factory Objects

A **Factory Object** is a dedicated class responsible for creating instances of other classes. This approach is suitable when the creation logic is complex or when multiple related objects need to be created together.

```java
public class OrderFactory {
    public Order createOrder(List<Item> items, Customer customer) {
        validateItems(items);
        return new Order(items, customer);
    }

    private void validateItems(List<Item> items) {
        if (items.isEmpty()) {
            throw new IllegalArgumentException("Order must contain at least one item.");
        }
    }
}
```

**Explanation**: The `OrderFactory` class encapsulates the creation logic, providing a clear separation of concerns and allowing for more complex instantiation processes.

### Implementing Factories for Entities and Aggregates

In DDD, entities and aggregates represent core concepts of the domain model. Factories are instrumental in creating these objects, ensuring that they are instantiated correctly and consistently.

#### Factories for Entities

Entities are objects that have a distinct identity and lifecycle. When creating entities, factories ensure that all necessary invariants are maintained.

```java
public class CustomerFactory {
    public Customer createCustomer(String name, String email) {
        validateEmail(email);
        return new Customer(UUID.randomUUID(), name, email);
    }

    private void validateEmail(String email) {
        if (!email.contains("@")) {
            throw new IllegalArgumentException("Invalid email address.");
        }
    }
}
```

**Explanation**: The `CustomerFactory` class is responsible for creating `Customer` entities, ensuring that each customer has a unique identifier and a valid email address.

#### Factories for Aggregates

Aggregates are clusters of domain objects that are treated as a single unit. Factories for aggregates ensure that all components of the aggregate are created and initialized correctly.

```java
public class ShoppingCartFactory {
    public ShoppingCart createShoppingCart(Customer customer) {
        ShoppingCart cart = new ShoppingCart(customer);
        cart.addItems(defaultItems());
        return cart;
    }

    private List<Item> defaultItems() {
        // Return a list of default items
        return Arrays.asList(new Item("Default Item", 1));
    }
}
```

**Explanation**: The `ShoppingCartFactory` class creates a `ShoppingCart` aggregate, initializing it with a customer and a set of default items.

### Maintaining Invariants and Encapsulation

Factories contribute significantly to maintaining invariants and encapsulation within the domain model. By centralizing the creation logic, factories ensure that domain objects are always in a valid state upon creation.

#### Ensuring Invariants

Invariants are conditions that must always hold true for a domain object. Factories enforce these invariants by validating input parameters and initializing objects correctly.

- **Validation**: Factories perform necessary validations to ensure that objects are created in a valid state.
- **Initialization**: They initialize objects with default or calculated values, ensuring consistency across the domain model.

#### Promoting Encapsulation

Encapsulation is a fundamental principle of object-oriented design, ensuring that an object's internal state is hidden from the outside world. Factories promote encapsulation by:

- **Hiding Complexity**: They hide the complexity of object creation, exposing only the necessary interfaces to the client.
- **Protecting State**: By controlling the instantiation process, factories protect the internal state of domain objects from unauthorized access or modification.

### Keeping Factories Within the Domain Layer

In DDD, it is crucial to keep factories within the domain layer to maintain the integrity and cohesion of the domain model. Factories should be part of the domain layer, as they are responsible for creating domain objects and enforcing domain rules.

- **Domain Logic**: Factories encapsulate domain-specific logic, ensuring that objects are created according to the business rules.
- **Cohesion**: By keeping factories within the domain layer, the domain model remains cohesive, with all related components residing in the same layer.

### Interaction Between Factories and Repositories

Factories and repositories often interact closely, especially in the creation and retrieval of aggregates. Understanding this interaction is essential for designing a robust domain model.

#### Aggregate Creation

When creating aggregates, factories and repositories work together to ensure that all components of the aggregate are correctly initialized and persisted.

- **Factory Role**: The factory is responsible for creating the aggregate and initializing its components.
- **Repository Role**: The repository is responsible for persisting the aggregate and retrieving it from the data store.

```java
public class OrderService {
    private final OrderFactory orderFactory;
    private final OrderRepository orderRepository;

    public OrderService(OrderFactory orderFactory, OrderRepository orderRepository) {
        this.orderFactory = orderFactory;
        this.orderRepository = orderRepository;
    }

    public Order createAndSaveOrder(List<Item> items, Customer customer) {
        Order order = orderFactory.createOrder(items, customer);
        orderRepository.save(order);
        return order;
    }
}
```

**Explanation**: The `OrderService` class demonstrates the interaction between the `OrderFactory` and `OrderRepository`, where the factory creates the order and the repository persists it.

### Conclusion

Factories are indispensable in DDD for creating complex domain objects, ensuring that they are instantiated correctly while maintaining encapsulation and invariants. By encapsulating the creation logic, factories promote consistency and separation of concerns within the domain model. Keeping factories within the domain layer and understanding their interaction with repositories are crucial for designing a cohesive and robust domain model.

### Key Takeaways

- Factories encapsulate the instantiation logic of complex domain objects, ensuring consistency and adherence to business rules.
- Factory methods and factory objects offer different approaches to implementing factories, each suited to specific scenarios.
- Factories play a vital role in maintaining invariants and encapsulation within the domain model.
- Keeping factories within the domain layer ensures cohesion and integrity of the domain model.
- Understanding the interaction between factories and repositories is essential for aggregate creation and persistence.

### Exercises

1. Implement a factory for a `Product` entity that ensures each product has a unique SKU and a valid price.
2. Create a factory for an `Invoice` aggregate that initializes it with a list of `LineItem` entities and calculates the total amount.
3. Modify the `OrderFactory` example to include a discount calculation based on the customer's membership level.

### Reflection

Consider how factories can be applied to your current projects. Are there areas where encapsulating the creation logic could improve consistency and maintainability? Reflect on the role of factories in your domain model and how they contribute to the overall architecture.

## Test Your Knowledge: Factories in Domain-Driven Design Quiz

{{< quizdown >}}

### What is the primary purpose of factories in DDD?

- [x] To encapsulate the instantiation logic of complex domain objects.
- [ ] To manage database transactions.
- [ ] To handle user authentication.
- [ ] To perform data validation.

> **Explanation:** Factories encapsulate the instantiation logic, ensuring that domain objects are created consistently and correctly.

### How do factory methods differ from factory objects?

- [x] Factory methods are static methods, while factory objects are dedicated classes.
- [ ] Factory methods are used for database operations, while factory objects manage user sessions.
- [ ] Factory methods are used for logging, while factory objects handle exceptions.
- [ ] Factory methods are dynamic, while factory objects are static.

> **Explanation:** Factory methods are static methods that return instances, while factory objects are classes dedicated to creating instances.

### Why is it important to keep factories within the domain layer?

- [x] To maintain the integrity and cohesion of the domain model.
- [ ] To improve database performance.
- [ ] To simplify user interface design.
- [ ] To enhance network security.

> **Explanation:** Keeping factories within the domain layer ensures that domain-specific logic and rules are encapsulated and cohesive.

### What role do factories play in maintaining invariants?

- [x] They validate input parameters and initialize objects correctly.
- [ ] They manage user permissions.
- [ ] They handle error logging.
- [ ] They optimize memory usage.

> **Explanation:** Factories ensure that objects are created in a valid state by performing necessary validations and initializations.

### How do factories promote encapsulation?

- [x] By hiding the complexity of object creation.
- [ ] By managing network connections.
- [ ] By encrypting data.
- [ ] By compressing files.

> **Explanation:** Factories hide the complexity of object creation, exposing only necessary interfaces to the client.

### What is the relationship between factories and repositories in aggregate creation?

- [x] Factories create aggregates, while repositories persist them.
- [ ] Factories manage user sessions, while repositories handle logging.
- [ ] Factories encrypt data, while repositories compress files.
- [ ] Factories optimize memory usage, while repositories manage network connections.

> **Explanation:** Factories are responsible for creating aggregates, and repositories handle their persistence and retrieval.

### Which of the following is a benefit of using factory objects?

- [x] They provide a clear separation of concerns.
- [ ] They increase network bandwidth.
- [ ] They reduce disk space usage.
- [ ] They enhance user interface design.

> **Explanation:** Factory objects encapsulate creation logic, providing a clear separation of concerns within the domain model.

### What is an invariant in the context of DDD?

- [x] A condition that must always hold true for a domain object.
- [ ] A temporary variable used in calculations.
- [ ] A user interface component.
- [ ] A network protocol.

> **Explanation:** Invariants are conditions that must always be true for a domain object, ensuring its validity.

### How do factories contribute to separation of concerns?

- [x] By delegating creation logic away from the domain model.
- [ ] By managing database connections.
- [ ] By handling user input.
- [ ] By optimizing CPU usage.

> **Explanation:** Factories delegate the creation logic, allowing the domain model to focus on business logic.

### True or False: Factories should always be implemented as static methods.

- [ ] True
- [x] False

> **Explanation:** Factories can be implemented as either static methods or dedicated classes, depending on the complexity of the creation logic.

{{< /quizdown >}}

---
