---
canonical: "https://softwarepatternslexicon.com/patterns-c-sharp/6/13"
title: "Specification Pattern in C#: Mastering Behavioral Design Patterns"
description: "Explore the Specification Pattern in C# to create reusable and combinable business rules. Learn how to implement, combine, and apply this pattern for validation and filtering with LINQ."
linkTitle: "6.13 Specification Pattern"
categories:
- CSharp Design Patterns
- Behavioral Patterns
- Software Architecture
tags:
- Specification Pattern
- CSharp Design Patterns
- Business Rules
- LINQ
- Software Architecture
date: 2024-11-17
type: docs
nav_weight: 7300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 6.13 Specification Pattern

The Specification Pattern is a powerful tool in the arsenal of software engineers and architects, particularly when dealing with complex business rules and logic. It allows for the encapsulation of business rules into reusable and combinable components, making your codebase more maintainable and flexible. In this section, we will delve into the Specification Pattern, exploring its implementation in C#, how to combine business rules, and practical use cases such as validation and filtering with LINQ.

### Specification Pattern Description

The Specification Pattern is a behavioral design pattern that defines a business rule in a separate class, allowing it to be reused and combined with other rules. This pattern is particularly useful when you have complex business logic that needs to be applied to various objects or entities within your application.

#### Intent

The primary intent of the Specification Pattern is to encapsulate business rules and logic in a way that they can be easily reused, combined, and modified without affecting the core business logic. This separation of concerns enhances the maintainability and scalability of the application.

#### Key Participants

- **Specification**: An interface or abstract class that defines the method(s) for evaluating the business rule.
- **ConcreteSpecification**: A class that implements the Specification interface and contains the actual business logic.
- **CompositeSpecification**: A class that allows for combining multiple specifications using logical operations such as AND, OR, and NOT.

### Implementing Specification in C#

To implement the Specification Pattern in C#, we start by defining a base `ISpecification<T>` interface that declares a method for evaluating the specification. This is followed by creating concrete specifications that implement this interface.

#### ISpecification Interface

```csharp
public interface ISpecification<T>
{
    bool IsSatisfiedBy(T candidate);
}
```

The `IsSatisfiedBy` method takes a candidate object and returns a boolean indicating whether the candidate satisfies the specification.

#### Concrete Specification

Let's create a concrete specification for a simple business rule: checking if a product is in stock.

```csharp
public class InStockSpecification : ISpecification<Product>
{
    public bool IsSatisfiedBy(Product product)
    {
        return product.Stock > 0;
    }
}
```

Here, the `InStockSpecification` checks if the `Stock` property of a `Product` object is greater than zero.

#### Composite Specification

To combine multiple specifications, we can create a `CompositeSpecification` class that allows logical operations.

```csharp
public abstract class CompositeSpecification<T> : ISpecification<T>
{
    public abstract bool IsSatisfiedBy(T candidate);

    public ISpecification<T> And(ISpecification<T> other)
    {
        return new AndSpecification<T>(this, other);
    }

    public ISpecification<T> Or(ISpecification<T> other)
    {
        return new OrSpecification<T>(this, other);
    }

    public ISpecification<T> Not()
    {
        return new NotSpecification<T>(this);
    }
}
```

#### Logical Operations

Implement the logical operations as separate classes.

```csharp
public class AndSpecification<T> : CompositeSpecification<T>
{
    private readonly ISpecification<T> _left;
    private readonly ISpecification<T> _right;

    public AndSpecification(ISpecification<T> left, ISpecification<T> right)
    {
        _left = left;
        _right = right;
    }

    public override bool IsSatisfiedBy(T candidate)
    {
        return _left.IsSatisfiedBy(candidate) && _right.IsSatisfiedBy(candidate);
    }
}

public class OrSpecification<T> : CompositeSpecification<T>
{
    private readonly ISpecification<T> _left;
    private readonly ISpecification<T> _right;

    public OrSpecification(ISpecification<T> left, ISpecification<T> right)
    {
        _left = left;
        _right = right;
    }

    public override bool IsSatisfiedBy(T candidate)
    {
        return _left.IsSatisfiedBy(candidate) || _right.IsSatisfiedBy(candidate);
    }
}

public class NotSpecification<T> : CompositeSpecification<T>
{
    private readonly ISpecification<T> _specification;

    public NotSpecification(ISpecification<T> specification)
    {
        _specification = specification;
    }

    public override bool IsSatisfiedBy(T candidate)
    {
        return !_specification.IsSatisfiedBy(candidate);
    }
}
```

### Combining Business Rules

The true power of the Specification Pattern lies in its ability to combine multiple specifications to form complex business rules. Let's see how we can achieve this.

#### Building Complex Rules

Suppose we have another specification to check if a product is expensive.

```csharp
public class ExpensiveSpecification : ISpecification<Product>
{
    private readonly decimal _priceThreshold;

    public ExpensiveSpecification(decimal priceThreshold)
    {
        _priceThreshold = priceThreshold;
    }

    public bool IsSatisfiedBy(Product product)
    {
        return product.Price > _priceThreshold;
    }
}
```

Now, we can combine the `InStockSpecification` and `ExpensiveSpecification` to create a new rule that checks if a product is both in stock and expensive.

```csharp
var inStockSpec = new InStockSpecification();
var expensiveSpec = new ExpensiveSpecification(1000);

var inStockAndExpensiveSpec = inStockSpec.And(expensiveSpec);

bool isSatisfied = inStockAndExpensiveSpec.IsSatisfiedBy(product);
```

### Use Cases and Examples

The Specification Pattern is versatile and can be applied in various scenarios, such as validation, filtering, and more.

#### Validation

In scenarios where multiple validation rules need to be applied to an object, the Specification Pattern provides a clean and maintainable approach.

```csharp
public class ProductValidator
{
    private readonly ISpecification<Product> _specification;

    public ProductValidator(ISpecification<Product> specification)
    {
        _specification = specification;
    }

    public bool Validate(Product product)
    {
        return _specification.IsSatisfiedBy(product);
    }
}
```

#### Filtering with LINQ

The Specification Pattern can be seamlessly integrated with LINQ to filter collections based on complex criteria.

```csharp
var products = GetProducts(); // Assume this returns a list of products

var inStockSpec = new InStockSpecification();
var expensiveSpec = new ExpensiveSpecification(1000);

var filteredProducts = products.Where(p => inStockSpec.And(expensiveSpec).IsSatisfiedBy(p)).ToList();
```

### Visualizing the Specification Pattern

To better understand the Specification Pattern, let's visualize the relationships between the different components using a class diagram.

```mermaid
classDiagram
    class ISpecification<T> {
        +bool IsSatisfiedBy(T candidate)
    }
    class InStockSpecification {
        +bool IsSatisfiedBy(Product product)
    }
    class ExpensiveSpecification {
        +bool IsSatisfiedBy(Product product)
    }
    class CompositeSpecification<T> {
        +ISpecification<T> And(ISpecification<T> other)
        +ISpecification<T> Or(ISpecification<T> other)
        +ISpecification<T> Not()
    }
    class AndSpecification<T> {
        +bool IsSatisfiedBy(T candidate)
    }
    class OrSpecification<T> {
        +bool IsSatisfiedBy(T candidate)
    }
    class NotSpecification<T> {
        +bool IsSatisfiedBy(T candidate)
    }

    ISpecification<T> <|-- InStockSpecification
    ISpecification<T> <|-- ExpensiveSpecification
    ISpecification<T> <|-- CompositeSpecification<T>
    CompositeSpecification<T> <|-- AndSpecification<T>
    CompositeSpecification<T> <|-- OrSpecification<T>
    CompositeSpecification<T> <|-- NotSpecification<T>
```

### Design Considerations

When implementing the Specification Pattern, consider the following:

- **Reusability**: Ensure that specifications are designed to be reusable across different parts of the application.
- **Composability**: Leverage the pattern's ability to combine specifications to create complex business rules.
- **Performance**: Be mindful of performance implications when combining multiple specifications, especially in large datasets.
- **C# Specific Features**: Utilize C# features such as generics and LINQ to enhance the flexibility and expressiveness of your specifications.

### Differences and Similarities

The Specification Pattern is often compared to other patterns like the Strategy Pattern and the Composite Pattern. While all these patterns deal with encapsulating behavior, the Specification Pattern is unique in its focus on business rules and logic evaluation.

- **Specification vs. Strategy**: The Strategy Pattern is used to encapsulate algorithms, while the Specification Pattern is used to encapsulate business rules.
- **Specification vs. Composite**: The Composite Pattern is used to treat individual objects and compositions uniformly, whereas the Specification Pattern is used to combine business rules.

### Try It Yourself

To solidify your understanding of the Specification Pattern, try modifying the code examples provided. For instance, create a new specification that checks if a product is on sale and combine it with existing specifications. Experiment with different combinations and see how they affect the outcome.

### Knowledge Check

Before we conclude, let's summarize the key takeaways:

- The Specification Pattern encapsulates business rules into reusable components.
- It allows for the combination of multiple specifications to form complex rules.
- The pattern is particularly useful in scenarios involving validation and filtering.
- CSharp features such as generics and LINQ enhance the implementation of the Specification Pattern.

Remember, mastering design patterns is a journey. As you continue to explore and apply these patterns, you'll gain deeper insights into building robust and maintainable software systems. Keep experimenting, stay curious, and enjoy the journey!

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Specification Pattern?

- [x] To encapsulate business rules into reusable components
- [ ] To encapsulate algorithms into interchangeable strategies
- [ ] To treat individual objects and compositions uniformly
- [ ] To separate the construction of a complex object from its representation

> **Explanation:** The Specification Pattern is designed to encapsulate business rules into reusable components, allowing for easy combination and modification.

### Which method is typically defined in the ISpecification interface?

- [x] IsSatisfiedBy
- [ ] Execute
- [ ] Validate
- [ ] Apply

> **Explanation:** The `IsSatisfiedBy` method is used to evaluate whether a candidate object satisfies the specification.

### How can multiple specifications be combined in the Specification Pattern?

- [x] Using logical operations such as AND, OR, and NOT
- [ ] By inheriting from a base specification class
- [ ] By implementing a common interface
- [ ] By using a factory method

> **Explanation:** The Specification Pattern allows for combining multiple specifications using logical operations like AND, OR, and NOT.

### What is a key advantage of using the Specification Pattern?

- [x] It enhances code reusability and maintainability
- [ ] It simplifies the user interface design
- [ ] It improves database performance
- [ ] It reduces network latency

> **Explanation:** The Specification Pattern enhances code reusability and maintainability by encapsulating business rules into reusable components.

### In what scenarios is the Specification Pattern particularly useful?

- [x] Validation and filtering
- [ ] User interface design
- [ ] Database schema design
- [ ] Network communication

> **Explanation:** The Specification Pattern is particularly useful in scenarios involving validation and filtering, where complex business rules need to be applied.

### What is the role of the CompositeSpecification class?

- [x] To allow for combining multiple specifications
- [ ] To define the core business logic
- [ ] To execute a specific algorithm
- [ ] To manage database connections

> **Explanation:** The CompositeSpecification class allows for combining multiple specifications using logical operations.

### How does the Specification Pattern differ from the Strategy Pattern?

- [x] Specification focuses on business rules, Strategy on algorithms
- [ ] Specification focuses on algorithms, Strategy on business rules
- [ ] Both patterns focus on encapsulating algorithms
- [ ] Both patterns focus on user interface design

> **Explanation:** The Specification Pattern focuses on encapsulating business rules, while the Strategy Pattern focuses on encapsulating algorithms.

### What C# feature enhances the implementation of the Specification Pattern?

- [x] Generics and LINQ
- [ ] Reflection
- [ ] Dynamic typing
- [ ] Anonymous methods

> **Explanation:** C# features such as generics and LINQ enhance the flexibility and expressiveness of the Specification Pattern.

### Which of the following is a logical operation used in the Specification Pattern?

- [x] AND
- [ ] XOR
- [ ] NAND
- [ ] NOR

> **Explanation:** The Specification Pattern commonly uses logical operations like AND, OR, and NOT to combine specifications.

### True or False: The Specification Pattern can be used to encapsulate user interface design.

- [ ] True
- [x] False

> **Explanation:** The Specification Pattern is not used for encapsulating user interface design; it is used for encapsulating business rules and logic.

{{< /quizdown >}}
