---
canonical: "https://softwarepatternslexicon.com/patterns-java/5/6/3"

title: "Understanding Sealed Classes in Java: A Comprehensive Guide"
description: "Explore the concept of sealed classes in Java, their purpose, syntax, and practical applications in controlling class hierarchies and enhancing pattern matching."
linkTitle: "5.6.3 Understanding Sealed Classes"
tags:
- "Java"
- "Sealed Classes"
- "Design Patterns"
- "Java 17"
- "Object-Oriented Programming"
- "Class Hierarchy"
- "Pattern Matching"
- "Advanced Java"
date: 2024-11-25
type: docs
nav_weight: 56300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 5.6.3 Understanding Sealed Classes

### Introduction to Sealed Classes

Sealed classes and interfaces, introduced in Java 17, represent a significant evolution in the language's type system. They provide developers with the ability to explicitly control which classes can extend or implement them. This feature is particularly useful in scenarios where a well-defined class hierarchy is essential, such as when modeling algebraic data types or ensuring exhaustive pattern matching.

### Purpose of Sealed Classes and Interfaces

Sealed classes and interfaces aim to offer more control over the inheritance hierarchy. By restricting which classes can extend a sealed class or implement a sealed interface, developers can:

- **Enhance Maintainability**: By controlling the class hierarchy, developers can prevent unauthorized or unintended extensions, reducing the risk of bugs and making the codebase easier to maintain.
- **Improve Security**: Limiting subclassing can prevent malicious or erroneous subclasses from being introduced, enhancing the security of the application.
- **Facilitate Exhaustive Pattern Matching**: Sealed types enable the compiler to perform exhaustive checks during pattern matching, ensuring all possible cases are handled.

### Syntax for Declaring Sealed Types

To declare a sealed class or interface, use the `sealed` keyword followed by a `permits` clause that specifies the permitted subclasses or implementing classes. Here's the basic syntax:

```java
public sealed class Shape permits Circle, Rectangle, Square {
    // Class body
}

public final class Circle extends Shape {
    // Class body
}

public final class Rectangle extends Shape {
    // Class body
}

public final class Square extends Shape {
    // Class body
}
```

### Key Characteristics of Sealed Classes

- **Explicit Permits Clause**: The `permits` clause explicitly lists all classes that are allowed to extend the sealed class. This list must be exhaustive.
- **Subclass Requirements**: Permitted subclasses must be either `final`, `sealed`, or `non-sealed`. This ensures that the hierarchy remains controlled.
- **Compile-Time Checks**: The Java compiler enforces the rules of sealed classes, providing compile-time errors if any constraints are violated.

### Practical Applications and Use Cases

#### Representing Algebraic Data Types

Sealed classes are particularly useful for representing algebraic data types (ADTs), which are common in functional programming languages. ADTs allow developers to define a type by enumerating its possible values. In Java, sealed classes can be used to model such types:

```java
public sealed interface Expression permits Constant, Add, Multiply {
    // Interface body
}

public final class Constant implements Expression {
    private final int value;

    public Constant(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}

public final class Add implements Expression {
    private final Expression left;
    private final Expression right;

    public Add(Expression left, Expression right) {
        this.left = left;
        this.right = right;
    }

    public Expression getLeft() {
        return left;
    }

    public Expression getRight() {
        return right;
    }
}

public final class Multiply implements Expression {
    private final Expression left;
    private final Expression right;

    public Multiply(Expression left, Expression right) {
        this.left = left;
        this.right = right;
    }

    public Expression getLeft() {
        return left;
    }

    public Expression getRight() {
        return right;
    }
}
```

In this example, the `Expression` interface is sealed, and its permitted implementations are `Constant`, `Add`, and `Multiply`. This setup allows for exhaustive pattern matching, as the compiler knows all possible implementations of `Expression`.

#### Enhancing Pattern Matching and Exhaustiveness Checks

Sealed classes improve pattern matching by enabling the compiler to perform exhaustiveness checks. This ensures that all possible cases are handled, reducing the likelihood of runtime errors. Consider the following example:

```java
public int evaluate(Expression expr) {
    return switch (expr) {
        case Constant c -> c.getValue();
        case Add a -> evaluate(a.getLeft()) + evaluate(a.getRight());
        case Multiply m -> evaluate(m.getLeft()) * evaluate(m.getRight());
    };
}
```

In this `switch` expression, the compiler can verify that all possible cases of `Expression` are covered, providing a compile-time guarantee of exhaustiveness.

### Historical Context and Evolution

The concept of sealed classes is not new and has been present in other programming languages like Scala and Kotlin. Java's introduction of sealed classes reflects a broader trend towards incorporating functional programming concepts and enhancing type safety. This evolution aligns with Java's ongoing efforts to modernize the language while maintaining backward compatibility.

### Best Practices and Considerations

- **Use Sealed Classes for Well-Defined Hierarchies**: Sealed classes are ideal for scenarios where the class hierarchy is fixed and should not be extended arbitrarily.
- **Balance Flexibility and Control**: While sealed classes provide control, they also limit flexibility. Consider the trade-offs before using them extensively.
- **Leverage Pattern Matching**: Combine sealed classes with pattern matching to take full advantage of Java's type system and ensure exhaustive handling of cases.

### Common Pitfalls and How to Avoid Them

- **Overuse of Sealed Classes**: Avoid using sealed classes in situations where flexibility and extensibility are required. Overuse can lead to rigid designs that are difficult to adapt.
- **Incorrect Permits Clause**: Ensure that the `permits` clause accurately reflects all intended subclasses. Omitting a subclass can lead to compile-time errors.
- **Ignoring Subclass Requirements**: Remember that subclasses must be `final`, `sealed`, or `non-sealed`. Failing to adhere to this requirement will result in compilation errors.

### Exercises and Practice Problems

1. **Exercise 1**: Create a sealed class hierarchy to represent different types of vehicles (e.g., Car, Truck, Motorcycle). Implement a method to calculate the total number of wheels for each vehicle type using pattern matching.

2. **Exercise 2**: Modify the `Expression` example to include a new operation, `Subtract`. Update the pattern matching logic to handle this new case.

3. **Exercise 3**: Design a sealed interface for a simple game character system, with different character classes (e.g., Warrior, Mage, Archer). Implement a method to calculate the attack power based on the character class.

### Summary and Key Takeaways

Sealed classes and interfaces in Java provide a powerful mechanism for controlling class hierarchies and enhancing type safety. By explicitly specifying permitted subclasses, developers can create well-defined and maintainable designs. Sealed classes also improve pattern matching by enabling exhaustive checks, reducing the risk of runtime errors. As Java continues to evolve, sealed classes represent a step towards embracing functional programming concepts and modernizing the language.

### References and Further Reading

- [Oracle Java Documentation](https://docs.oracle.com/en/java/)
- [Java 17 Release Notes](https://openjdk.java.net/projects/jdk/17/)
- [Sealed Classes in Scala](https://docs.scala-lang.org/tour/abstract-types.html)
- [Kotlin Sealed Classes](https://kotlinlang.org/docs/sealed-classes.html)

---

## Test Your Knowledge: Sealed Classes in Java Quiz

{{< quizdown >}}

### What is the primary purpose of sealed classes in Java?

- [x] To restrict which classes can extend or implement them.
- [ ] To enhance performance by reducing class loading time.
- [ ] To simplify the syntax of class declarations.
- [ ] To allow dynamic class loading at runtime.

> **Explanation:** Sealed classes restrict which classes can extend or implement them, providing more control over the class hierarchy.

### Which keyword is used to declare a sealed class in Java?

- [x] sealed
- [ ] final
- [ ] abstract
- [ ] static

> **Explanation:** The `sealed` keyword is used to declare a sealed class in Java.

### What must be included in a sealed class declaration to specify permitted subclasses?

- [x] A permits clause
- [ ] An extends clause
- [ ] An implements clause
- [ ] A final clause

> **Explanation:** A `permits` clause must be included in a sealed class declaration to specify permitted subclasses.

### How do sealed classes improve pattern matching in Java?

- [x] By enabling exhaustive checks during pattern matching.
- [ ] By allowing dynamic pattern creation.
- [ ] By reducing the number of patterns needed.
- [ ] By simplifying the pattern syntax.

> **Explanation:** Sealed classes enable exhaustive checks during pattern matching, ensuring all possible cases are handled.

### Which of the following is NOT a valid subclass type for a sealed class?

- [ ] final
- [x] abstract
- [ ] sealed
- [ ] non-sealed

> **Explanation:** Subclasses of a sealed class must be `final`, `sealed`, or `non-sealed`. `abstract` is not a valid subclass type for a sealed class.

### What is a common use case for sealed classes?

- [x] Representing algebraic data types
- [ ] Enhancing runtime performance
- [ ] Simplifying class loading
- [ ] Allowing dynamic subclass creation

> **Explanation:** Sealed classes are commonly used for representing algebraic data types, where a fixed set of possible values is defined.

### What happens if a subclass not listed in the permits clause tries to extend a sealed class?

- [x] A compile-time error occurs.
- [ ] A runtime exception is thrown.
- [ ] The subclass is ignored.
- [ ] The subclass is automatically added to the permits clause.

> **Explanation:** A compile-time error occurs if a subclass not listed in the permits clause tries to extend a sealed class.

### How can sealed classes enhance security in Java applications?

- [x] By preventing unauthorized subclassing.
- [ ] By encrypting class files.
- [ ] By reducing memory usage.
- [ ] By allowing dynamic security checks.

> **Explanation:** Sealed classes enhance security by preventing unauthorized subclassing, reducing the risk of malicious or erroneous subclasses.

### What is the effect of using the non-sealed modifier on a subclass of a sealed class?

- [x] It allows further subclassing of the non-sealed class.
- [ ] It prevents any subclassing of the non-sealed class.
- [ ] It makes the class abstract.
- [ ] It converts the class to a sealed class.

> **Explanation:** The `non-sealed` modifier allows further subclassing of the non-sealed class, providing flexibility within the sealed hierarchy.

### True or False: Sealed classes can be used to model open-ended class hierarchies.

- [ ] True
- [x] False

> **Explanation:** False. Sealed classes are used to model closed, well-defined class hierarchies, not open-ended ones.

{{< /quizdown >}}

---
