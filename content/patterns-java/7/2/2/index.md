---
canonical: "https://softwarepatternslexicon.com/patterns-java/7/2/2"
title: "Class Adapter vs. Object Adapter: Understanding Java Design Patterns"
description: "Explore the differences between class adapters and object adapters in Java, comparing their implementation, use cases, and limitations."
linkTitle: "7.2.2 Class Adapter vs. Object Adapter"
tags:
- "Java"
- "Design Patterns"
- "Adapter Pattern"
- "Class Adapter"
- "Object Adapter"
- "Inheritance"
- "Composition"
- "Software Architecture"
date: 2024-11-25
type: docs
nav_weight: 72200
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 7.2.2 Class Adapter vs. Object Adapter

In the realm of software design patterns, the Adapter Pattern stands out as a crucial tool for achieving compatibility between incompatible interfaces. This section delves into the nuances of the Adapter Pattern, specifically focusing on the two primary forms: **Class Adapter** and **Object Adapter**. Understanding these concepts is vital for Java developers and software architects aiming to create flexible and maintainable codebases.

### Introduction to Adapter Pattern

The Adapter Pattern is a structural design pattern that allows objects with incompatible interfaces to collaborate. It acts as a bridge between two incompatible interfaces, enabling them to work together without modifying their existing code. This pattern is particularly useful when integrating legacy systems with new functionalities or when using third-party libraries that do not match the expected interface.

### Class Adapter

#### Definition

A **Class Adapter** uses inheritance to adapt one interface to another. It involves creating a new class that inherits from both the target interface and the adaptee class. This approach leverages Java's multiple interface inheritance capability to achieve the desired adaptation.

#### Implementation

In a class adapter, the adapter class extends the adaptee class and implements the target interface. This allows the adapter to inherit the behavior of the adaptee while providing the interface expected by the client.

```java
// Target interface
interface Target {
    void request();
}

// Adaptee class
class Adaptee {
    void specificRequest() {
        System.out.println("Adaptee's specific request.");
    }
}

// Class Adapter
class ClassAdapter extends Adaptee implements Target {
    @Override
    public void request() {
        // Adapting the specific request to the target interface
        specificRequest();
    }
}

// Client code
public class Client {
    public static void main(String[] args) {
        Target target = new ClassAdapter();
        target.request(); // Output: Adaptee's specific request.
    }
}
```

#### Advantages and Limitations

- **Advantages**:
  - **Simplicity**: The class adapter is straightforward to implement when the adaptee class is not complex.
  - **Performance**: Since it uses inheritance, there is no additional overhead of object composition.

- **Limitations**:
  - **Single Inheritance Constraint**: Java's single inheritance model restricts the class adapter to extend only one class, which can be a significant limitation if the adapter needs to inherit from multiple classes.
  - **Tight Coupling**: The adapter is tightly coupled to the adaptee class, making it less flexible if the adaptee's implementation changes.

### Object Adapter

#### Definition

An **Object Adapter** uses composition to achieve the adaptation. Instead of inheriting from the adaptee, the adapter holds a reference to an instance of the adaptee class and delegates calls to it.

#### Implementation

In an object adapter, the adapter class implements the target interface and contains an instance of the adaptee class. This allows the adapter to forward requests to the adaptee instance.

```java
// Target interface
interface Target {
    void request();
}

// Adaptee class
class Adaptee {
    void specificRequest() {
        System.out.println("Adaptee's specific request.");
    }
}

// Object Adapter
class ObjectAdapter implements Target {
    private Adaptee adaptee;

    public ObjectAdapter(Adaptee adaptee) {
        this.adaptee = adaptee;
    }

    @Override
    public void request() {
        // Delegating the request to the adaptee's specific request
        adaptee.specificRequest();
    }
}

// Client code
public class Client {
    public static void main(String[] args) {
        Adaptee adaptee = new Adaptee();
        Target target = new ObjectAdapter(adaptee);
        target.request(); // Output: Adaptee's specific request.
    }
}
```

#### Advantages and Limitations

- **Advantages**:
  - **Flexibility**: The object adapter is more flexible as it can work with any subclass of the adaptee class.
  - **Loose Coupling**: It promotes loose coupling between the adapter and the adaptee, making it easier to modify or extend the adaptee's behavior.

- **Limitations**:
  - **Complexity**: The object adapter can be more complex to implement, especially if the adaptee has a large number of methods.
  - **Performance Overhead**: There is a slight performance overhead due to the additional level of indirection introduced by composition.

### Comparison: Class Adapter vs. Object Adapter

| Aspect               | Class Adapter                          | Object Adapter                         |
|----------------------|----------------------------------------|----------------------------------------|
| **Inheritance**      | Uses inheritance                       | Uses composition                       |
| **Flexibility**      | Less flexible due to single inheritance| More flexible, can adapt multiple adaptees |
| **Coupling**         | Tightly coupled to adaptee             | Loosely coupled, easier to modify      |
| **Complexity**       | Simpler to implement                   | More complex due to composition        |
| **Performance**      | Slightly better due to direct inheritance | Slight overhead due to delegation     |

### Historical Context and Evolution

The Adapter Pattern has its roots in the early days of object-oriented programming, where the need to integrate disparate systems became apparent. As software systems grew in complexity, the demand for patterns that could facilitate integration without extensive rework increased. The Adapter Pattern emerged as a solution to this challenge, providing a way to reconcile incompatible interfaces.

Over time, the pattern has evolved to accommodate modern programming paradigms, such as dependency injection and interface-based design. The distinction between class and object adapters reflects the broader shift towards composition over inheritance, a principle that has gained traction in contemporary software design.

### Practical Applications

- **Legacy System Integration**: Adapters are commonly used to integrate legacy systems with new applications, allowing old and new components to interact seamlessly.
- **Third-Party Library Integration**: When using third-party libraries that do not match the expected interface, adapters can bridge the gap without modifying the library code.
- **Testing and Mocking**: Adapters can facilitate testing by allowing mock implementations to be easily swapped in place of real components.

### Choosing Between Class and Object Adapter

The choice between class and object adapters depends on several factors:

- **Inheritance Constraints**: If the adaptee class is already part of an inheritance hierarchy, an object adapter is more suitable due to Java's single inheritance limitation.
- **Flexibility Requirements**: If flexibility and loose coupling are priorities, an object adapter is preferable.
- **Performance Considerations**: In performance-critical applications, a class adapter may be advantageous due to its direct method invocation.

### Conclusion

Both class and object adapters play a vital role in the Adapter Pattern, each offering distinct advantages and trade-offs. By understanding their differences and applications, Java developers and software architects can make informed decisions about which approach best suits their needs. As with any design pattern, the key is to balance flexibility, complexity, and performance to achieve the desired outcome.

### Further Reading

- [Oracle Java Documentation](https://docs.oracle.com/en/java/)
- [Design Patterns: Elements of Reusable Object-Oriented Software](https://en.wikipedia.org/wiki/Design_Patterns)

## Test Your Knowledge: Class Adapter vs. Object Adapter Quiz

{{< quizdown >}}

### What is the primary difference between a class adapter and an object adapter?

- [x] A class adapter uses inheritance, while an object adapter uses composition.
- [ ] A class adapter uses composition, while an object adapter uses inheritance.
- [ ] Both use inheritance.
- [ ] Both use composition.

> **Explanation:** A class adapter uses inheritance to adapt interfaces, whereas an object adapter uses composition.

### Which adapter type is more flexible in Java?

- [ ] Class Adapter
- [x] Object Adapter
- [ ] Both are equally flexible
- [ ] Neither is flexible

> **Explanation:** Object adapters are more flexible because they use composition, allowing them to work with any subclass of the adaptee.

### What is a limitation of class adapters in Java?

- [x] They are constrained by Java's single inheritance model.
- [ ] They cannot be used with interfaces.
- [ ] They are always slower than object adapters.
- [ ] They require more memory.

> **Explanation:** Class adapters are limited by Java's single inheritance model, which restricts them to extending only one class.

### In which scenario would an object adapter be more suitable than a class adapter?

- [x] When the adaptee is part of an existing inheritance hierarchy.
- [ ] When performance is the primary concern.
- [ ] When the adaptee has only one method.
- [ ] When the target interface is very simple.

> **Explanation:** An object adapter is more suitable when the adaptee is part of an existing inheritance hierarchy, as it avoids the single inheritance limitation.

### What is a common use case for adapters?

- [x] Integrating legacy systems with new applications.
- [ ] Improving application performance.
- [ ] Reducing memory usage.
- [ ] Simplifying user interfaces.

> **Explanation:** Adapters are often used to integrate legacy systems with new applications by bridging incompatible interfaces.

### Which adapter type promotes loose coupling?

- [ ] Class Adapter
- [x] Object Adapter
- [ ] Both promote loose coupling equally
- [ ] Neither promotes loose coupling

> **Explanation:** Object adapters promote loose coupling because they use composition, allowing for easier modification and extension of the adaptee's behavior.

### What is a potential drawback of using object adapters?

- [x] They can introduce a slight performance overhead due to delegation.
- [ ] They cannot adapt multiple adaptees.
- [ ] They are tightly coupled to the adaptee.
- [ ] They are limited by Java's single inheritance model.

> **Explanation:** Object adapters can introduce a slight performance overhead due to the additional level of indirection introduced by composition.

### Which adapter type is simpler to implement?

- [x] Class Adapter
- [ ] Object Adapter
- [ ] Both are equally simple
- [ ] Neither is simple

> **Explanation:** Class adapters are generally simpler to implement because they use direct inheritance, avoiding the need for composition.

### How does the Adapter Pattern facilitate testing?

- [x] By allowing mock implementations to be easily swapped in place of real components.
- [ ] By reducing the number of test cases needed.
- [ ] By improving test execution speed.
- [ ] By eliminating the need for test doubles.

> **Explanation:** The Adapter Pattern facilitates testing by allowing mock implementations to be easily swapped in place of real components, enabling more flexible and isolated tests.

### True or False: Class adapters can adapt multiple adaptees simultaneously.

- [ ] True
- [x] False

> **Explanation:** Class adapters cannot adapt multiple adaptees simultaneously due to Java's single inheritance constraint, which limits them to extending only one class.

{{< /quizdown >}}
