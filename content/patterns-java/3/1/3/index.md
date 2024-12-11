---
canonical: "https://softwarepatternslexicon.com/patterns-java/3/1/3"
title: "Liskov Substitution Principle (LSP) in Java Design Patterns"
description: "Explore the Liskov Substitution Principle (LSP) in Java, its role in polymorphism, and how it ensures reliable inheritance hierarchies. Learn through examples and design patterns like Template Method and Factory Method."
linkTitle: "3.1.3 Liskov Substitution Principle (LSP)"
tags:
- "Java"
- "Design Patterns"
- "Liskov Substitution Principle"
- "SOLID Principles"
- "Polymorphism"
- "Inheritance"
- "Template Method"
- "Factory Method"
date: 2024-11-25
type: docs
nav_weight: 31300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 3.1.3 Liskov Substitution Principle (LSP)

The Liskov Substitution Principle (LSP) is a fundamental concept in object-oriented design, forming the 'L' in the SOLID principles. It was introduced by Barbara Liskov in 1987 and is crucial for ensuring that a system's architecture remains robust and maintainable. The principle states that objects of a superclass should be replaceable with objects of a subclass without affecting the correctness of the program. This principle is essential for achieving polymorphism and reliable inheritance hierarchies in Java.

### Understanding Liskov Substitution Principle

#### Definition and Role in Polymorphism

The Liskov Substitution Principle can be formally defined as follows:

> If `S` is a subtype of `T`, then objects of type `T` may be replaced with objects of type `S` without altering any of the desirable properties of the program (correctness, task performed, etc.).

This principle is integral to polymorphism, a core concept in object-oriented programming that allows objects to be treated as instances of their parent class. By adhering to LSP, developers ensure that subclasses can stand in for their parent classes seamlessly, allowing for flexible and reusable code.

#### Violations of LSP and Their Consequences

Violating LSP can lead to unexpected behavior and bugs. Consider a scenario where a subclass overrides a method in a way that changes the expected behavior of the superclass. This can cause issues when the subclass is used in place of the superclass, leading to incorrect program logic.

**Example of LSP Violation:**

```java
class Rectangle {
    private int width;
    private int height;

    public void setWidth(int width) {
        this.width = width;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getArea() {
        return width * height;
    }
}

class Square extends Rectangle {
    @Override
    public void setWidth(int width) {
        super.setWidth(width);
        super.setHeight(width);
    }

    @Override
    public void setHeight(int height) {
        super.setWidth(height);
        super.setHeight(height);
    }
}

// Client code
Rectangle rect = new Square();
rect.setWidth(5);
rect.setHeight(10);
System.out.println(rect.getArea()); // Expected: 50, Actual: 100
```

In this example, the `Square` class violates LSP by altering the behavior of `setWidth` and `setHeight` methods. The client code expects a `Rectangle` behavior, but the `Square` class changes it, leading to incorrect results.

### Rules for Creating Subclasses that Adhere to LSP

To ensure that subclasses adhere to LSP, follow these guidelines:

1. **Method Signature Compatibility**: Subclasses should not change the method signatures of the superclass.
2. **Behavioral Consistency**: Subclasses should maintain the expected behavior of the superclass methods.
3. **Preconditions and Postconditions**: Subclasses should not strengthen preconditions or weaken postconditions.
4. **Invariant Preservation**: Subclasses should maintain the invariants established by the superclass.
5. **Exception Handling**: Subclasses should not introduce new exceptions that are not present in the superclass.

### Supporting Reliable Inheritance Hierarchies

By adhering to LSP, developers can create reliable inheritance hierarchies that facilitate code reuse and maintainability. LSP ensures that subclasses can be used interchangeably with their parent classes, promoting a flexible and extensible system architecture.

### Design Patterns and LSP

Several design patterns inherently support LSP by promoting a clear separation of concerns and ensuring that subclasses adhere to the expected behavior of their parent classes.

#### Template Method Pattern

The Template Method pattern defines the skeleton of an algorithm in a superclass, allowing subclasses to override specific steps without changing the algorithm's structure. This pattern supports LSP by ensuring that subclasses can modify behavior without altering the overall algorithm.

**Example of Template Method Pattern:**

```java
abstract class DataProcessor {
    public final void process() {
        readData();
        processData();
        writeData();
    }

    abstract void readData();
    abstract void processData();
    abstract void writeData();
}

class CSVDataProcessor extends DataProcessor {
    @Override
    void readData() {
        System.out.println("Reading CSV data");
    }

    @Override
    void processData() {
        System.out.println("Processing CSV data");
    }

    @Override
    void writeData() {
        System.out.println("Writing CSV data");
    }
}

// Client code
DataProcessor processor = new CSVDataProcessor();
processor.process();
```

In this example, the `CSVDataProcessor` class adheres to LSP by implementing the abstract methods of the `DataProcessor` class without altering the `process` method's structure.

#### Factory Method Pattern

The Factory Method pattern provides an interface for creating objects in a superclass but allows subclasses to alter the type of objects that will be created. This pattern supports LSP by ensuring that object creation is consistent with the expected behavior of the superclass.

**Example of Factory Method Pattern:**

```java
abstract class Document {
    public abstract void open();
}

class WordDocument extends Document {
    @Override
    public void open() {
        System.out.println("Opening Word document");
    }
}

class DocumentFactory {
    public static Document createDocument(String type) {
        if (type.equals("Word")) {
            return new WordDocument();
        }
        // Add more document types as needed
        return null;
    }
}

// Client code
Document doc = DocumentFactory.createDocument("Word");
doc.open();
```

In this example, the `WordDocument` class adheres to LSP by implementing the `open` method defined in the `Document` class, ensuring consistent behavior across different document types.

### Conclusion

The Liskov Substitution Principle is a cornerstone of object-oriented design, ensuring that subclasses can be used interchangeably with their parent classes without altering the program's correctness. By adhering to LSP, developers can create flexible, maintainable, and robust systems that leverage polymorphism and reliable inheritance hierarchies. Design patterns like Template Method and Factory Method inherently support LSP by promoting consistent behavior across subclasses.

### Key Takeaways

- LSP ensures that subclasses can replace their parent classes without affecting program correctness.
- Violating LSP can lead to unexpected behavior and bugs.
- Adhering to LSP promotes reliable inheritance hierarchies and facilitates code reuse.
- Design patterns like Template Method and Factory Method support LSP by ensuring consistent behavior across subclasses.

### Exercises

1. Refactor the `Rectangle` and `Square` example to adhere to LSP.
2. Implement a new subclass using the Template Method pattern and ensure it adheres to LSP.
3. Create a new document type using the Factory Method pattern and verify its adherence to LSP.

### Further Reading

- [Java Documentation](https://docs.oracle.com/en/java/)
- [Cloud Design Patterns](https://learn.microsoft.com/en-us/azure/architecture/patterns/)

## Test Your Knowledge: Liskov Substitution Principle in Java Quiz

{{< quizdown >}}

### What is the primary goal of the Liskov Substitution Principle?

- [x] To ensure that subclasses can replace their parent classes without affecting program correctness.
- [ ] To allow subclasses to override any method of the parent class.
- [ ] To enforce strict type checking in Java.
- [ ] To prevent inheritance in object-oriented programming.

> **Explanation:** The Liskov Substitution Principle ensures that subclasses can replace their parent classes without affecting the correctness of the program.

### Which of the following is a violation of LSP?

- [x] A subclass changes the expected behavior of a method in the superclass.
- [ ] A subclass adds new methods not present in the superclass.
- [ ] A subclass implements an interface method.
- [ ] A subclass inherits fields from the superclass.

> **Explanation:** Violating LSP occurs when a subclass changes the expected behavior of a method in the superclass, leading to incorrect program logic.

### How does the Template Method pattern support LSP?

- [x] By defining the skeleton of an algorithm in a superclass and allowing subclasses to override specific steps.
- [ ] By allowing subclasses to change the entire algorithm structure.
- [ ] By enforcing strict method signatures in subclasses.
- [ ] By preventing subclasses from overriding methods.

> **Explanation:** The Template Method pattern supports LSP by defining the skeleton of an algorithm in a superclass and allowing subclasses to override specific steps without changing the algorithm's structure.

### What is a key rule for creating subclasses that adhere to LSP?

- [x] Subclasses should not strengthen preconditions or weaken postconditions.
- [ ] Subclasses should override all methods of the superclass.
- [ ] Subclasses should not implement any new methods.
- [ ] Subclasses should always use the same method names as the superclass.

> **Explanation:** A key rule for adhering to LSP is that subclasses should not strengthen preconditions or weaken postconditions, ensuring consistent behavior.

### Which design pattern inherently supports LSP by providing an interface for creating objects?

- [x] Factory Method
- [ ] Singleton
- [ ] Observer
- [ ] Decorator

> **Explanation:** The Factory Method pattern inherently supports LSP by providing an interface for creating objects, ensuring consistent behavior across different object types.

### What is the consequence of violating LSP in a program?

- [x] It can lead to unexpected behavior and bugs.
- [ ] It improves program performance.
- [ ] It simplifies code maintenance.
- [ ] It enhances code readability.

> **Explanation:** Violating LSP can lead to unexpected behavior and bugs, as it disrupts the expected behavior of subclasses replacing their parent classes.

### How can subclasses adhere to LSP in terms of exception handling?

- [x] Subclasses should not introduce new exceptions that are not present in the superclass.
- [ ] Subclasses should catch all exceptions.
- [ ] Subclasses should throw more exceptions than the superclass.
- [ ] Subclasses should ignore exceptions thrown by the superclass.

> **Explanation:** To adhere to LSP, subclasses should not introduce new exceptions that are not present in the superclass, ensuring consistent behavior.

### What is the role of polymorphism in LSP?

- [x] It allows objects to be treated as instances of their parent class.
- [ ] It enforces strict type checking.
- [ ] It prevents inheritance in object-oriented programming.
- [ ] It allows subclasses to override any method of the parent class.

> **Explanation:** Polymorphism plays a crucial role in LSP by allowing objects to be treated as instances of their parent class, facilitating flexible and reusable code.

### Which of the following is NOT a guideline for adhering to LSP?

- [x] Subclasses should override all methods of the superclass.
- [ ] Subclasses should maintain the expected behavior of the superclass methods.
- [ ] Subclasses should not strengthen preconditions or weaken postconditions.
- [ ] Subclasses should maintain the invariants established by the superclass.

> **Explanation:** Adhering to LSP does not require subclasses to override all methods of the superclass; instead, they should maintain expected behavior and invariants.

### True or False: The Liskov Substitution Principle is only applicable to Java.

- [ ] True
- [x] False

> **Explanation:** The Liskov Substitution Principle is applicable to all object-oriented programming languages, not just Java.

{{< /quizdown >}}
