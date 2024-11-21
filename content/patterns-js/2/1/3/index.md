---

linkTitle: "2.1.3 Factory Method"
title: "Factory Method Design Pattern in JavaScript and TypeScript"
description: "Explore the Factory Method design pattern in JavaScript and TypeScript, its components, implementation, and real-world applications."
categories:
- Design Patterns
- JavaScript
- TypeScript
tags:
- Factory Method
- Creational Patterns
- JavaScript
- TypeScript
- Design Patterns
date: 2024-10-25
type: docs
nav_weight: 213000
canonical: "https://softwarepatternslexicon.com/patterns-js/2/1/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.1.3 Factory Method

The Factory Method is a creational design pattern that provides an interface for creating objects in a superclass but allows subclasses to alter the type of objects that will be created. This pattern promotes loose coupling by eliminating the need to bind application-specific classes into your code.

### Understand the Intent

The primary intent of the Factory Method pattern is to define an interface for creating an object, but to let subclasses decide which class to instantiate. This pattern is particularly useful when a class cannot anticipate the type of objects it needs to create beforehand or when it wants to delegate the responsibility of object creation to its subclasses.

### Key Components

1. **Product Interface:** Defines the type of objects the factory method creates.
2. **Concrete Products:** Implement the product interface.
3. **Creator (Abstract Class):** Declares the factory method returning a product instance.
4. **Concrete Creators:** Override the factory method to return an instance of a concrete product.

### Implementation Steps

1. **Define a Product Interface:** Create an interface that defines the structure of the objects the factory method will create.
2. **Create Concrete Product Classes:** Implement the product interface in concrete classes.
3. **Create an Abstract Creator Class:** Define a class with a factory method that returns a product instance.
4. **Subclass the Creator:** Override the factory method in subclasses to create specific products.

### Code Examples

Let's explore how the Factory Method pattern can be implemented in JavaScript and TypeScript through a practical example.

#### JavaScript Example

Suppose we are building a document editor application that can create different types of documents (e.g., Word, PDF).

```javascript
// Product Interface
class Document {
  open() {}
}

// Concrete Products
class WordDocument extends Document {
  open() {
    console.log("Opening a Word document.");
  }
}

class PDFDocument extends Document {
  open() {
    console.log("Opening a PDF document.");
  }
}

// Creator
class DocumentCreator {
  createDocument() {
    throw new Error("This method should be overridden!");
  }
}

// Concrete Creators
class WordDocumentCreator extends DocumentCreator {
  createDocument() {
    return new WordDocument();
  }
}

class PDFDocumentCreator extends DocumentCreator {
  createDocument() {
    return new PDFDocument();
  }
}

// Usage
const wordCreator = new WordDocumentCreator();
const wordDoc = wordCreator.createDocument();
wordDoc.open(); // Output: Opening a Word document.

const pdfCreator = new PDFDocumentCreator();
const pdfDoc = pdfCreator.createDocument();
pdfDoc.open(); // Output: Opening a PDF document.
```

#### TypeScript Example

```typescript
// Product Interface
interface Document {
  open(): void;
}

// Concrete Products
class WordDocument implements Document {
  open(): void {
    console.log("Opening a Word document.");
  }
}

class PDFDocument implements Document {
  open(): void {
    console.log("Opening a PDF document.");
  }
}

// Creator
abstract class DocumentCreator {
  abstract createDocument(): Document;
}

// Concrete Creators
class WordDocumentCreator extends DocumentCreator {
  createDocument(): Document {
    return new WordDocument();
  }
}

class PDFDocumentCreator extends DocumentCreator {
  createDocument(): Document {
    return new PDFDocument();
  }
}

// Usage
const wordCreator: DocumentCreator = new WordDocumentCreator();
const wordDoc: Document = wordCreator.createDocument();
wordDoc.open(); // Output: Opening a Word document.

const pdfCreator: DocumentCreator = new PDFDocumentCreator();
const pdfDoc: Document = pdfCreator.createDocument();
pdfDoc.open(); // Output: Opening a PDF document.
```

### Use Cases

- **When a class cannot anticipate the type of objects it needs to create beforehand:** The Factory Method pattern allows the class to delegate the responsibility of object creation to its subclasses.
- **To delegate object creation to subclasses:** This pattern is beneficial when the base class wants to defer the instantiation of objects to its subclasses.

### Practice: Implement a Factory Method for a Game

Imagine a game where you need to create different types of enemy objects. You can use the Factory Method pattern to achieve this.

```typescript
// Product Interface
interface Enemy {
  attack(): void;
}

// Concrete Products
class Goblin implements Enemy {
  attack(): void {
    console.log("Goblin attacks with a club!");
  }
}

class Orc implements Enemy {
  attack(): void {
    console.log("Orc attacks with a sword!");
  }
}

// Creator
abstract class EnemyCreator {
  abstract createEnemy(): Enemy;
}

// Concrete Creators
class GoblinCreator extends EnemyCreator {
  createEnemy(): Enemy {
    return new Goblin();
  }
}

class OrcCreator extends EnemyCreator {
  createEnemy(): Enemy {
    return new Orc();
  }
}

// Usage
const goblinCreator: EnemyCreator = new GoblinCreator();
const goblin: Enemy = goblinCreator.createEnemy();
goblin.attack(); // Output: Goblin attacks with a club!

const orcCreator: EnemyCreator = new OrcCreator();
const orc: Enemy = orcCreator.createEnemy();
orc.attack(); // Output: Orc attacks with a sword!
```

### Considerations

- **Scalability:** The Factory Method pattern promotes scalability by allowing new products to be added without changing existing code.
- **Class Proliferation:** Be cautious of the potential increase in the number of classes, as each product requires a new subclass.

### Advantages and Disadvantages

#### Advantages

- **Flexibility:** The pattern provides flexibility in terms of object creation.
- **Loose Coupling:** It promotes loose coupling by separating the code that creates objects from the code that uses them.

#### Disadvantages

- **Complexity:** The pattern can introduce additional complexity due to the increased number of classes.
- **Overhead:** It may lead to unnecessary overhead if not used appropriately.

### Best Practices

- **Use Abstract Classes:** Define the creator as an abstract class to enforce the implementation of the factory method in subclasses.
- **Adhere to SOLID Principles:** Ensure that your implementation adheres to SOLID principles, particularly the Open/Closed Principle, to enhance maintainability.

### Conclusion

The Factory Method pattern is a powerful tool in the software developer's toolkit, offering a way to create objects without specifying the exact class of object that will be created. By understanding and applying this pattern, developers can build scalable, maintainable, and flexible applications.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Factory Method pattern?

- [x] To define an interface for creating an object but let subclasses decide which class to instantiate.
- [ ] To create a single instance of a class.
- [ ] To provide a way to access the elements of an aggregate object sequentially.
- [ ] To separate the construction of a complex object from its representation.

> **Explanation:** The Factory Method pattern defines an interface for creating an object but allows subclasses to decide which class to instantiate, promoting flexibility and scalability.

### Which component in the Factory Method pattern declares the factory method?

- [ ] Concrete Product
- [ ] Product Interface
- [x] Creator (Abstract Class)
- [ ] Concrete Creator

> **Explanation:** The Creator (Abstract Class) declares the factory method, which is then overridden by Concrete Creators to return instances of Concrete Products.

### What is a potential disadvantage of using the Factory Method pattern?

- [ ] It reduces flexibility.
- [x] It can lead to an increase in the number of classes.
- [ ] It tightly couples the code.
- [ ] It makes the code less maintainable.

> **Explanation:** The Factory Method pattern can lead to an increase in the number of classes, as each product requires a new subclass.

### In the Factory Method pattern, who decides which class to instantiate?

- [ ] The Product Interface
- [ ] The Concrete Product
- [x] The Concrete Creator
- [ ] The Client

> **Explanation:** The Concrete Creator decides which class to instantiate by overriding the factory method to return an instance of a specific Concrete Product.

### Which of the following is a use case for the Factory Method pattern?

- [x] When a class cannot anticipate the type of objects it needs to create beforehand.
- [ ] When a single instance of a class is needed.
- [ ] When you need to provide a way to access elements of an aggregate object sequentially.
- [ ] When you want to separate the construction of a complex object from its representation.

> **Explanation:** The Factory Method pattern is useful when a class cannot anticipate the type of objects it needs to create beforehand, allowing subclasses to decide which class to instantiate.

### How does the Factory Method pattern promote scalability?

- [ ] By reducing the number of classes.
- [x] By allowing new products to be added without changing existing code.
- [ ] By tightly coupling the code.
- [ ] By making the code less maintainable.

> **Explanation:** The Factory Method pattern promotes scalability by allowing new products to be added without changing existing code, adhering to the Open/Closed Principle.

### What is the role of the Product Interface in the Factory Method pattern?

- [x] It defines the type of objects the factory method creates.
- [ ] It declares the factory method.
- [ ] It implements the product interface.
- [ ] It overrides the factory method.

> **Explanation:** The Product Interface defines the type of objects the factory method creates, providing a common interface for all products.

### Which principle does the Factory Method pattern adhere to by allowing new products to be added without changing existing code?

- [ ] Single Responsibility Principle
- [ ] Liskov Substitution Principle
- [x] Open/Closed Principle
- [ ] Dependency Inversion Principle

> **Explanation:** The Factory Method pattern adheres to the Open/Closed Principle by allowing new products to be added without changing existing code.

### What is the role of the Concrete Product in the Factory Method pattern?

- [ ] To declare the factory method.
- [ ] To define the type of objects the factory method creates.
- [x] To implement the product interface.
- [ ] To override the factory method.

> **Explanation:** The Concrete Product implements the product interface, providing specific implementations for the product.

### True or False: The Factory Method pattern tightly couples the code.

- [ ] True
- [x] False

> **Explanation:** False. The Factory Method pattern promotes loose coupling by separating the code that creates objects from the code that uses them.

{{< /quizdown >}}
