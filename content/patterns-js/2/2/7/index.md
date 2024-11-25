---
linkTitle: "2.2.7 Proxy Design Pattern in JavaScript and TypeScript"
title: "Proxy Design Pattern: Control Access with JavaScript and TypeScript"
description: "Explore the Proxy Design Pattern in JavaScript and TypeScript, its intent, components, types, and practical implementation with code examples."
categories:
- Design Patterns
- JavaScript
- TypeScript
tags:
- Proxy Pattern
- Structural Patterns
- JavaScript
- TypeScript
- Design Patterns
date: 2024-10-25
type: docs
nav_weight: 227000
canonical: "https://softwarepatternslexicon.com/patterns-js/2/2/7"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.2.7 Proxy Design Pattern

### Introduction

The Proxy Design Pattern is a structural pattern that provides a surrogate or placeholder for another object to control access to it. This pattern is particularly useful when you need to add an additional layer of control over the access to an object, such as lazy initialization, access control, logging, or caching.

### Understand the Intent

The primary intent of the Proxy Design Pattern is to control access to an object. By using a proxy, you can add additional functionality when accessing an object without changing the object's code. This can include tasks such as lazy-loading, access control, logging, and more.

### Key Components

- **Subject Interface:** This declares the common interface for both the RealSubject and the Proxy. It ensures that the Proxy can be used anywhere the RealSubject is expected.
  
- **RealSubject:** The actual object that the proxy represents. This is where the real work happens.

- **Proxy:** This controls access to the RealSubject. It can add additional functionality such as access control, lazy initialization, or logging.

### Implementation Steps

1. **Define the Subject Interface:** Create an interface that both the RealSubject and Proxy will implement.
   
2. **Implement the RealSubject:** Develop the actual functionality within the RealSubject class.

3. **Implement the Proxy Class:** Create the Proxy class that controls access to the RealSubject. This class will implement the Subject interface and contain a reference to the RealSubject.

### Types of Proxies

- **Virtual Proxy:** This type of proxy delays the creation and initialization of the RealSubject until it is actually needed. This is useful for resource-intensive objects.

- **Protection Proxy:** This proxy controls access to the RealSubject based on permissions. It can restrict access to certain users or roles.

- **Remote Proxy:** This represents an object that resides in a different address space, such as on a different server or in a different process.

### Code Examples

Let's explore how to implement a Proxy Design Pattern in JavaScript/TypeScript with examples of a Virtual Proxy and a Protection Proxy.

#### Virtual Proxy Example

A Virtual Proxy can be used to lazy-load a heavy resource, such as an image.

```typescript
interface Image {
  display(): void;
}

class RealImage implements Image {
  private filename: string;

  constructor(filename: string) {
    this.filename = filename;
    this.loadFromDisk();
  }

  private loadFromDisk() {
    console.log(`Loading ${this.filename}`);
  }

  display() {
    console.log(`Displaying ${this.filename}`);
  }
}

class ProxyImage implements Image {
  private realImage: RealImage | null = null;
  private filename: string;

  constructor(filename: string) {
    this.filename = filename;
  }

  display() {
    if (this.realImage === null) {
      this.realImage = new RealImage(this.filename);
    }
    this.realImage.display();
  }
}

// Usage
const image = new ProxyImage("test.jpg");
image.display(); // Loading test.jpg
image.display(); // Displaying test.jpg
```

#### Protection Proxy Example

A Protection Proxy can restrict access to certain methods based on user roles.

```typescript
interface Document {
  display(): void;
  edit(): void;
}

class RealDocument implements Document {
  display() {
    console.log("Displaying document");
  }

  edit() {
    console.log("Editing document");
  }
}

class DocumentProxy implements Document {
  private realDocument: RealDocument;
  private userRole: string;

  constructor(userRole: string) {
    this.realDocument = new RealDocument();
    this.userRole = userRole;
  }

  display() {
    this.realDocument.display();
  }

  edit() {
    if (this.userRole === "Admin") {
      this.realDocument.edit();
    } else {
      console.log("Access denied: You do not have permission to edit this document.");
    }
  }
}

// Usage
const adminDocument = new DocumentProxy("Admin");
adminDocument.display(); // Displaying document
adminDocument.edit(); // Editing document

const guestDocument = new DocumentProxy("Guest");
guestDocument.display(); // Displaying document
guestDocument.edit(); // Access denied
```

### Use Cases

- **Control Access:** When you need to control access to an object, such as restricting access based on user roles.
  
- **Lazy Initialization:** To delay the creation and initialization of a resource-intensive object until it is needed.

- **Logging and Caching:** To add logging or caching functionality when accessing an object.

### Practice

Try creating a proxy for an image viewer that loads images on demand. This will help you understand how to implement a Virtual Proxy in a real-world scenario.

### Considerations

- **Interface Matching:** Ensure that the proxy interface matches the real subject's interface to maintain compatibility.

- **Complexity and Performance:** Be cautious of added complexity and potential performance overhead introduced by the proxy.

### Advantages and Disadvantages

#### Advantages

- **Controlled Access:** Provides a controlled way to access an object, which is useful for security and resource management.
  
- **Lazy Initialization:** Helps in deferring the creation of expensive objects until they are needed.

- **Additional Functionality:** Allows adding additional functionality like logging, caching, or access control without modifying the original object.

#### Disadvantages

- **Increased Complexity:** Introduces an additional layer of abstraction, which can increase the complexity of the code.

- **Performance Overhead:** May introduce performance overhead due to the additional layer of indirection.

### Best Practices

- Use proxies when you need to add additional control or functionality to an object without modifying its code.
  
- Ensure that the proxy and real subject share a common interface to maintain compatibility.

- Be mindful of the performance implications when using proxies, especially in performance-critical applications.

### Comparisons

The Proxy pattern is often compared with the Decorator pattern. While both patterns add functionality to an object, the Proxy pattern focuses on controlling access, whereas the Decorator pattern focuses on adding behavior.

### Conclusion

The Proxy Design Pattern is a powerful tool for controlling access to objects in JavaScript and TypeScript applications. By understanding its components, types, and implementation, you can effectively use proxies to manage access, lazy-load resources, and add additional functionality to objects.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Proxy Design Pattern?

- [x] To provide a surrogate or placeholder for another object to control access to it.
- [ ] To add new functionality to an existing object without altering its structure.
- [ ] To allow an object to alter its behavior when its internal state changes.
- [ ] To define a family of algorithms and make them interchangeable.

> **Explanation:** The Proxy Design Pattern provides a surrogate or placeholder for another object to control access to it.

### Which of the following is NOT a type of Proxy?

- [ ] Virtual Proxy
- [ ] Protection Proxy
- [ ] Remote Proxy
- [x] Observer Proxy

> **Explanation:** Observer Proxy is not a recognized type of Proxy. The common types are Virtual, Protection, and Remote Proxies.

### In the Proxy Design Pattern, what is the role of the RealSubject?

- [x] It is the actual object that the proxy represents and where the real work happens.
- [ ] It declares the common interface for both the RealSubject and the Proxy.
- [ ] It controls access to the RealSubject.
- [ ] It delays the creation and initialization of the RealSubject.

> **Explanation:** The RealSubject is the actual object that the proxy represents and where the real work happens.

### What is a Virtual Proxy used for?

- [x] Delaying the creation and initialization of a resource-intensive object until it is needed.
- [ ] Controlling access to an object based on user permissions.
- [ ] Representing an object in a different address space.
- [ ] Adding new functionality to an existing object.

> **Explanation:** A Virtual Proxy delays the creation and initialization of a resource-intensive object until it is needed.

### How does a Protection Proxy control access?

- [x] By restricting access to certain methods based on user roles or permissions.
- [ ] By delaying the creation of the RealSubject.
- [ ] By representing an object in a different address space.
- [ ] By adding logging functionality.

> **Explanation:** A Protection Proxy controls access by restricting access to certain methods based on user roles or permissions.

### What is a common disadvantage of using the Proxy Design Pattern?

- [x] It can introduce performance overhead due to the additional layer of indirection.
- [ ] It makes the code less flexible and harder to maintain.
- [ ] It cannot be used with interfaces.
- [ ] It requires modifying the original object.

> **Explanation:** A common disadvantage of the Proxy Design Pattern is that it can introduce performance overhead due to the additional layer of indirection.

### Which pattern is often compared with the Proxy pattern?

- [x] Decorator Pattern
- [ ] Singleton Pattern
- [ ] Observer Pattern
- [ ] Strategy Pattern

> **Explanation:** The Proxy pattern is often compared with the Decorator pattern, as both add functionality to an object.

### What should you ensure when implementing a Proxy?

- [x] The proxy interface matches the real subject's interface.
- [ ] The proxy has more methods than the real subject.
- [ ] The proxy and real subject are in the same class.
- [ ] The proxy does not implement any interface.

> **Explanation:** When implementing a Proxy, ensure that the proxy interface matches the real subject's interface to maintain compatibility.

### Which of the following is a use case for the Proxy Design Pattern?

- [x] Adding logging or caching functionality when accessing an object.
- [ ] Creating a single instance of a class.
- [ ] Observing changes in an object's state.
- [ ] Defining a family of algorithms.

> **Explanation:** A use case for the Proxy Design Pattern is adding logging or caching functionality when accessing an object.

### True or False: The Proxy Design Pattern can be used to represent an object in a different address space.

- [x] True
- [ ] False

> **Explanation:** True. A Remote Proxy can be used to represent an object in a different address space.

{{< /quizdown >}}
