---
linkTitle: "9.4 Decorators"
title: "Mastering Decorators in TypeScript: Enhance Your Code with Annotations and Metadata"
description: "Explore the power of decorators in TypeScript to add annotations and metadata to classes and class members, enabling code modification without altering the original code."
categories:
- Design Patterns
- TypeScript
- JavaScript
tags:
- Decorators
- TypeScript
- Design Patterns
- Code Enhancement
- Metadata
date: 2024-10-25
type: docs
nav_weight: 940000
canonical: "https://softwarepatternslexicon.com/patterns-js/9/4"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 9.4 Decorators

Decorators in TypeScript provide a powerful way to add annotations and metadata to classes and class members at design time. They enable developers to modify existing code behavior without altering the original code, making them a valuable tool for enhancing functionality and maintaining clean code. In this article, we'll delve into the concept of decorators, their implementation, and practical use cases.

### Understanding the Concept of Decorators

Decorators are a form of syntactic sugar in TypeScript that allow you to attach metadata to classes, methods, properties, or parameters. They are inspired by a similar feature in languages like Python and Java and are part of the ECMAScript proposal. Decorators can be used to:

- **Add Annotations:** Provide additional information about a class or its members.
- **Modify Behavior:** Change how methods or classes behave without modifying their actual code.
- **Enhance Functionality:** Implement cross-cutting concerns like logging, security, or caching.

### Implementation Steps

#### 1. Enable Decorators

Before you can use decorators in TypeScript, you need to enable them in your project. This is done by setting the `"experimentalDecorators": true` flag in your `tsconfig.json` file:

```json
{
  "compilerOptions": {
    "experimentalDecorators": true
  }
}
```

#### 2. Create Decorator Functions

A decorator is essentially a function that takes a target and optionally a property key and descriptor. Here's a simple example of a method decorator:

```typescript
function Log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log(`Calling ${propertyKey} with arguments: ${JSON.stringify(args)}`);
    return originalMethod.apply(this, args);
  };
  return descriptor;
}
```

#### 3. Apply Decorators

Decorators are applied using the `@` symbol. You can apply them to classes, methods, properties, or parameters:

```typescript
class Example {
  @Log
  greet(name: string) {
    return `Hello, ${name}!`;
  }
}

const example = new Example();
example.greet('World'); // Logs: Calling greet with arguments: ["World"]
```

### Code Examples

#### Implementing a `@Log` Method Decorator

The `@Log` decorator logs method calls and their parameters, providing insight into method usage without altering the method's core logic.

```typescript
function Log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log(`Calling ${propertyKey} with arguments: ${JSON.stringify(args)}`);
    return originalMethod.apply(this, args);
  };
  return descriptor;
}

class Calculator {
  @Log
  add(a: number, b: number): number {
    return a + b;
  }
}

const calculator = new Calculator();
calculator.add(2, 3); // Logs: Calling add with arguments: [2,3]
```

#### Creating a `@Singleton` Class Decorator

The `@Singleton` decorator ensures that only one instance of a class is created, enforcing the Singleton design pattern.

```typescript
function Singleton<T extends { new (...args: any[]): {} }>(constructor: T) {
  return class extends constructor {
    private static instance: T;
    constructor(...args: any[]) {
      if (!Singleton.instance) {
        super(...args);
        Singleton.instance = this;
      }
      return Singleton.instance;
    }
  };
}

@Singleton
class Database {
  connect() {
    console.log('Connected to the database.');
  }
}

const db1 = new Database();
const db2 = new Database();
console.log(db1 === db2); // true
```

### Use Cases for Decorators

Decorators are versatile and can be used in various scenarios, such as:

- **Aspect-Oriented Programming:** Implement cross-cutting concerns like logging, security, or caching.
- **Dependency Injection:** Automatically inject dependencies into classes.
- **Validation:** Validate method parameters or class properties.
- **Authorization:** Check user permissions before executing a method.

### Practice: Writing a `@Debounce` Decorator

A `@Debounce` decorator can be used to prevent a method from being called too frequently, which is useful in scenarios like handling button clicks or search input.

```typescript
function Debounce(delay: number) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    let timeout: NodeJS.Timeout;
    descriptor.value = function (...args: any[]) {
      clearTimeout(timeout);
      timeout = setTimeout(() => originalMethod.apply(this, args), delay);
    };
    return descriptor;
  };
}

class Search {
  @Debounce(300)
  search(query: string) {
    console.log(`Searching for ${query}`);
  }
}

const search = new Search();
search.search('TypeScript');
search.search('Decorators');
search.search('Patterns'); // Only the last call will be executed after 300ms
```

### Considerations

While decorators offer many benefits, there are some considerations to keep in mind:

- **Experimental Feature:** Decorators are an experimental feature in TypeScript and may change in future versions.
- **Readability:** Overusing decorators can make code harder to read and maintain. Use them judiciously.
- **Performance:** Be aware of the potential performance impact, especially when using decorators that modify method behavior.

### Conclusion

Decorators in TypeScript provide a powerful mechanism for enhancing and modifying code behavior without altering the original implementation. By understanding how to implement and apply decorators, you can leverage their capabilities to improve code maintainability and functionality. Whether you're implementing logging, enforcing design patterns, or adding metadata, decorators offer a flexible and elegant solution.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of decorators in TypeScript?

- [x] To add annotations and metadata to classes and class members
- [ ] To compile TypeScript code into JavaScript
- [ ] To manage package dependencies
- [ ] To perform runtime type checking

> **Explanation:** Decorators are used to add annotations and metadata to classes and class members, enabling modification of behavior without altering the original code.

### How do you enable decorators in a TypeScript project?

- [x] Set `"experimentalDecorators": true` in `tsconfig.json`
- [ ] Install a specific TypeScript plugin
- [ ] Use a special command-line flag during compilation
- [ ] Decorators are enabled by default in TypeScript

> **Explanation:** To use decorators, you must set `"experimentalDecorators": true` in your `tsconfig.json` file.

### Which symbol is used to apply decorators in TypeScript?

- [x] @
- [ ] #
- [ ] $
- [ ] %

> **Explanation:** The `@` symbol is used to apply decorators to classes, methods, properties, or parameters.

### What does the `@Log` decorator do in the provided example?

- [x] Logs method calls and their parameters
- [ ] Prevents a method from being called
- [ ] Ensures only one instance of a class is created
- [ ] Validates method parameters

> **Explanation:** The `@Log` decorator logs method calls and their parameters, providing insight into method usage.

### What is a potential drawback of using decorators?

- [x] They can make code harder to read and maintain
- [ ] They are not supported in TypeScript
- [ ] They increase the size of the compiled JavaScript
- [ ] They require additional runtime libraries

> **Explanation:** Overusing decorators can make code harder to read and maintain, so they should be used judiciously.

### What pattern does the `@Singleton` decorator enforce?

- [x] Singleton
- [ ] Factory
- [ ] Observer
- [ ] Strategy

> **Explanation:** The `@Singleton` decorator ensures that only one instance of a class is created, enforcing the Singleton design pattern.

### In which programming paradigm are decorators particularly useful?

- [x] Aspect-Oriented Programming
- [ ] Functional Programming
- [ ] Procedural Programming
- [ ] Object-Oriented Programming

> **Explanation:** Decorators are particularly useful in Aspect-Oriented Programming for implementing cross-cutting concerns like logging and security.

### What is the purpose of the `@Debounce` decorator in the example?

- [x] To prevent a method from being called too frequently
- [ ] To log method calls
- [ ] To validate method parameters
- [ ] To ensure a method is only called once

> **Explanation:** The `@Debounce` decorator prevents a method from being called too frequently, which is useful in scenarios like handling button clicks or search input.

### Are decorators a stable feature in TypeScript?

- [ ] Yes, they are stable and will not change
- [x] No, they are an experimental feature
- [ ] Yes, but only in the latest version
- [ ] No, they are deprecated

> **Explanation:** Decorators are an experimental feature in TypeScript and may change in future versions.

### True or False: Decorators can be applied to classes, methods, properties, and parameters.

- [x] True
- [ ] False

> **Explanation:** Decorators can be applied to classes, methods, properties, and parameters, allowing for flexible code enhancement.

{{< /quizdown >}}
