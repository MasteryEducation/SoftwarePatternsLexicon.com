---
canonical: "https://softwarepatternslexicon.com/patterns-ts/14/3"
title: "TypeScript Decorator: An In-Depth Guide to Enhancing Code with Decorators"
description: "Explore the use of decorators in TypeScript, an experimental feature inspired by the Decorator Pattern, allowing for annotation and modification of classes and their members in a declarative fashion."
linkTitle: "14.3 Decorator in TypeScript"
categories:
- TypeScript
- Design Patterns
- Software Engineering
tags:
- Decorators
- TypeScript
- Design Patterns
- Class Decorators
- Method Decorators
date: 2024-11-17
type: docs
nav_weight: 14300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 14.3 Decorator in TypeScript

Decorators in TypeScript provide a powerful way to add annotations and modify classes and their members. Inspired by the Decorator Pattern, TypeScript decorators offer a declarative approach to augmenting code functionality. In this section, we will delve into the intricacies of decorators, exploring their syntax, types, practical applications, and best practices.

### Understanding Decorators in TypeScript

Decorators are a special kind of declaration that can be attached to a class, method, accessor, property, or parameter. They are a form of syntactic sugar that allows developers to apply reusable logic to various parts of their codebase. In TypeScript, decorators are an experimental feature, meaning they are subject to change and require enabling in the `tsconfig.json` file with the `experimentalDecorators` flag.

#### Syntax and Current Status

The basic syntax for a decorator involves prefixing a function with the `@` symbol, followed by the decorator name. Here is a simple example of a class decorator:

```typescript
function sealed(constructor: Function) {
  Object.seal(constructor);
  Object.seal(constructor.prototype);
}

@sealed
class Greeter {
  greeting: string;
  constructor(message: string) {
    this.greeting = message;
  }
  greet() {
    return `Hello, ${this.greeting}`;
  }
}
```

In this example, the `sealed` decorator is applied to the `Greeter` class, sealing its constructor and prototype. Decorators are currently experimental, which means they may not be fully stable across all TypeScript versions. Developers should be cautious and stay updated with TypeScript releases to ensure compatibility.

### Types of Decorators

TypeScript supports several types of decorators, each serving a unique purpose. Let's explore each type in detail:

#### Class Decorators

Class decorators are applied to the constructor of a class. They can be used to observe, modify, or replace a class definition. A class decorator is defined just before a class declaration.

```typescript
function logClass(target: Function) {
  console.log(`Class: ${target.name}`);
}

@logClass
class MyClass {}
```

In this example, the `logClass` decorator logs the name of the class to the console.

#### Method Decorators

Method decorators are applied to a method within a class. They can be used to modify the method's behavior, such as logging or validation.

```typescript
function logMethod(target: Object, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log(`Method: ${propertyKey}, Arguments: ${args}`);
    return originalMethod.apply(this, args);
  };
}

class Calculator {
  @logMethod
  add(a: number, b: number): number {
    return a + b;
  }
}
```

Here, the `logMethod` decorator logs the method name and its arguments each time the `add` method is called.

#### Accessor Decorators

Accessor decorators are applied to the getters and setters of a class property. They can be used to modify the behavior of property access.

```typescript
function configurable(value: boolean) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    descriptor.configurable = value;
  };
}

class Point {
  private _x: number = 0;

  @configurable(false)
  get x() {
    return this._x;
  }
}
```

In this example, the `configurable` decorator sets the configurability of the `x` property accessor.

#### Property Decorators

Property decorators are applied to individual properties within a class. They are useful for adding metadata or validation.

```typescript
function readonly(target: Object, propertyKey: string) {
  Object.defineProperty(target, propertyKey, {
    writable: false
  });
}

class Person {
  @readonly
  name: string = "John Doe";
}
```

The `readonly` decorator makes the `name` property immutable.

#### Parameter Decorators

Parameter decorators are applied to the parameters of a class method. They can be used to inject dependencies or validate input.

```typescript
function logParameter(target: Object, propertyKey: string, parameterIndex: number) {
  console.log(`Parameter in method ${propertyKey} at index ${parameterIndex}`);
}

class User {
  greet(@logParameter message: string) {
    console.log(message);
  }
}
```

This example logs the parameter index of the `greet` method.

### Decorator Factories

Decorator factories are functions that return a decorator function. They allow for parameterized decorators, providing flexibility and reusability.

```typescript
function log(level: string) {
  return function (target: Object, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = function (...args: any[]) {
      console.log(`[${level}] Method: ${propertyKey}, Arguments: ${args}`);
      return originalMethod.apply(this, args);
    };
  };
}

class Logger {
  @log('INFO')
  info(message: string) {
    console.log(message);
  }

  @log('ERROR')
  error(message: string) {
    console.error(message);
  }
}
```

In this example, the `log` decorator factory allows for different log levels to be specified for each method.

### Practical Usage

Decorators have practical applications in various scenarios, such as metadata annotation, dependency injection, and aspect-oriented programming.

#### Metadata Annotation

Decorators can be used to add metadata to classes and methods, which can be retrieved at runtime using libraries like `reflect-metadata`.

```typescript
import "reflect-metadata";

function metadata(key: string, value: any) {
  return Reflect.metadata(key, value);
}

class Example {
  @metadata('role', 'admin')
  adminMethod() {}
}

const role = Reflect.getMetadata('role', Example.prototype, 'adminMethod');
console.log(role); // Output: admin
```

#### Dependency Injection

Decorators are commonly used in frameworks like Angular to inject dependencies into classes.

```typescript
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class DataService {
  constructor() {}
}
```

The `@Injectable` decorator marks the `DataService` class as a service that can be injected into other components.

#### Aspect-Oriented Programming

Decorators can implement cross-cutting concerns such as logging, security, and transaction management.

```typescript
function authorize(role: string) {
  return function (target: Object, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    descriptor.value = function (...args: any[]) {
      if (role !== 'admin') {
        throw new Error('Unauthorized');
      }
      return originalMethod.apply(this, args);
    };
  };
}

class SecureService {
  @authorize('admin')
  sensitiveOperation() {
    console.log('Sensitive operation performed');
  }
}
```

### Implementing Custom Decorators

Creating custom decorators allows developers to encapsulate reusable logic and apply it across different parts of their application.

#### Custom Class Decorators

```typescript
function timestamp(constructor: Function) {
  constructor.prototype.timestamp = new Date();
}

@timestamp
class Document {
  title: string;
  constructor(title: string) {
    this.title = title;
  }
}

const doc = new Document("My Document");
console.log(doc.timestamp); // Outputs the current date and time
```

#### Custom Method Decorators

```typescript
function measure(target: Object, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    const start = performance.now();
    const result = originalMethod.apply(this, args);
    const end = performance.now();
    console.log(`Execution time: ${end - start}ms`);
    return result;
  };
}

class PerformanceTest {
  @measure
  compute() {
    // Simulate a heavy computation
    for (let i = 0; i < 1e6; i++) {}
  }
}

const test = new PerformanceTest();
test.compute();
```

### Interplay with Design Patterns

While TypeScript decorators share a name with the Decorator Pattern, they serve a different purpose. The structural Decorator Pattern involves wrapping an object to add new behavior, whereas TypeScript decorators are more about annotating and modifying existing code.

#### Differences and Similarities

- **Purpose**: The Decorator Pattern is used to add new functionality to objects dynamically, while TypeScript decorators are used to add metadata or modify code behavior.
- **Implementation**: The Decorator Pattern involves creating wrapper classes, whereas TypeScript decorators use functions to modify classes and members.
- **Use Cases**: The Decorator Pattern is often used in scenarios requiring dynamic behavior changes, while TypeScript decorators are used for static code annotations and modifications.

### Best Practices and Considerations

When using decorators, it's essential to follow best practices to ensure maintainability and avoid potential pitfalls.

#### Effective Use of Decorators

- **Clarity**: Use decorators to enhance code readability and maintainability. Avoid complex logic within decorators.
- **Reusability**: Create reusable decorators that can be applied across different parts of the application.
- **Documentation**: Document decorators thoroughly to explain their purpose and usage.

#### Potential Issues

- **Evaluation Order**: Decorators are applied in a specific order, which can affect their behavior. Be mindful of the order when applying multiple decorators.
- **Compatibility**: As decorators are experimental, ensure compatibility with the TypeScript version used in your project.
- **Performance**: Avoid heavy computations within decorators to prevent performance bottlenecks.

### Conclusion

Decorators in TypeScript offer a powerful mechanism for annotating and modifying code in a declarative manner. While they share similarities with the Decorator Pattern, their primary use is for static code enhancements rather than dynamic behavior changes. By following best practices and understanding their limitations, developers can leverage decorators to create clean, maintainable, and efficient code.

Remember, this is just the beginning. As you progress, you'll discover more ways to utilize decorators effectively. Keep experimenting, stay curious, and enjoy the journey!

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of decorators in TypeScript?

- [x] To add metadata and modify code behavior
- [ ] To dynamically add new functionality to objects
- [ ] To replace the need for interfaces
- [ ] To simplify asynchronous programming

> **Explanation:** Decorators in TypeScript are primarily used to add metadata and modify code behavior, rather than dynamically adding new functionality to objects.

### Which of the following is NOT a type of decorator in TypeScript?

- [ ] Class decorator
- [ ] Method decorator
- [ ] Property decorator
- [x] Interface decorator

> **Explanation:** TypeScript does not support interface decorators. The supported types are class, method, property, accessor, and parameter decorators.

### How do you enable decorators in a TypeScript project?

- [x] By setting the `experimentalDecorators` flag to `true` in `tsconfig.json`
- [ ] By importing a special module
- [ ] By using a specific TypeScript version
- [ ] By enabling them in the JavaScript runtime

> **Explanation:** Decorators are an experimental feature in TypeScript and need to be enabled by setting the `experimentalDecorators` flag to `true` in the `tsconfig.json` file.

### What is a decorator factory?

- [x] A function that returns a decorator function
- [ ] A method that decorates a class
- [ ] A class that implements multiple decorators
- [ ] A property that holds decorator metadata

> **Explanation:** A decorator factory is a function that returns a decorator function, allowing for parameterized decorators.

### Which decorator type is used to modify the behavior of a method's parameters?

- [ ] Class decorator
- [ ] Method decorator
- [ ] Property decorator
- [x] Parameter decorator

> **Explanation:** Parameter decorators are used to modify the behavior of a method's parameters.

### In what order are multiple decorators applied to a single class member?

- [x] From bottom to top
- [ ] From top to bottom
- [ ] In alphabetical order
- [ ] In reverse alphabetical order

> **Explanation:** When multiple decorators are applied to a single class member, they are evaluated from bottom to top.

### What is a common use case for decorators in Angular?

- [x] Dependency injection
- [ ] State management
- [ ] Event handling
- [ ] Data binding

> **Explanation:** In Angular, decorators are commonly used for dependency injection, marking classes as injectable services.

### What is the difference between TypeScript decorators and the Decorator Pattern?

- [x] TypeScript decorators are for static code modifications, while the Decorator Pattern is for dynamic behavior changes
- [ ] TypeScript decorators are for dynamic behavior changes, while the Decorator Pattern is for static code modifications
- [ ] Both are used for the same purpose
- [ ] TypeScript decorators are only used in Angular

> **Explanation:** TypeScript decorators are used for static code modifications, such as adding metadata, while the Decorator Pattern is used for dynamically adding new behavior to objects.

### What should be avoided within decorators to prevent performance issues?

- [x] Heavy computations
- [ ] Metadata annotations
- [ ] Logging
- [ ] Dependency injection

> **Explanation:** Heavy computations should be avoided within decorators to prevent performance bottlenecks.

### True or False: Decorators can be used to replace the need for interfaces in TypeScript.

- [ ] True
- [x] False

> **Explanation:** False. Decorators are not a replacement for interfaces. They serve different purposes, with decorators adding metadata and modifying behavior, while interfaces define contracts for types.

{{< /quizdown >}}
