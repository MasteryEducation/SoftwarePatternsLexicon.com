---
canonical: "https://softwarepatternslexicon.com/patterns-ts/15/1/1"
title: "TypeScript Decorators for Metaprogramming: Enhance Your Code with Dynamic Behavior"
description: "Explore the power of TypeScript decorators for metaprogramming, enabling dynamic modification of class behavior, method interception, and more."
linkTitle: "15.1.1 Using Decorators for Metaprogramming"
categories:
- TypeScript
- Metaprogramming
- Design Patterns
tags:
- TypeScript
- Decorators
- Metaprogramming
- Design Patterns
- Reflection
date: 2024-11-17
type: docs
nav_weight: 15110
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 15.1.1 Using Decorators for Metaprogramming

In the realm of TypeScript, decorators offer a powerful mechanism for metaprogramming, allowing developers to dynamically modify or extend the behavior of classes, methods, and properties at runtime. This section delves into the intricacies of decorators, exploring their types, use cases, and integration with design patterns.

### Understanding Decorators in TypeScript

#### What are Decorators?

Decorators in TypeScript are a special kind of declaration that can be attached to classes, methods, accessors, properties, or parameters. They provide a way to add annotations and a meta-programming syntax for class declarations and members. Essentially, decorators are functions that are invoked with a specific target, allowing you to modify or augment the behavior of the target.

**Syntactic Structure of Decorators**

The basic syntax of a decorator involves the `@` symbol followed by the decorator name, which is a function. Here's a simple example:

```typescript
function MyDecorator(target: any) {
    // Decorator logic
}

@MyDecorator
class MyClass {
    // Class definition
}
```

In this example, `MyDecorator` is applied to `MyClass`, allowing the decorator function to manipulate or enhance the class.

#### Historical Context

Decorators were inspired by similar concepts in other programming languages like Python and Java. They were introduced as part of the ECMAScript proposal process and are currently at Stage 2. This means they are still experimental and subject to change, but TypeScript has adopted them with some extensions to support various use cases.

### Types of Decorators

TypeScript supports several types of decorators, each serving different purposes:

#### Class Decorators

Class decorators are applied to the class constructor. They can be used to modify or replace the class definition.

**Use Case**: Logging class instantiation, applying singleton patterns.

```typescript
function Singleton<T extends { new(...args: any[]): {} }>(constructor: T) {
    return class extends constructor {
        private static instance: T;
        
        constructor(...args: any[]) {
            if (!Singleton.instance) {
                Singleton.instance = new constructor(...args);
            }
            return Singleton.instance;
        }
    }
}

@Singleton
class MyService {
    constructor() {
        console.log("MyService instance created");
    }
}

const service1 = new MyService();
const service2 = new MyService();
console.log(service1 === service2); // true
```

#### Method Decorators

Method decorators are applied to methods within a class. They can intercept method calls, modify method behavior, or even replace the method.

**Use Case**: Logging method calls, enforcing access control.

```typescript
function Log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;

    descriptor.value = function (...args: any[]) {
        console.log(`Calling ${propertyKey} with arguments: ${JSON.stringify(args)}`);
        return originalMethod.apply(this, args);
    };
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

#### Accessor Decorators

Accessor decorators are applied to getters and setters of class properties. They can modify the behavior of property accessors.

**Use Case**: Validating property values, lazy loading.

```typescript
function Validate(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalSet = descriptor.set;

    descriptor.set = function (value: any) {
        if (value < 0) {
            throw new Error("Value cannot be negative");
        }
        originalSet.call(this, value);
    };
}

class BankAccount {
    private _balance: number = 0;

    @Validate
    set balance(amount: number) {
        this._balance = amount;
    }

    get balance(): number {
        return this._balance;
    }
}

const account = new BankAccount();
account.balance = 100; // Works fine
// account.balance = -50; // Throws error
```

#### Property Decorators

Property decorators are applied to class properties. They can be used to add metadata or modify property behavior.

**Use Case**: Dependency injection, data binding.

```typescript
function ReadOnly(target: any, propertyKey: string) {
    Object.defineProperty(target, propertyKey, {
        writable: false
    });
}

class Configuration {
    @ReadOnly
    apiUrl: string = "https://api.example.com";
}

const config = new Configuration();
// config.apiUrl = "https://api.newurl.com"; // Error: Cannot assign to read-only property
```

#### Parameter Decorators

Parameter decorators are applied to function parameters. They can be used to inject dependencies or validate parameters.

**Use Case**: Dependency injection frameworks.

```typescript
function Inject(target: any, propertyKey: string, parameterIndex: number) {
    console.log(`Injected parameter at index ${parameterIndex} in ${propertyKey}`);
}

class Service {
    constructor(@Inject private dependency: any) {}
}
```

### Metaprogramming with Decorators

Decorators enable powerful metaprogramming capabilities, allowing you to modify class behavior, intercept method calls, and inject dependencies.

#### Modifying Class Behavior

Decorators can be used to extend or modify class behavior dynamically. For example, you can create a decorator that automatically logs all method calls within a class.

```typescript
function LogAllMethods(target: any) {
    for (const key of Object.getOwnPropertyNames(target.prototype)) {
        const descriptor = Object.getOwnPropertyDescriptor(target.prototype, key);
        if (descriptor && typeof descriptor.value === 'function') {
            const originalMethod = descriptor.value;
            descriptor.value = function (...args: any[]) {
                console.log(`Calling ${key} with arguments: ${JSON.stringify(args)}`);
                return originalMethod.apply(this, args);
            };
            Object.defineProperty(target.prototype, key, descriptor);
        }
    }
}

@LogAllMethods
class Logger {
    action1() {
        console.log("Action 1 executed");
    }

    action2() {
        console.log("Action 2 executed");
    }
}

const logger = new Logger();
logger.action1(); // Logs: Calling action1 with arguments: []
logger.action2(); // Logs: Calling action2 with arguments: []
```

#### Intercepting Method Calls

Method decorators can intercept calls to a method, allowing you to add pre- or post-processing logic.

```typescript
function TimeExecution(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;

    descriptor.value = function (...args: any[]) {
        console.time(propertyKey);
        const result = originalMethod.apply(this, args);
        console.timeEnd(propertyKey);
        return result;
    };
}

class PerformanceTester {
    @TimeExecution
    compute() {
        // Simulate a time-consuming task
        for (let i = 0; i < 1e6; i++) {}
    }
}

const tester = new PerformanceTester();
tester.compute(); // Logs execution time
```

#### Injecting Dependencies

Decorators can be used to inject dependencies into classes, facilitating dependency injection patterns.

```typescript
function InjectService(serviceIdentifier: string) {
    return function (target: any, propertyKey: string) {
        const service = ServiceLocator.getService(serviceIdentifier);
        target[propertyKey] = service;
    };
}

class Consumer {
    @InjectService('MyService')
    private service: any;

    useService() {
        this.service.performAction();
    }
}
```

### Integration with Design Patterns

Decorators can enhance or implement various design patterns, such as Singleton, Observer, or Dependency Injection.

#### Singleton Pattern

The Singleton pattern ensures a class has only one instance. Decorators can enforce this pattern by modifying the class constructor.

```typescript
function Singleton<T extends { new(...args: any[]): {} }>(constructor: T) {
    return class extends constructor {
        private static instance: T;

        constructor(...args: any[]) {
            if (!Singleton.instance) {
                Singleton.instance = new constructor(...args);
            }
            return Singleton.instance;
        }
    }
}

@Singleton
class DatabaseConnection {
    constructor() {
        console.log("Database connection established");
    }
}

const db1 = new DatabaseConnection();
const db2 = new DatabaseConnection();
console.log(db1 === db2); // true
```

#### Observer Pattern

The Observer pattern involves notifying observers when a subject changes state. Decorators can simplify the implementation of this pattern.

```typescript
function Observable(target: any, propertyKey: string) {
    let value = target[propertyKey];

    const observers: Function[] = [];

    Object.defineProperty(target, propertyKey, {
        get: () => value,
        set: (newValue) => {
            value = newValue;
            observers.forEach(observer => observer(newValue));
        }
    });

    target[`addObserver`] = (observer: Function) => {
        observers.push(observer);
    };
}

class Subject {
    @Observable
    state: number = 0;
}

const subject = new Subject();
subject.addObserver((newState: number) => console.log(`State changed to: ${newState}`));

subject.state = 10; // Logs: State changed to: 10
```

#### Dependency Injection

Decorators can facilitate dependency injection by automatically injecting dependencies into class properties or constructor parameters.

```typescript
function InjectService(serviceIdentifier: string) {
    return function (target: any, propertyKey: string) {
        const service = ServiceLocator.getService(serviceIdentifier);
        target[propertyKey] = service;
    };
}

class Consumer {
    @InjectService('MyService')
    private service: any;

    useService() {
        this.service.performAction();
    }
}
```

### Reflection and Metadata

#### Reflection in TypeScript

Reflection is the ability of a program to inspect and modify its own structure. In TypeScript, decorators can work with reflection to store and retrieve metadata.

#### Using Reflect Metadata API

The Reflect Metadata API allows you to define and retrieve metadata for class members. This is particularly useful for building frameworks and libraries.

```typescript
import 'reflect-metadata';

function Metadata(key: string, value: any) {
    return Reflect.metadata(key, value);
}

class Example {
    @Metadata('role', 'admin')
    method() {}
}

const metadataValue = Reflect.getMetadata('role', Example.prototype, 'method');
console.log(metadataValue); // Outputs: admin
```

### Best Practices and Considerations

#### Potential Pitfalls

- **Evaluation Order**: Decorators are evaluated in reverse order of their declaration. Be mindful of this when stacking multiple decorators.
- **Type Checking**: Decorators can obscure type checking, so ensure your decorators do not introduce type inconsistencies.

#### Performance Considerations

Decorators can introduce overhead, especially if they perform complex logic. Use them judiciously to avoid performance bottlenecks.

#### Maintaining Code Clarity

While decorators can simplify code, overuse or misuse can lead to confusion. Use decorators to enhance readability and maintainability, not to obscure logic.

### Conclusion

Decorators in TypeScript offer a powerful tool for metaprogramming, enabling dynamic modification of class behavior, method interception, and dependency injection. By understanding and leveraging decorators, you can implement sophisticated design patterns and enhance your code's flexibility and maintainability. As you explore decorators, remember to balance their power with clarity, ensuring your code remains understandable and efficient.

---

## Quiz Time!

{{< quizdown >}}

### What is a decorator in TypeScript?

- [x] A special kind of declaration that can be attached to classes, methods, accessors, properties, or parameters.
- [ ] A function that only modifies method behavior.
- [ ] A built-in TypeScript feature that replaces classes.
- [ ] A syntax error in TypeScript.

> **Explanation:** Decorators are special declarations in TypeScript that can be applied to various elements like classes and methods to modify their behavior.

### Which decorator type is used to modify class constructors?

- [x] Class decorator
- [ ] Method decorator
- [ ] Property decorator
- [ ] Parameter decorator

> **Explanation:** Class decorators are applied to class constructors to modify or replace the class definition.

### What is a common use case for method decorators?

- [x] Logging method calls
- [ ] Injecting services
- [ ] Replacing class constructors
- [ ] Modifying property accessors

> **Explanation:** Method decorators are often used to log method calls or modify method behavior.

### How can decorators be used in the Singleton pattern?

- [x] By modifying the class constructor to ensure only one instance is created.
- [ ] By replacing all methods with singletons.
- [ ] By injecting dependencies into methods.
- [ ] By changing property values to singleton instances.

> **Explanation:** Decorators can modify the class constructor to enforce the Singleton pattern by ensuring only one instance is created.

### What does the Reflect Metadata API allow you to do?

- [x] Define and retrieve metadata for class members.
- [ ] Replace class constructors.
- [ ] Modify method behavior.
- [ ] Inject dependencies into properties.

> **Explanation:** The Reflect Metadata API allows you to define and retrieve metadata for class members, useful for frameworks and libraries.

### What is a potential pitfall of using decorators?

- [x] They can obscure type checking.
- [ ] They always improve performance.
- [ ] They are only applicable to methods.
- [ ] They replace class definitions.

> **Explanation:** Decorators can obscure type checking, so it's important to ensure they don't introduce type inconsistencies.

### Which decorator type can be used to inject dependencies into class properties?

- [x] Property decorator
- [ ] Class decorator
- [ ] Method decorator
- [ ] Accessor decorator

> **Explanation:** Property decorators can be used to inject dependencies into class properties.

### How are decorators evaluated when multiple are applied?

- [x] In reverse order of their declaration.
- [ ] In the order they are declared.
- [ ] Randomly.
- [ ] Only the first one is applied.

> **Explanation:** Decorators are evaluated in reverse order of their declaration.

### What is a best practice when using decorators?

- [x] Use them to enhance readability and maintainability.
- [ ] Use them to obscure logic.
- [ ] Apply them to every method and property.
- [ ] Avoid using them entirely.

> **Explanation:** Decorators should be used to enhance readability and maintainability, not to obscure logic.

### True or False: Decorators can only be used with classes in TypeScript.

- [ ] True
- [x] False

> **Explanation:** Decorators can be applied to classes, methods, accessors, properties, and parameters in TypeScript.

{{< /quizdown >}}
