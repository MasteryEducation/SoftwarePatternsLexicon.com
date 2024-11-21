---
linkTitle: "9.1 Generics"
title: "Mastering Generics in TypeScript: Enhance Code Reusability and Type Safety"
description: "Explore the power of generics in TypeScript to create reusable and type-safe components and functions. Learn how to implement generic functions, classes, and interfaces with practical examples and best practices."
categories:
- TypeScript
- Programming
- Software Development
tags:
- Generics
- TypeScript
- Code Reusability
- Type Safety
- Programming Patterns
date: 2024-10-25
type: docs
nav_weight: 910000
canonical: "https://softwarepatternslexicon.com/patterns-js/9/1"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 9.1 Generics

Generics in TypeScript are a powerful feature that allows developers to create reusable and type-safe components and functions. By leveraging generics, you can write code that works with a variety of data types while maintaining strict type safety. This section will guide you through understanding, implementing, and applying generics effectively in your TypeScript projects.

### Understand the Concept

Generics enable the creation of components and functions that can operate over different types without sacrificing type safety. The syntax for generics involves using angle brackets `<T>`, where `T` is a placeholder for the type that will be specified later.

#### Key Benefits of Generics:
- **Reusability:** Write code once and use it with different data types.
- **Type Safety:** Ensure that operations on data types are valid at compile time.
- **Flexibility:** Adapt to various data structures and algorithms without rewriting code.

### Implementation Steps

#### Create Generic Functions

Generic functions allow you to define functions that can work with any data type. Here's how you can create a simple generic function:

```typescript
function identity<T>(value: T): T {
    return value;
}

const numberIdentity = identity<number>(42);
const stringIdentity = identity<string>("Hello, Generics!");
```

In this example, the `identity` function takes a type parameter `T` and returns a value of the same type. This ensures that the function can handle any type while maintaining type safety.

#### Develop Generic Classes and Interfaces

Generics can also be applied to classes and interfaces, allowing you to create flexible and reusable data structures.

**Generic Class Example:**

```typescript
class Repository<T> {
    private items: T[] = [];

    add(item: T): void {
        this.items.push(item);
    }

    getAll(): T[] {
        return this.items;
    }
}

const numberRepo = new Repository<number>();
numberRepo.add(10);
numberRepo.add(20);

const stringRepo = new Repository<string>();
stringRepo.add("TypeScript");
stringRepo.add("Generics");
```

**Generic Interface Example:**

```typescript
interface Pair<K, V> {
    key: K;
    value: V;
}

const numberStringPair: Pair<number, string> = { key: 1, value: "One" };
const stringBooleanPair: Pair<string, boolean> = { key: "isActive", value: true };
```

#### Use Constraints

Constraints in generics allow you to limit the types that can be used with a generic function or class. This is done using the `extends` keyword.

```typescript
function logLength<T extends { length: number }>(arg: T): void {
    console.log(arg.length);
}

logLength("Hello, World!"); // Works because string has a length property
logLength([1, 2, 3, 4]);    // Works because array has a length property
```

### Code Examples

#### Implementing a Generic `Stack<T>` Class

A stack is a common data structure that follows the Last In First Out (LIFO) principle. Here's how you can implement a generic stack in TypeScript:

```typescript
class Stack<T> {
    private elements: T[] = [];

    push(element: T): void {
        this.elements.push(element);
    }

    pop(): T | undefined {
        return this.elements.pop();
    }

    peek(): T | undefined {
        return this.elements[this.elements.length - 1];
    }

    isEmpty(): boolean {
        return this.elements.length === 0;
    }
}

const numberStack = new Stack<number>();
numberStack.push(1);
numberStack.push(2);
console.log(numberStack.pop()); // Outputs: 2

const stringStack = new Stack<string>();
stringStack.push("A");
stringStack.push("B");
console.log(stringStack.pop()); // Outputs: B
```

### Use Cases

Generics are particularly useful in scenarios where you need to build reusable data structures or utility functions. Here are some common use cases:

- **Reusable Data Structures:** Implement lists, trees, or maps that can handle various data types.
- **Utility Functions:** Write functions that perform operations on different types of data, such as sorting or filtering.

### Practice

Try writing a generic function that filters elements in an array based on a provided predicate function.

```typescript
function filterArray<T>(array: T[], predicate: (value: T) => boolean): T[] {
    return array.filter(predicate);
}

const numbers = [1, 2, 3, 4, 5];
const evenNumbers = filterArray(numbers, num => num % 2 === 0);
console.log(evenNumbers); // Outputs: [2, 4]
```

### Considerations

While generics enhance code reusability and type safety, it's important to use them judiciously. Overusing generics can lead to complex type inference and make the code harder to understand. Always strive for a balance between flexibility and simplicity.

### Best Practices

- **Use Descriptive Type Parameters:** Instead of using single letters like `T`, consider using more descriptive names like `ItemType` or `KeyType` for better readability.
- **Limit Generic Parameters:** Avoid using too many generic parameters, which can complicate the code.
- **Document Generic Functions and Classes:** Provide clear documentation to explain the purpose and constraints of generic parameters.

### Conclusion

Generics are a cornerstone of TypeScript's type system, enabling developers to write flexible, reusable, and type-safe code. By understanding and applying generics effectively, you can create robust applications that handle a wide range of data types without compromising on type safety.

## Quiz Time!

{{< quizdown >}}

### What is the primary benefit of using generics in TypeScript?

- [x] They allow for code reusability and type safety.
- [ ] They improve runtime performance.
- [ ] They simplify the syntax of TypeScript.
- [ ] They eliminate the need for type annotations.

> **Explanation:** Generics enable code reusability and maintain type safety by allowing functions and classes to work with any data type.

### How do you define a generic function in TypeScript?

- [x] Using angle brackets `<T>` to specify a type parameter.
- [ ] Using parentheses `()` to specify a type parameter.
- [ ] Using square brackets `[]` to specify a type parameter.
- [ ] Using curly braces `{}` to specify a type parameter.

> **Explanation:** Generic functions in TypeScript are defined using angle brackets `<T>` to specify type parameters.

### Which of the following is a correct implementation of a generic class?

- [x] `class Repository<T> { ... }`
- [ ] `class Repository(T) { ... }`
- [ ] `class Repository[T] { ... }`
- [ ] `class Repository{T} { ... }`

> **Explanation:** The correct syntax for a generic class in TypeScript uses angle brackets `<T>`.

### What is the purpose of using constraints in generics?

- [x] To limit the types that can be used with a generic function or class.
- [ ] To improve the performance of generic functions.
- [ ] To simplify the syntax of generics.
- [ ] To allow generics to work with any type.

> **Explanation:** Constraints in generics are used to limit the types that can be used, ensuring that only compatible types are allowed.

### How can you implement a generic stack in TypeScript?

- [x] By using a class with a type parameter `<T>`.
- [ ] By using a function with a type parameter `<T>`.
- [ ] By using an interface with a type parameter `<T>`.
- [ ] By using a module with a type parameter `<T>`.

> **Explanation:** A generic stack can be implemented using a class with a type parameter `<T>` to handle different data types.

### What is a common use case for generics?

- [x] Building reusable data structures like lists or maps.
- [ ] Improving the performance of TypeScript applications.
- [ ] Simplifying the syntax of TypeScript.
- [ ] Eliminating the need for type annotations.

> **Explanation:** Generics are commonly used to build reusable data structures that can handle various data types.

### What should you consider when using generics?

- [x] Avoid overusing generics to prevent complex type inference.
- [ ] Use as many generic parameters as possible.
- [ ] Avoid using constraints with generics.
- [ ] Use single-letter type parameters for simplicity.

> **Explanation:** It's important to avoid overusing generics, as it can lead to complex type inference and make the code harder to understand.

### How can you filter elements in an array using a generic function?

- [x] By defining a function with a type parameter `<T>` and a predicate function.
- [ ] By using a class with a type parameter `<T>` and a predicate function.
- [ ] By using an interface with a type parameter `<T>` and a predicate function.
- [ ] By using a module with a type parameter `<T>` and a predicate function.

> **Explanation:** A generic function with a type parameter `<T>` and a predicate function can be used to filter elements in an array.

### What is the syntax to apply constraints in generics?

- [x] Using the `extends` keyword.
- [ ] Using the `implements` keyword.
- [ ] Using the `with` keyword.
- [ ] Using the `as` keyword.

> **Explanation:** Constraints in generics are applied using the `extends` keyword to limit allowable types.

### True or False: Generics can only be used with classes in TypeScript.

- [ ] True
- [x] False

> **Explanation:** Generics can be used with functions, classes, and interfaces in TypeScript, not just classes.

{{< /quizdown >}}
