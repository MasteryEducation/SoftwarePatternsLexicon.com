---
linkTitle: "9.6 Type Guards"
title: "Type Guards in TypeScript: Enhancing Type Safety and Code Reliability"
description: "Explore the concept of Type Guards in TypeScript, learn how to implement them using custom functions, and understand their role in refining types at runtime."
categories:
- TypeScript
- Design Patterns
- Programming
tags:
- Type Guards
- TypeScript
- Type Safety
- JavaScript
- Programming Patterns
date: 2024-10-25
type: docs
nav_weight: 960000
canonical: "https://softwarepatternslexicon.com/patterns-js/9/6"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 9.6 Type Guards in TypeScript

TypeScript is a powerful language that extends JavaScript by adding static types. One of the key features that make TypeScript robust is its ability to perform type checks at compile time. However, there are scenarios where runtime checks are necessary to ensure type safety, and this is where Type Guards come into play.

### Understanding Type Guards

Type Guards are a mechanism in TypeScript that allows developers to perform runtime checks to refine types within a specific block of code. They are particularly useful when working with union types or when you need to ensure that a variable is of a certain type before performing operations on it.

#### Key Concepts

- **Type Guard Functions:** These are custom functions that return a type predicate, allowing TypeScript to narrow down the type of a variable within a block.
- **Built-in Operators:** TypeScript provides operators like `typeof`, `instanceof`, and `in` to perform type checks.
- **Discriminated Unions:** A pattern that uses a common property with literal types to differentiate between variants in a union type.

### Implementation Steps

#### 1. Write Type Guard Functions

A Type Guard function is a function that returns a type predicate. The return type is specified using the syntax `value is Type`, which tells TypeScript that if the function returns true, the value is of the specified type.

**Example:**

```typescript
function isString(value: unknown): value is string {
  return typeof value === 'string';
}
```

In this example, `isString` is a type guard function that checks if a given value is a string.

#### 2. Use `typeof`, `instanceof`, and `in`

- **`typeof`:** Used for checking primitive types like `string`, `number`, `boolean`, etc.
- **`instanceof`:** Used for checking if an object is an instance of a class.
- **`in`:** Used for checking if a property exists in an object.

**Example using `instanceof`:**

```typescript
function isDate(value: unknown): value is Date {
  return value instanceof Date;
}
```

#### 3. Implement Discriminated Unions

Discriminated Unions use a common property to distinguish between different types in a union. This property is usually a literal type.

**Example:**

```typescript
type Circle = { kind: 'circle'; radius: number };
type Square = { kind: 'square'; sideLength: number };

type Shape = Circle | Square;

function getArea(shape: Shape): number {
  if (shape.kind === 'circle') {
    return Math.PI * shape.radius ** 2;
  } else {
    return shape.sideLength ** 2;
  }
}
```

In this example, the `kind` property is used to discriminate between `Circle` and `Square`.

### Code Examples

Let's explore some practical code examples to understand how Type Guards can be implemented and used effectively.

#### Using `typeof` Type Guard

```typescript
function isNumber(value: unknown): value is number {
  return typeof value === 'number';
}

function doubleValue(value: unknown): number | null {
  if (isNumber(value)) {
    return value * 2;
  }
  return null;
}
```

#### Using `instanceof` Type Guard

```typescript
class Animal {
  speak() {
    console.log("Animal speaks");
  }
}

class Dog extends Animal {
  bark() {
    console.log("Woof!");
  }
}

function isDog(animal: Animal): animal is Dog {
  return animal instanceof Dog;
}

const pet: Animal = new Dog();

if (isDog(pet)) {
  pet.bark(); // TypeScript knows pet is Dog here
}
```

#### Discriminated Union Example

```typescript
type Vehicle = Car | Bike;

interface Car {
  type: 'car';
  numberOfDoors: number;
}

interface Bike {
  type: 'bike';
  hasCarrier: boolean;
}

function describeVehicle(vehicle: Vehicle): string {
  switch (vehicle.type) {
    case 'car':
      return `Car with ${vehicle.numberOfDoors} doors.`;
    case 'bike':
      return `Bike with${vehicle.hasCarrier ? '' : 'out'} a carrier.`;
  }
}
```

### Use Cases

Type Guards are particularly useful in the following scenarios:

- **Narrowing Down Types from a Union:** When dealing with union types, Type Guards help in narrowing down to a specific type.
- **Safely Accessing Properties or Methods:** Before accessing a property or method, Type Guards ensure that the variable is of the correct type, preventing runtime errors.

### Practice

#### Exercise 1: Create a Type Guard `isNumber`

```typescript
function isNumber(value: unknown): value is number {
  return typeof value === 'number';
}
```

#### Exercise 2: Implement a Function to Check if an Object is an `Array`

```typescript
function isArray(value: unknown): value is Array<any> {
  return Array.isArray(value);
}
```

### Considerations

- **Enhance Code Safety:** Type Guards provide an additional layer of safety by ensuring that operations are performed on the correct types.
- **Accuracy:** Ensure that your Type Guard functions are accurate to prevent false assumptions about the types.

### Best Practices

- **Use Built-in Operators Wisely:** Leverage `typeof`, `instanceof`, and `in` for simple type checks.
- **Custom Type Guards for Complex Types:** For complex types, write custom Type Guard functions to encapsulate the logic.
- **Discriminated Unions for Union Types:** Use discriminated unions to manage complex union types effectively.

### Conclusion

Type Guards are a powerful feature in TypeScript that enhance type safety and reliability of your code. By using Type Guards, you can perform runtime checks that refine types, ensuring that your code is both robust and maintainable. As you continue to work with TypeScript, mastering Type Guards will be an invaluable skill in writing safe and efficient code.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of Type Guards in TypeScript?

- [x] To perform runtime checks that refine types within a specific block
- [ ] To compile TypeScript code to JavaScript
- [ ] To enhance the performance of TypeScript applications
- [ ] To manage state in a TypeScript application

> **Explanation:** Type Guards are used to perform runtime checks that refine types within a specific block, ensuring type safety.

### Which of the following is a Type Guard function?

- [x] `function isString(value: unknown): value is string { return typeof value === 'string'; }`
- [ ] `function isString(value: unknown): boolean { return typeof value === 'string'; }`
- [ ] `function isString(value: unknown): string { return typeof value === 'string'; }`
- [ ] `function isString(value: unknown): void { return typeof value === 'string'; }`

> **Explanation:** A Type Guard function returns a type predicate using the syntax `value is Type`.

### Which operator is used to check if a value is an instance of a class?

- [ ] `typeof`
- [x] `instanceof`
- [ ] `in`
- [ ] `is`

> **Explanation:** The `instanceof` operator is used to check if a value is an instance of a class.

### What is a discriminated union?

- [x] A pattern that uses a common property with literal types to differentiate between variants in a union type
- [ ] A union type that cannot be discriminated
- [ ] A type that discriminates against other types
- [ ] A type that uses only primitive types

> **Explanation:** Discriminated unions use a common property with literal types to differentiate between variants in a union type.

### How can you check if a value is a number using a Type Guard?

- [x] `function isNumber(value: unknown): value is number { return typeof value === 'number'; }`
- [ ] `function isNumber(value: unknown): boolean { return typeof value === 'number'; }`
- [ ] `function isNumber(value: unknown): number { return typeof value === 'number'; }`
- [ ] `function isNumber(value: unknown): void { return typeof value === 'number'; }`

> **Explanation:** The Type Guard function should return a type predicate using the syntax `value is Type`.

### Which of the following is not a built-in operator for type checking in TypeScript?

- [ ] `typeof`
- [ ] `instanceof`
- [ ] `in`
- [x] `is`

> **Explanation:** `is` is not a built-in operator for type checking in TypeScript.

### What is the return type of a Type Guard function?

- [x] `value is Type`
- [ ] `boolean`
- [ ] `Type`
- [ ] `void`

> **Explanation:** The return type of a Type Guard function is `value is Type`, which is a type predicate.

### Why are Type Guards important in TypeScript?

- [x] They enhance code safety by ensuring operations are performed on the correct types.
- [ ] They increase the execution speed of TypeScript code.
- [ ] They allow TypeScript to compile faster.
- [ ] They reduce the size of TypeScript files.

> **Explanation:** Type Guards enhance code safety by ensuring operations are performed on the correct types.

### Which of the following is a correct use of the `in` operator?

- [x] `if ('property' in object) { /* ... */ }`
- [ ] `if (object in 'property') { /* ... */ }`
- [ ] `if ('property' instanceof object) { /* ... */ }`
- [ ] `if (object typeof 'property') { /* ... */ }`

> **Explanation:** The `in` operator is used to check if a property exists in an object.

### Type Guards can be used to narrow down types from a union.

- [x] True
- [ ] False

> **Explanation:** Type Guards are often used to narrow down types from a union, ensuring that the correct type is used in a specific block of code.

{{< /quizdown >}}
