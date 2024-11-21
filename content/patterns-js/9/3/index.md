---
linkTitle: "9.3 Discriminated Unions"
title: "Discriminated Unions in TypeScript: Enhancing Type Safety and Code Clarity"
description: "Explore discriminated unions in TypeScript, a powerful pattern for combining multiple types with a common discriminant property to enable type guarding and improve type safety."
categories:
- TypeScript
- Design Patterns
- Software Development
tags:
- TypeScript
- Discriminated Unions
- Type Safety
- Code Clarity
- Type Guards
date: 2024-10-25
type: docs
nav_weight: 930000
canonical: "https://softwarepatternslexicon.com/patterns-js/9/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 9.3 Discriminated Unions

### Introduction

Discriminated unions, also known as tagged unions or algebraic data types, are a powerful feature in TypeScript that allow developers to combine multiple types into a union with a common discriminant property. This pattern enables type guarding, improving type safety and code clarity. In this article, we will delve into the concept of discriminated unions, explore their implementation, and examine practical use cases.

### Understanding the Concept

Discriminated unions are particularly useful when modeling scenarios where a value can be one of several types. By using a common discriminant property, such as `type` or `kind`, developers can simplify complex type checks and ensure all cases are handled. This approach not only enhances type safety but also makes the code more readable and maintainable.

### Implementation Steps

#### Define Interfaces with Common Property

The first step in implementing discriminated unions is to define interfaces that share a literal type property. This property acts as a discriminant, allowing TypeScript to differentiate between the various types in the union.

```typescript
interface Circle {
  kind: 'circle';
  radius: number;
}

interface Square {
  kind: 'square';
  sideLength: number;
}

interface Rectangle {
  kind: 'rectangle';
  width: number;
  height: number;
}
```

#### Create Union Type

Next, combine the interfaces into a union type. This union type represents a value that can be one of the defined interfaces.

```typescript
type Shape = Circle | Square | Rectangle;
```

#### Implement Type Guards

To utilize the discriminant property for type guarding, write functions that check the `kind` property to narrow down the types. This ensures that the correct operations are performed based on the specific type.

```typescript
function calculateArea(shape: Shape): number {
  switch (shape.kind) {
    case 'circle':
      return Math.PI * shape.radius ** 2;
    case 'square':
      return shape.sideLength ** 2;
    case 'rectangle':
      return shape.width * shape.height;
    default:
      // Exhaustive check
      const _exhaustiveCheck: never = shape;
      return _exhaustiveCheck;
  }
}
```

### Code Examples

Let's explore a practical example of using discriminated unions to calculate the area of different shapes.

```typescript
const myCircle: Circle = { kind: 'circle', radius: 5 };
const mySquare: Square = { kind: 'square', sideLength: 4 };
const myRectangle: Rectangle = { kind: 'rectangle', width: 3, height: 6 };

console.log(calculateArea(myCircle)); // Outputs: 78.53981633974483
console.log(calculateArea(mySquare)); // Outputs: 16
console.log(calculateArea(myRectangle)); // Outputs: 18
```

### Use Cases

Discriminated unions are ideal for scenarios where a value can be one of several types, such as:

- **Event Handling Systems:** Each event type can have specific properties, and discriminated unions can simplify event processing.
- **Form Validation:** Different form fields can have different validation rules, and discriminated unions can help manage these variations.
- **State Management:** Representing different states of an application, such as loading, success, and error states.

### Practice

To practice implementing discriminated unions, consider creating an event handling system where each event type has specific properties. Use the `kind` property to differentiate between event types and handle them accordingly.

### Considerations

When using discriminated unions, keep the following considerations in mind:

- **Improved Type Safety:** Discriminated unions enhance type safety by ensuring that all possible cases are handled.
- **Code Clarity:** By using a common discriminant property, the code becomes more readable and maintainable.
- **Exhaustive Type Checking:** Use the `never` type to ensure that all cases are handled, providing an additional layer of type safety.

### Conclusion

Discriminated unions are a powerful feature in TypeScript that enhance type safety and code clarity. By combining multiple types into a union with a common discriminant property, developers can simplify complex type checks and ensure all cases are handled. Whether you're modeling event handling systems, form validation, or state management, discriminated unions offer a robust solution for managing multiple types in TypeScript.

## Quiz Time!

{{< quizdown >}}

### What is a discriminated union in TypeScript?

- [x] A union of multiple types with a common discriminant property
- [ ] A single type with multiple properties
- [ ] A type that can only have one value
- [ ] A function that returns different types

> **Explanation:** Discriminated unions combine multiple types into a union with a common discriminant property, allowing for type guarding.

### Which property is typically used as a discriminant in a discriminated union?

- [x] A literal type property like `kind` or `type`
- [ ] A numeric property
- [ ] A boolean property
- [ ] A function property

> **Explanation:** A literal type property such as `kind` or `type` is used as a discriminant to differentiate between types in the union.

### How do discriminated unions improve type safety?

- [x] By ensuring all possible cases are handled
- [ ] By allowing any type to be used
- [ ] By removing type checks
- [ ] By using only primitive types

> **Explanation:** Discriminated unions improve type safety by ensuring that all possible cases are handled through type guards.

### What is the purpose of the `never` type in discriminated unions?

- [x] To ensure exhaustive type checking
- [ ] To allow any value
- [ ] To represent a nullable type
- [ ] To define a default value

> **Explanation:** The `never` type is used to ensure exhaustive type checking, indicating that all cases have been handled.

### Which of the following is a use case for discriminated unions?

- [x] Event handling systems
- [ ] Simple arithmetic operations
- [ ] String concatenation
- [ ] File I/O operations

> **Explanation:** Discriminated unions are useful in scenarios like event handling systems where different event types have specific properties.

### What is the first step in implementing discriminated unions?

- [x] Define interfaces with a common property
- [ ] Write a function to handle all types
- [ ] Create a class for each type
- [ ] Use a switch statement to differentiate types

> **Explanation:** The first step is to define interfaces with a common property that acts as a discriminant.

### How can discriminated unions simplify complex type checks?

- [x] By using a common discriminant property for type guarding
- [ ] By removing all type checks
- [ ] By using only primitive types
- [ ] By allowing any type to be used

> **Explanation:** Discriminated unions simplify complex type checks by using a common discriminant property to differentiate between types.

### What is the benefit of using a union type in discriminated unions?

- [x] It represents a value that can be one of several types
- [ ] It restricts a value to a single type
- [ ] It allows only primitive types
- [ ] It removes the need for type checks

> **Explanation:** A union type represents a value that can be one of several types, allowing for flexibility and type safety.

### How do discriminated unions enhance code clarity?

- [x] By making the code more readable and maintainable
- [ ] By removing all comments
- [ ] By using only one type
- [ ] By allowing any value

> **Explanation:** Discriminated unions enhance code clarity by using a common discriminant property, making the code more readable and maintainable.

### True or False: Discriminated unions can only be used with primitive types.

- [ ] True
- [x] False

> **Explanation:** Discriminated unions can be used with any types, not just primitive types, as long as they share a common discriminant property.

{{< /quizdown >}}
