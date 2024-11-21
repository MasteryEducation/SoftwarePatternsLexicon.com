---
canonical: "https://softwarepatternslexicon.com/patterns-ts/15/3"
title: "Advanced TypeScript: Type Guards and Discriminated Unions"
description: "Explore advanced TypeScript features like type guards and discriminated unions to enhance type safety and flexibility in design pattern implementations."
linkTitle: "15.3 Type Guards and Discriminated Unions"
categories:
- TypeScript
- Design Patterns
- Software Engineering
tags:
- Type Guards
- Discriminated Unions
- TypeScript
- Advanced Types
- Design Patterns
date: 2024-11-17
type: docs
nav_weight: 15300
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 15.3 Type Guards and Discriminated Unions

In the realm of TypeScript, understanding and utilizing advanced types such as type guards and discriminated unions can significantly enhance the robustness and flexibility of your code. These features are particularly useful in implementing design patterns, where precise type handling is crucial. In this section, we will delve into these advanced TypeScript features, exploring their definitions, applications, and best practices.

### Understanding Type Guards

Type guards are a mechanism in TypeScript that allow you to narrow down the type of a variable within a conditional block. This is especially useful when dealing with union types, where a variable can be of multiple types. By using type guards, you can ensure that your code handles each possible type appropriately, thereby reducing runtime errors and enhancing type safety.

#### Built-in Type Guards

TypeScript provides several built-in type guards that you can use to check the type of a variable:

1. **`typeof` Operator**: This operator is used to check the type of primitive values like `string`, `number`, `boolean`, etc.

   ```typescript
   function isString(value: unknown): value is string {
     return typeof value === 'string';
   }

   const example: unknown = "Hello, TypeScript!";
   if (isString(example)) {
     console.log(example.toUpperCase()); // Safe to call string methods
   }
   ```

2. **`instanceof` Operator**: This operator is used to check if an object is an instance of a particular class.

   ```typescript
   class Animal {
     name: string;
     constructor(name: string) {
       this.name = name;
     }
   }

   class Dog extends Animal {
     breed: string;
     constructor(name: string, breed: string) {
       super(name);
       this.breed = breed;
     }
   }

   function isDog(animal: Animal): animal is Dog {
     return animal instanceof Dog;
   }

   const pet: Animal = new Dog("Buddy", "Golden Retriever");
   if (isDog(pet)) {
     console.log(pet.breed); // Safe to access breed
   }
   ```

#### Custom Type Guard Functions

In addition to built-in type guards, you can create custom type guard functions to handle more complex type checks. These functions return a type predicate, which is a special return type that indicates a successful type check.

```typescript
interface Bird {
  fly(): void;
  layEggs(): void;
}

interface Fish {
  swim(): void;
  layEggs(): void;
}

function isFish(pet: Bird | Fish): pet is Fish {
  return (pet as Fish).swim !== undefined;
}

const pet: Bird | Fish = { swim: () => console.log("Swimming"), layEggs: () => console.log("Laying eggs") };

if (isFish(pet)) {
  pet.swim(); // Safe to call swim
}
```

### Introducing Discriminated Unions

Discriminated unions, also known as tagged unions or algebraic data types, are a powerful feature in TypeScript that allow you to define a union of types with a common discriminant property. This property acts as a tag to distinguish between the different types in the union, enabling exhaustive type checking and safer code.

#### Syntax and Usage

A discriminated union is typically composed of several types that share a common property with a literal type. This common property is used to determine the specific type of the union at runtime.

```typescript
type Shape =
  | { kind: 'circle'; radius: number }
  | { kind: 'square'; sideLength: number }
  | { kind: 'rectangle'; width: number; height: number };

function area(shape: Shape): number {
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

In this example, the `kind` property serves as the discriminant, allowing the `area` function to handle each shape type appropriately.

### Applications in Design Patterns

Type guards and discriminated unions can be particularly useful in implementing various design patterns, such as the State, Visitor, and Interpreter patterns. Let's explore how these advanced types can enhance pattern implementations.

#### State Pattern

In the State pattern, an object changes its behavior when its internal state changes. Discriminated unions can be used to represent different states, while type guards ensure that state transitions are handled correctly.

```typescript
type State =
  | { type: 'idle' }
  | { type: 'loading' }
  | { type: 'success'; data: string }
  | { type: 'error'; message: string };

class Context {
  private state: State = { type: 'idle' };

  transition(newState: State) {
    this.state = newState;
  }

  handle() {
    switch (this.state.type) {
      case 'idle':
        console.log("System is idle.");
        break;
      case 'loading':
        console.log("System is loading.");
        break;
      case 'success':
        console.log(`Data: ${this.state.data}`);
        break;
      case 'error':
        console.error(`Error: ${this.state.message}`);
        break;
    }
  }
}

const context = new Context();
context.transition({ type: 'loading' });
context.handle(); // Output: System is loading.
```

#### Visitor Pattern

The Visitor pattern involves performing operations on elements of an object structure. Discriminated unions can represent different element types, while type guards ensure that visitors handle each type correctly.

```typescript
interface Circle {
  kind: 'circle';
  radius: number;
}

interface Square {
  kind: 'square';
  sideLength: number;
}

type ShapeElement = Circle | Square;

class ShapeVisitor {
  visit(shape: ShapeElement) {
    switch (shape.kind) {
      case 'circle':
        console.log(`Visiting circle with radius ${shape.radius}`);
        break;
      case 'square':
        console.log(`Visiting square with side length ${shape.sideLength}`);
        break;
    }
  }
}

const visitor = new ShapeVisitor();
const shapes: ShapeElement[] = [{ kind: 'circle', radius: 10 }, { kind: 'square', sideLength: 5 }];
shapes.forEach(shape => visitor.visit(shape));
```

#### Interpreter Pattern

In the Interpreter pattern, you define a grammar for a language and an interpreter to process it. Discriminated unions can represent different grammar rules, while type guards ensure that the interpreter processes each rule correctly.

```typescript
interface NumberExpression {
  type: 'number';
  value: number;
}

interface AdditionExpression {
  type: 'addition';
  left: Expression;
  right: Expression;
}

type Expression = NumberExpression | AdditionExpression;

function evaluate(expr: Expression): number {
  switch (expr.type) {
    case 'number':
      return expr.value;
    case 'addition':
      return evaluate(expr.left) + evaluate(expr.right);
  }
}

const expr: Expression = {
  type: 'addition',
  left: { type: 'number', value: 5 },
  right: { type: 'number', value: 10 }
};

console.log(evaluate(expr)); // Output: 15
```

### Enhancing Type Safety

One of the primary benefits of using type guards and discriminated unions is the enhanced type safety they provide. By catching errors at compile-time, these features help prevent runtime errors and ensure that your code behaves as expected.

#### Exhaustive Type Checking

Discriminated unions enable exhaustive type checking, where the TypeScript compiler ensures that all possible cases in a union are handled. This is particularly useful in switch statements, where failing to handle a case results in a compile-time error.

```typescript
function handleShape(shape: Shape) {
  switch (shape.kind) {
    case 'circle':
      console.log(`Circle with radius ${shape.radius}`);
      break;
    case 'square':
      console.log(`Square with side length ${shape.sideLength}`);
      break;
    case 'rectangle':
      console.log(`Rectangle with width ${shape.width} and height ${shape.height}`);
      break;
    default:
      const _exhaustiveCheck: never = shape;
      return _exhaustiveCheck;
  }
}
```

### Potential Issues

While type guards and discriminated unions offer many benefits, they can also introduce complexity into your type definitions. It's important to keep type definitions manageable and avoid overcomplicating your code.

#### Performance Considerations

Type guards and discriminated unions can impact performance, especially if your type checks involve complex logic or large data structures. It's important to balance type safety with performance, ensuring that your code remains efficient.

### Best Practices

To make the most of type guards and discriminated unions, consider the following best practices:

- **Write Clear Type Guard Functions**: Ensure that your custom type guard functions are easy to understand and maintain. Use descriptive names and comments to explain their purpose.

- **Use Discriminated Unions Appropriately**: Discriminated unions are ideal for modeling complex data structures with multiple variants. Use them when you need to represent a set of related types with distinct properties.

- **Leverage Exhaustive Type Checking**: Take advantage of TypeScript's exhaustive type checking to catch errors early and ensure that all possible cases are handled.

- **Keep Type Definitions Manageable**: Avoid overly complex type definitions that can be difficult to understand and maintain. Break down complex types into smaller, more manageable components.

### Conclusion

Advanced TypeScript features like type guards and discriminated unions are powerful tools for enhancing type safety and flexibility in your code. By leveraging these features, you can create more reliable and maintainable applications, particularly when implementing design patterns. Remember to keep your type definitions clear and manageable, and embrace the journey of exploring these advanced types in your projects.

## Quiz Time!

{{< quizdown >}}

### What is a type guard in TypeScript?

- [x] A mechanism to narrow down the type of a variable within a conditional block.
- [ ] A way to declare variables in TypeScript.
- [ ] A method to convert types in TypeScript.
- [ ] A function to handle errors in TypeScript.

> **Explanation:** Type guards allow you to narrow down the type of a variable within a conditional block, ensuring type safety.

### Which of the following is a built-in type guard in TypeScript?

- [x] `typeof`
- [ ] `convert`
- [ ] `parse`
- [ ] `transform`

> **Explanation:** `typeof` is a built-in type guard used to check the type of primitive values.

### What is a discriminated union in TypeScript?

- [x] A union of types with a common discriminant property.
- [ ] A type that can be any value.
- [ ] A function that returns multiple types.
- [ ] A method to combine objects.

> **Explanation:** Discriminated unions are a union of types with a common discriminant property, enabling safer type definitions.

### How do discriminated unions enhance type safety?

- [x] By enabling exhaustive type checking.
- [ ] By allowing any type to be used.
- [ ] By converting types at runtime.
- [ ] By ignoring type errors.

> **Explanation:** Discriminated unions enable exhaustive type checking, ensuring all cases are handled.

### Which pattern can benefit from using discriminated unions?

- [x] State Pattern
- [x] Visitor Pattern
- [ ] Singleton Pattern
- [ ] Factory Pattern

> **Explanation:** Discriminated unions can be used in patterns like State and Visitor to handle different states or elements.

### What is the purpose of a type predicate in a custom type guard?

- [x] To indicate a successful type check.
- [ ] To convert a type to another type.
- [ ] To handle runtime errors.
- [ ] To declare a variable.

> **Explanation:** A type predicate in a custom type guard indicates a successful type check, allowing TypeScript to narrow the type.

### What is the `instanceof` operator used for?

- [x] To check if an object is an instance of a particular class.
- [ ] To convert a string to a number.
- [ ] To declare a variable.
- [ ] To handle errors.

> **Explanation:** `instanceof` is used to check if an object is an instance of a particular class.

### What is a potential issue with using type guards and discriminated unions?

- [x] They can introduce complexity into type definitions.
- [ ] They reduce type safety.
- [ ] They make code less readable.
- [ ] They are not supported in TypeScript.

> **Explanation:** Type guards and discriminated unions can introduce complexity into type definitions, making them harder to manage.

### What is the best practice for writing type guard functions?

- [x] Ensure they are clear and maintainable.
- [ ] Use them sparingly.
- [ ] Avoid using them in large projects.
- [ ] Write them without comments.

> **Explanation:** Type guard functions should be clear and maintainable, with descriptive names and comments.

### True or False: Discriminated unions are ideal for modeling complex data structures with multiple variants.

- [x] True
- [ ] False

> **Explanation:** Discriminated unions are ideal for modeling complex data structures with multiple variants, providing safer type definitions.

{{< /quizdown >}}
