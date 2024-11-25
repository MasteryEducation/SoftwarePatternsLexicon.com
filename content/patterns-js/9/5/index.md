---
linkTitle: "9.5 Mapped Types"
title: "Understanding and Implementing Mapped Types in TypeScript"
description: "Explore the concept of mapped types in TypeScript, learn how to transform existing types, and see practical examples and use cases."
categories:
- TypeScript
- Design Patterns
- Programming
tags:
- TypeScript
- Mapped Types
- Type Transformation
- Utility Types
- JavaScript
date: 2024-10-25
type: docs
nav_weight: 950000
canonical: "https://softwarepatternslexicon.com/patterns-js/9/5"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 9.5 Mapped Types

### Introduction

Mapped types in TypeScript are a powerful feature that allows developers to transform existing types into new ones by applying a transformation to each property. This capability is essential for creating flexible and reusable code, enabling developers to modify types without rewriting them. In this article, we will delve into the concept of mapped types, explore their syntax, and provide practical examples and use cases.

### Understanding the Concept

Mapped types transform existing types into new types by iterating over each property and applying a transformation. The syntax `{ [P in K]: T }` is used, where `P` iterates over property keys `K`. This allows for the creation of utility types that can modify properties, such as making them optional, readonly, or nullable.

### Implementation Steps

#### Create Basic Mapped Types

To create a basic mapped type, you define a type that modifies the properties of an existing type. For example, you can create a type that makes all properties optional:

```typescript
type Partial<T> = { [P in keyof T]?: T[P] };
```

#### Use `keyof` Operator

The `keyof` operator is used to obtain a union of all property names of a type. For example, `keyof T` returns the keys of type `T`.

#### Apply Modifiers

Modifiers like `readonly` or `optional` can be added or removed using `+` or `-`. For example, to remove the `readonly` modifier:

```typescript
type Mutable<T> = { -readonly [P in keyof T]: T[P] };
```

### Code Examples

#### Readonly Type

The `Readonly` type makes all properties of a type immutable:

```typescript
type Readonly<T> = { readonly [P in keyof T]: T[P] };
```

#### Pick Properties Type

The `Pick` type creates a new type by selecting specific properties from an existing type:

```typescript
type Pick<T, K extends keyof T> = { [P in K]: T[P] };
```

### Use Cases

Mapped types are useful for modifying existing types without rewriting them. They are commonly used to create utility types for common transformations, such as making properties optional, readonly, or nullable.

### Practice

#### Exercise 1: Define a `Required<T>` Type

Create a type `Required<T>` that makes all properties non-optional:

```typescript
type Required<T> = { [P in keyof T]-?: T[P] };
```

#### Exercise 2: Create a `Nullable<T>` Type

Define a type `Nullable<T>` that makes all properties nullable:

```typescript
type Nullable<T> = { [P in keyof T]: T[P] | null };
```

### Considerations

When working with mapped types, it's important to be mindful of how modifiers affect property behavior. Ensure that transformations are appropriate for the intended use and that they do not introduce unintended side effects.

### Advanced Topics

#### Domain-Driven Design (DDD) Integration

Mapped types can be integrated into Domain-Driven Design (DDD) by creating types that represent domain concepts and applying transformations to adapt them to different contexts.

#### Event Sourcing

In event sourcing architectures, mapped types can be used to transform event types into different representations, such as projections or views.

### Comparative Analysis

Mapped types can be compared with other type transformation techniques, such as interfaces and type aliases. While interfaces provide a way to define contracts, mapped types offer more flexibility in transforming existing types.

### Performance Considerations

Mapped types do not impact runtime performance, as they are a compile-time feature. However, complex transformations can increase compile time, so it's important to balance flexibility with simplicity.

### Conclusion

Mapped types are a versatile feature in TypeScript that enable developers to transform existing types into new ones. By understanding their syntax and use cases, you can create flexible and reusable code that adapts to changing requirements. Explore the exercises and examples provided to deepen your understanding and apply mapped types effectively in your projects.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of mapped types in TypeScript?

- [x] To transform existing types into new types by applying a transformation to each property.
- [ ] To create new types from scratch.
- [ ] To enforce strict type checking.
- [ ] To optimize runtime performance.

> **Explanation:** Mapped types are used to transform existing types into new types by applying a transformation to each property.

### Which syntax is used to define a mapped type in TypeScript?

- [x] `{ [P in K]: T }`
- [ ] `{ P: K }`
- [ ] `{ T[P]: K }`
- [ ] `{ K[P]: T }`

> **Explanation:** The syntax `{ [P in K]: T }` is used to define a mapped type, where `P` iterates over property keys `K`.

### What does the `keyof` operator do in TypeScript?

- [x] It obtains a union of all property names of a type.
- [ ] It creates a new type from existing properties.
- [ ] It removes properties from a type.
- [ ] It adds new properties to a type.

> **Explanation:** The `keyof` operator obtains a union of all property names of a type.

### How can you make all properties of a type optional using mapped types?

- [x] `type Partial<T> = { [P in keyof T]?: T[P] };`
- [ ] `type Partial<T> = { readonly [P in keyof T]: T[P] };`
- [ ] `type Partial<T> = { [P in keyof T]-?: T[P] };`
- [ ] `type Partial<T> = { [P in keyof T]: T[P] | null };`

> **Explanation:** The `Partial<T>` type makes all properties optional by using the `?` modifier.

### Which mapped type makes all properties of a type immutable?

- [x] `type Readonly<T> = { readonly [P in keyof T]: T[P] };`
- [ ] `type Mutable<T> = { -readonly [P in keyof T]: T[P] };`
- [ ] `type Required<T> = { [P in keyof T]-?: T[P] };`
- [ ] `type Nullable<T> = { [P in keyof T]: T[P] | null };`

> **Explanation:** The `Readonly<T>` type makes all properties immutable by using the `readonly` modifier.

### What is the result of using the `Pick` type in TypeScript?

- [x] It creates a new type by selecting specific properties from an existing type.
- [ ] It makes all properties of a type optional.
- [ ] It removes the `readonly` modifier from all properties.
- [ ] It makes all properties of a type nullable.

> **Explanation:** The `Pick` type creates a new type by selecting specific properties from an existing type.

### How can you make all properties of a type non-optional using mapped types?

- [x] `type Required<T> = { [P in keyof T]-?: T[P] };`
- [ ] `type Partial<T> = { [P in keyof T]?: T[P] };`
- [ ] `type Readonly<T> = { readonly [P in keyof T]: T[P] };`
- [ ] `type Nullable<T> = { [P in keyof T]: T[P] | null };`

> **Explanation:** The `Required<T>` type makes all properties non-optional by using the `-?` modifier.

### What does the `Nullable` type do in TypeScript?

- [x] It makes all properties of a type nullable.
- [ ] It makes all properties of a type optional.
- [ ] It makes all properties of a type readonly.
- [ ] It removes properties from a type.

> **Explanation:** The `Nullable<T>` type makes all properties nullable by adding `| null` to each property type.

### Which of the following is a consideration when using mapped types?

- [x] Be mindful of how modifiers affect property behavior.
- [ ] Ensure transformations are always complex.
- [ ] Avoid using the `keyof` operator.
- [ ] Use mapped types only for runtime performance optimization.

> **Explanation:** When using mapped types, it's important to be mindful of how modifiers affect property behavior to ensure transformations are appropriate.

### Mapped types impact runtime performance in TypeScript.

- [ ] True
- [x] False

> **Explanation:** Mapped types do not impact runtime performance as they are a compile-time feature.

{{< /quizdown >}}
