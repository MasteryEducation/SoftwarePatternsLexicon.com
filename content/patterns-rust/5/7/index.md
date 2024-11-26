---
canonical: "https://softwarepatternslexicon.com/patterns-rust/5/7"
title: "Understanding Rust's `Deref` and `DerefMut` Traits for Smart Pointers"
description: "Explore the `Deref` and `DerefMut` traits in Rust, which enable smart pointers to behave like regular references, facilitating seamless data access."
linkTitle: "5.7. The `Deref` and `DerefMut` Traits"
tags:
- "Rust"
- "Deref"
- "DerefMut"
- "Smart Pointers"
- "Dereferencing"
- "Rust Programming"
- "Rust Traits"
- "Dereference Coercion"
date: 2024-11-25
type: docs
nav_weight: 57000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 5.7. The `Deref` and `DerefMut` Traits

In Rust, the `Deref` and `DerefMut` traits play a crucial role in enabling smart pointers to behave like regular references. This capability is essential for seamless access to the data they point to, allowing developers to write more intuitive and flexible code. In this section, we will delve into the purpose and functionality of these traits, explore how they enable dereferencing logic and conversion, and provide examples of implementing these traits for custom smart pointers. We will also discuss the concept of deref coercion and highlight scenarios where implementing `Deref` and `DerefMut` is particularly beneficial.

### Purpose of the `Deref` and `DerefMut` Traits

The `Deref` and `DerefMut` traits are part of Rust's standard library and are used to customize the behavior of the dereference operator (`*`). By implementing these traits, you can define how a smart pointer should behave when it is dereferenced. This is particularly useful for types that act as smart pointers, such as `Box`, `Rc`, and `Arc`, allowing them to provide access to the data they encapsulate as if they were regular references.

#### The `Deref` Trait

The `Deref` trait is used to override the behavior of the dereference operator for immutable references. It allows you to specify how a type should be dereferenced to yield a reference to its inner data. Here's the definition of the `Deref` trait:

```rust
trait Deref {
    type Target: ?Sized;

    fn deref(&self) -> &Self::Target;
}
```

- **`type Target`**: This associated type specifies the type of the data that the smart pointer points to.
- **`fn deref(&self) -> &Self::Target`**: This method returns a reference to the data, allowing the smart pointer to be used like a regular reference.

#### The `DerefMut` Trait

The `DerefMut` trait is similar to `Deref`, but it is used for mutable references. It allows you to define how a smart pointer should be dereferenced to yield a mutable reference to its inner data. Here's the definition of the `DerefMut` trait:

```rust
trait DerefMut: Deref {
    fn deref_mut(&mut self) -> &mut Self::Target;
}
```

- **`fn deref_mut(&mut self) -> &mut Self::Target`**: This method returns a mutable reference to the data, enabling the smart pointer to be used like a mutable reference.

### Enabling Dereferencing Logic and Conversion

By implementing the `Deref` and `DerefMut` traits, you can control how your custom types behave when they are dereferenced. This is particularly useful for smart pointers, which encapsulate data and provide additional functionality, such as reference counting or memory management.

#### Example: Implementing `Deref` for a Custom Smart Pointer

Let's implement the `Deref` trait for a simple custom smart pointer called `MyBox`. This smart pointer will wrap a value and provide access to it through the dereference operator.

```rust
use std::ops::Deref;

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x: T) -> MyBox<T> {
        MyBox(x)
    }
}

impl<T> Deref for MyBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

fn main() {
    let x = 5;
    let y = MyBox::new(x);

    assert_eq!(5, *y);
}
```

In this example, we define a `MyBox` struct that wraps a value of type `T`. By implementing the `Deref` trait, we specify that dereferencing a `MyBox<T>` should yield a reference to the inner value `T`. This allows us to use the `*` operator to access the value inside `MyBox`.

#### Example: Implementing `DerefMut` for a Custom Smart Pointer

Now, let's extend our `MyBox` example to support mutable dereferencing by implementing the `DerefMut` trait.

```rust
use std::ops::{Deref, DerefMut};

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x: T) -> MyBox<T> {
        MyBox(x)
    }
}

impl<T> Deref for MyBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for MyBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

fn main() {
    let mut x = MyBox::new(5);
    *x = 10;
    assert_eq!(10, *x);
}
```

By implementing `DerefMut`, we allow `MyBox` to be dereferenced mutably, enabling us to modify the value it contains.

### The Concept of Deref Coercion

Deref coercion is a powerful feature in Rust that automatically converts a reference to a type implementing `Deref` into a reference to its `Target` type. This conversion happens implicitly when a function or method expects a reference to the `Target` type. Deref coercion simplifies code by allowing smart pointers to be used in place of regular references without explicit dereferencing.

#### Example: Deref Coercion in Action

Consider the following example, where deref coercion allows a `MyBox<String>` to be used as a `&str`:

```rust
use std::ops::Deref;

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x: T) -> MyBox<T> {
        MyBox(x)
    }
}

impl<T> Deref for MyBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

fn hello(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    let m = MyBox::new(String::from("Rust"));
    hello(&m);
}
```

In this example, the `hello` function expects a `&str`, but we pass a `&MyBox<String>`. Deref coercion automatically converts `&MyBox<String>` to `&String`, and then to `&str`, allowing the code to compile and run successfully.

### Benefits of Deref Coercion

Deref coercion provides several benefits:

- **Simplicity**: It reduces the need for explicit dereferencing, making code cleaner and easier to read.
- **Flexibility**: It allows smart pointers to be used interchangeably with regular references in many contexts.
- **Interoperability**: It enables seamless integration with existing APIs that expect references.

### Scenarios for Implementing `Deref` and `DerefMut`

Implementing `Deref` and `DerefMut` is particularly useful in the following scenarios:

- **Custom Smart Pointers**: When creating custom smart pointers, implementing these traits allows them to behave like regular references, providing a familiar interface to users.
- **Wrapper Types**: If you have a type that wraps another type and you want to expose the inner type's functionality, implementing `Deref` and `DerefMut` can make the wrapper type more convenient to use.
- **Interfacing with Libraries**: When integrating with libraries that expect references, implementing these traits can simplify the interaction by allowing your types to be used directly.

### Design Considerations

When implementing `Deref` and `DerefMut`, consider the following:

- **Safety**: Ensure that the dereferencing logic is safe and does not lead to undefined behavior.
- **Consistency**: The behavior of `deref` and `deref_mut` should be consistent with the semantics of the type.
- **Performance**: Be mindful of performance implications, especially if dereferencing involves complex logic.

### Rust Unique Features

Rust's ownership and borrowing system, combined with the `Deref` and `DerefMut` traits, provide a powerful mechanism for managing memory safely and efficiently. The language's emphasis on safety and performance makes these traits an integral part of idiomatic Rust programming.

### Differences and Similarities

The `Deref` and `DerefMut` traits are similar in that they both enable dereferencing logic, but they differ in their mutability. `Deref` is used for immutable references, while `DerefMut` is used for mutable references. Understanding this distinction is crucial for implementing these traits correctly.

### Try It Yourself

To deepen your understanding of the `Deref` and `DerefMut` traits, try modifying the examples provided:

- Implement a custom smart pointer that wraps a vector and allows access to its elements through dereferencing.
- Experiment with deref coercion by creating functions that expect different types of references and see how Rust handles the conversions.

### Visualizing Deref Coercion

To better understand how deref coercion works, let's visualize the process using a flowchart.

```mermaid
graph TD;
    A[&MyBox<String>] --> B[&String];
    B --> C[&str];
    D[hello(&str)] --> C;
```

**Figure 1**: This flowchart illustrates how a `&MyBox<String>` is coerced into a `&str` through deref coercion, allowing it to be passed to the `hello` function.

### Knowledge Check

Before moving on, consider the following questions to test your understanding:

- What is the primary purpose of the `Deref` trait in Rust?
- How does deref coercion simplify code in Rust?
- In what scenarios would you implement the `DerefMut` trait?

### Embrace the Journey

Remember, mastering the `Deref` and `DerefMut` traits is just one step in your Rust journey. As you continue to explore the language, you'll discover more powerful features and patterns that will enhance your programming skills. Keep experimenting, stay curious, and enjoy the process!

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of the `Deref` trait in Rust?

- [x] To customize the behavior of the dereference operator for immutable references.
- [ ] To enable mutable references to be dereferenced.
- [ ] To manage memory allocation for smart pointers.
- [ ] To provide a default implementation for the `Drop` trait.

> **Explanation:** The `Deref` trait is used to customize how a type is dereferenced, specifically for immutable references.

### How does deref coercion benefit Rust developers?

- [x] It simplifies code by reducing the need for explicit dereferencing.
- [ ] It increases the performance of smart pointers.
- [ ] It allows for automatic memory management.
- [ ] It provides a way to implement custom smart pointers.

> **Explanation:** Deref coercion simplifies code by allowing smart pointers to be used in place of regular references without explicit dereferencing.

### Which trait should you implement to allow mutable dereferencing of a custom smart pointer?

- [ ] `Deref`
- [x] `DerefMut`
- [ ] `Drop`
- [ ] `Clone`

> **Explanation:** The `DerefMut` trait is implemented to allow mutable dereferencing of a custom smart pointer.

### What is the associated type `Target` used for in the `Deref` trait?

- [x] It specifies the type of the data that the smart pointer points to.
- [ ] It defines the size of the smart pointer.
- [ ] It indicates whether the smart pointer is mutable.
- [ ] It provides a default value for the smart pointer.

> **Explanation:** The `Target` associated type specifies the type of data that the smart pointer points to.

### In which scenario is deref coercion particularly useful?

- [x] When a function expects a reference to a type and a smart pointer is passed.
- [ ] When implementing memory management for smart pointers.
- [ ] When creating a new type of smart pointer.
- [ ] When performing arithmetic operations on smart pointers.

> **Explanation:** Deref coercion is useful when a function expects a reference to a type, and a smart pointer is passed, allowing for automatic conversion.

### What is the main difference between `Deref` and `DerefMut`?

- [x] `Deref` is for immutable references, while `DerefMut` is for mutable references.
- [ ] `Deref` is for memory allocation, while `DerefMut` is for memory deallocation.
- [ ] `Deref` is for smart pointers, while `DerefMut` is for regular pointers.
- [ ] `Deref` is for performance optimization, while `DerefMut` is for safety.

> **Explanation:** The main difference is that `Deref` is used for immutable references, while `DerefMut` is used for mutable references.

### Which of the following is a benefit of implementing `Deref` for a custom type?

- [x] It allows the custom type to be used like a regular reference.
- [ ] It automatically manages memory allocation.
- [ ] It provides thread safety for the custom type.
- [ ] It enables the custom type to be cloned.

> **Explanation:** Implementing `Deref` allows the custom type to be used like a regular reference, providing a familiar interface.

### What is the result of deref coercion when a `&MyBox<String>` is passed to a function expecting a `&str`?

- [x] The `&MyBox<String>` is automatically converted to a `&str`.
- [ ] The function will fail to compile.
- [ ] The `&MyBox<String>` is converted to a `&String`.
- [ ] The `&MyBox<String>` is cloned.

> **Explanation:** Deref coercion automatically converts `&MyBox<String>` to `&str`, allowing it to be passed to the function.

### Which method must be implemented for the `Deref` trait?

- [x] `fn deref(&self) -> &Self::Target`
- [ ] `fn deref_mut(&mut self) -> &mut Self::Target`
- [ ] `fn drop(&mut self)`
- [ ] `fn clone(&self) -> Self`

> **Explanation:** The `deref` method must be implemented for the `Deref` trait to specify how the type should be dereferenced.

### True or False: Deref coercion can convert a mutable reference to an immutable reference.

- [x] True
- [ ] False

> **Explanation:** Deref coercion can convert a mutable reference to an immutable reference, but not the other way around.

{{< /quizdown >}}
