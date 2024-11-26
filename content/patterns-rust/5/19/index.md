---
canonical: "https://softwarepatternslexicon.com/patterns-rust/5/19"
title: "Module Privacy Pattern in Rust: Encapsulation and Visibility"
description: "Explore Rust's module privacy pattern to encapsulate implementation details, expose public interfaces, and control visibility for better code organization."
linkTitle: "5.19. The Module Privacy Pattern"
tags:
- "Rust"
- "Module System"
- "Encapsulation"
- "Visibility"
- "API Design"
- "Code Organization"
- "pub"
- "Code Maintainability"
date: 2024-11-25
type: docs
nav_weight: 69000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 5.19. The Module Privacy Pattern

In Rust, the module privacy pattern is a powerful tool for controlling the visibility of code components, encapsulating implementation details, and exposing only the necessary parts of your codebase. This pattern is crucial for designing clean, maintainable, and robust APIs. Let's explore how Rust's module system can be leveraged to achieve these goals.

### Understanding Rust's Module System

Rust's module system is designed to help developers organize code into logical units. Modules can contain functions, structs, enums, traits, and other modules. By default, items in a module are private, meaning they are accessible only within the module itself. This default behavior encourages encapsulation and helps prevent unintended interactions between different parts of a codebase.

#### Key Concepts

- **Modules**: Containers for organizing code. They can be nested and can contain other modules.
- **Visibility**: Determines whether an item can be accessed from outside its module.
- **Encapsulation**: Hiding the internal details of a module, exposing only what is necessary.

### Controlling Visibility with `pub`, `pub(crate)`, and `pub(super)`

Rust provides several keywords to control the visibility of items within modules:

- **`pub`**: Makes an item public, allowing it to be accessed from outside the module.
- **`pub(crate)`**: Restricts visibility to the current crate, preventing access from other crates.
- **`pub(super)`**: Limits visibility to the parent module, useful for sharing items between sibling modules.

#### Example: Basic Module Structure

```rust
mod outer {
    pub mod inner {
        pub fn public_function() {
            println!("This function is public and accessible from outside the module.");
        }

        fn private_function() {
            println!("This function is private and cannot be accessed from outside the module.");
        }
    }
}

fn main() {
    outer::inner::public_function();
    // outer::inner::private_function(); // This line would cause a compile-time error
}
```

In this example, `public_function` is accessible from outside the `inner` module because it is marked with `pub`. However, `private_function` is not accessible because it is private by default.

### Structuring Modules to Hide Internal Details

When designing a module, it's essential to consider which parts of the module should be exposed and which should remain hidden. This decision impacts the module's API and its maintainability.

#### Example: Encapsulation with Modules

```rust
mod library {
    pub struct Book {
        title: String,
        author: String,
    }

    impl Book {
        pub fn new(title: &str, author: &str) -> Book {
            Book {
                title: title.to_string(),
                author: author.to_string(),
            }
        }

        pub fn title(&self) -> &str {
            &self.title
        }

        pub fn author(&self) -> &str {
            &self.author
        }
    }

    fn secret_function() {
        println!("This is a secret function that should not be exposed.");
    }
}

fn main() {
    let book = library::Book::new("1984", "George Orwell");
    println!("Title: {}, Author: {}", book.title(), book.author());
    // library::secret_function(); // This line would cause a compile-time error
}
```

In this example, the `Book` struct and its methods are public, allowing them to be used outside the `library` module. However, `secret_function` remains private, encapsulating the module's internal logic.

### Benefits of Module Privacy in API Design

Module privacy is a critical aspect of API design. By carefully controlling what is exposed, you can create a clean and intuitive API that is easy to use and maintain. Here are some benefits:

- **Encapsulation**: Hides implementation details, reducing the risk of breaking changes when internal logic is modified.
- **Clarity**: Exposes only the necessary parts of a module, making the API easier to understand.
- **Safety**: Prevents unintended interactions between different parts of a codebase, reducing the likelihood of bugs.

### Guidelines for Balancing Encapsulation with Accessibility

When designing a module, it's essential to strike a balance between encapsulation and accessibility. Here are some guidelines to help you achieve this balance:

1. **Expose Only What Is Necessary**: Limit the visibility of items to what is required for the module's intended use.
2. **Use `pub(crate)` and `pub(super)`**: These keywords provide more granular control over visibility, allowing you to expose items within a crate or to a parent module without making them fully public.
3. **Document Public APIs**: Clearly document the purpose and usage of public items to help users understand how to interact with the module.
4. **Refactor Regularly**: As your codebase evolves, revisit module boundaries and visibility to ensure they still align with your design goals.

### Rust Unique Features: Module Privacy

Rust's module privacy system is unique in its emphasis on safety and encapsulation. By default, items are private, encouraging developers to think carefully about what should be exposed. This approach contrasts with some other languages where items are public by default, leading to potential overexposure of internal details.

### Differences and Similarities with Other Patterns

The module privacy pattern is often compared to the facade pattern, which also involves controlling access to a subsystem. However, while the facade pattern focuses on providing a simplified interface to a complex system, the module privacy pattern emphasizes encapsulation and visibility control within a module.

### Try It Yourself

To deepen your understanding of the module privacy pattern, try modifying the examples provided:

- Change the visibility of `private_function` to `pub(crate)` and observe how it affects accessibility.
- Create a new module within `library` and experiment with `pub(super)` to share items between sibling modules.
- Refactor the `Book` struct to include private fields and public methods that manipulate these fields.

### Visualizing Module Privacy

To better understand how module privacy works, let's visualize the relationships between modules and their visibility:

```mermaid
graph TD;
    A[Outer Module] -->|pub| B[Inner Module];
    B -->|pub| C[Public Function];
    B -->|private| D[Private Function];
    A -->|pub(crate)| E[Crate-Level Function];
    A -->|pub(super)| F[Parent-Level Function];
```

In this diagram, the outer module contains an inner module with both public and private functions. Additionally, crate-level and parent-level functions demonstrate the use of `pub(crate)` and `pub(super)`.

### Knowledge Check

- What is the default visibility of items in a Rust module?
- How does the `pub(crate)` keyword differ from `pub`?
- Why is encapsulation important in API design?
- How can module privacy help prevent bugs in a codebase?

### Key Takeaways

- Rust's module system is designed to encourage encapsulation and control visibility.
- Use `pub`, `pub(crate)`, and `pub(super)` to manage item visibility effectively.
- Carefully consider which parts of a module should be exposed to create a clean and maintainable API.
- Regularly refactor module boundaries and visibility to align with evolving design goals.

Remember, mastering the module privacy pattern is just one step in your Rust journey. As you continue to explore Rust's features and patterns, you'll gain a deeper understanding of how to write clean, efficient, and maintainable code. Keep experimenting, stay curious, and enjoy the journey!

## Quiz Time!

{{< quizdown >}}

### What is the default visibility of items in a Rust module?

- [x] Private
- [ ] Public
- [ ] Protected
- [ ] Internal

> **Explanation:** By default, items in a Rust module are private, meaning they are accessible only within the module itself.

### Which keyword would you use to make an item visible only within the current crate?

- [ ] pub
- [x] pub(crate)
- [ ] pub(super)
- [ ] pub(in)

> **Explanation:** The `pub(crate)` keyword restricts visibility to the current crate, preventing access from other crates.

### How does the `pub(super)` keyword affect visibility?

- [ ] Makes an item public to all modules
- [x] Limits visibility to the parent module
- [ ] Restricts visibility to the current crate
- [ ] Makes an item private

> **Explanation:** The `pub(super)` keyword limits visibility to the parent module, allowing sharing between sibling modules.

### Why is encapsulation important in API design?

- [x] It hides implementation details and reduces the risk of breaking changes.
- [ ] It makes all items public for easier access.
- [ ] It allows for more complex APIs.
- [ ] It increases the number of dependencies.

> **Explanation:** Encapsulation hides implementation details, reducing the risk of breaking changes when internal logic is modified.

### What is a benefit of using `pub(crate)` over `pub`?

- [x] It provides more granular control over visibility.
- [ ] It makes items accessible from any module.
- [ ] It automatically documents the item.
- [ ] It increases the performance of the module.

> **Explanation:** `pub(crate)` provides more granular control over visibility, allowing you to expose items within a crate without making them fully public.

### How can module privacy help prevent bugs in a codebase?

- [x] By preventing unintended interactions between different parts of the codebase.
- [ ] By making all functions public for easier debugging.
- [ ] By increasing the number of global variables.
- [ ] By allowing direct access to all internal data.

> **Explanation:** Module privacy prevents unintended interactions between different parts of the codebase, reducing the likelihood of bugs.

### What is a similarity between the module privacy pattern and the facade pattern?

- [x] Both involve controlling access to a subsystem.
- [ ] Both make all items public.
- [ ] Both are used for performance optimization.
- [ ] Both require the use of `pub(crate)`.

> **Explanation:** Both the module privacy pattern and the facade pattern involve controlling access to a subsystem, though their focus differs.

### What is a key difference between `pub` and `pub(crate)`?

- [x] `pub` makes an item accessible from outside the crate, while `pub(crate)` restricts it to the current crate.
- [ ] `pub` is used for private items, while `pub(crate)` is for public items.
- [ ] `pub` is faster than `pub(crate)`.
- [ ] `pub(crate)` is used only for structs.

> **Explanation:** `pub` makes an item accessible from outside the crate, while `pub(crate)` restricts it to the current crate.

### How can you share items between sibling modules?

- [ ] Use `pub`
- [x] Use `pub(super)`
- [ ] Use `pub(crate)`
- [ ] Use `pub(in)`

> **Explanation:** Using `pub(super)` allows sharing items between sibling modules by limiting visibility to the parent module.

### True or False: Rust's module privacy system encourages developers to think carefully about what should be exposed.

- [x] True
- [ ] False

> **Explanation:** Rust's module privacy system, with its default private visibility, encourages developers to think carefully about what should be exposed.

{{< /quizdown >}}
