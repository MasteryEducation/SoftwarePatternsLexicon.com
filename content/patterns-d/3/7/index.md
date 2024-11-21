---
canonical: "https://softwarepatternslexicon.com/patterns-d/3/7"
title: "Memory Safety in D: Understanding `@safe`, `@trusted`, and `@system`"
description: "Explore memory safety in the D programming language with a focus on `@safe`, `@trusted`, and `@system` attributes. Learn how these attributes ensure safe, efficient, and reliable systems programming."
linkTitle: "3.7 Memory Safety: `@safe`, `@trusted`, and `@system`"
categories:
- Systems Programming
- Memory Safety
- D Programming Language
tags:
- Memory Safety
- D Language
- "@safe"
- "@trusted"
- "@system"
date: 2024-11-17
type: docs
nav_weight: 3700
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 3.7 Memory Safety: `@safe`, `@trusted`, and `@system`

Memory safety is a critical aspect of systems programming, ensuring that software can operate without unexpected behavior due to memory access errors. The D programming language provides a robust framework for memory safety through the use of attributes: `@safe`, `@trusted`, and `@system`. These attributes allow developers to explicitly define the safety level of their code, enabling both high-level safety guarantees and low-level control when necessary.

### Understanding Memory Safety in D

Memory safety in D is about preventing common programming errors such as buffer overflows, null pointer dereferences, and other forms of undefined behavior that can lead to security vulnerabilities and system crashes. D achieves this by categorizing functions into three safety levels:

- **`@safe`**: Functions that are guaranteed by the compiler to not perform any unsafe operations.
- **`@trusted`**: Functions that are marked as safe by the programmer, even if they contain operations that the compiler cannot verify as safe.
- **`@system`**: Functions that may perform unsafe operations and require the programmer to ensure their safety.

Let's delve deeper into each of these attributes to understand how they contribute to memory safety in D.

### `@safe` Functions: Ensuring Code Cannot Perform Unsafe Operations

`@safe` is the highest level of safety in D. When a function is marked as `@safe`, the compiler enforces strict checks to ensure that the function does not perform any operations that could lead to memory corruption or undefined behavior. This includes:

- **No Pointer Arithmetic**: `@safe` functions cannot perform pointer arithmetic, which can lead to accessing memory outside of intended bounds.
- **No Casting to Pointers**: Casting types to pointers is prohibited, as it can lead to invalid memory access.
- **No Inline Assembly**: Inline assembly is inherently unsafe and is not allowed in `@safe` functions.
- **No Access to Uninitialized Memory**: The compiler ensures that all memory is properly initialized before use.

Here's an example of a `@safe` function:

```d
@safe int add(int a, int b) {
    return a + b;
}
```

In this simple example, the function `add` is marked as `@safe`, meaning it performs no operations that could compromise memory safety.

#### Benefits of `@safe` Functions

- **Compiler Guarantees**: The compiler provides guarantees that `@safe` functions will not perform unsafe operations, reducing the risk of memory-related bugs.
- **Ease of Maintenance**: Code marked as `@safe` is easier to maintain and refactor, as developers can be confident in its safety.
- **Security**: By preventing unsafe operations, `@safe` functions contribute to the overall security of the software.

### `@trusted` Code: Marking Code as Safe Despite Potential Unsafe Operations

`@trusted` is a middle ground between `@safe` and `@system`. It allows developers to mark certain parts of their code as safe, even if they perform operations that the compiler cannot verify as safe. This is useful when the developer knows that the code is safe, but the compiler lacks the ability to prove it.

For example, consider a function that performs pointer arithmetic but is carefully designed to avoid accessing invalid memory:

```d
@trusted int sumArray(int* arr, size_t length) {
    int sum = 0;
    for (size_t i = 0; i < length; ++i) {
        sum += arr[i];
    }
    return sum;
}
```

In this example, the function `sumArray` is marked as `@trusted` because it performs pointer arithmetic, which is not allowed in `@safe` functions. However, the developer has ensured that the function does not access memory outside the bounds of the array.

#### When to Use `@trusted`

- **Performance**: When performance is critical, and the developer can ensure the safety of certain operations, `@trusted` can be used to bypass some of the restrictions of `@safe`.
- **Interfacing with External Libraries**: When interfacing with external libraries or system calls that are inherently unsafe but necessary, `@trusted` can be used to wrap these calls safely.
- **Legacy Code**: When working with legacy code that cannot be easily refactored to be `@safe`, `@trusted` can be used to mark the code as safe after careful review.

### `@system` Functions: Writing Lower-Level Code with Explicit Safety Considerations

`@system` is the default safety level in D and allows for the full range of operations, including those that are unsafe. This includes pointer arithmetic, casting, and inline assembly. `@system` functions are necessary for low-level programming tasks where the developer needs full control over memory and hardware.

Here's an example of a `@system` function:

```d
@system void manipulateMemory(void* ptr, size_t size) {
    // Perform low-level memory manipulation
    // Developer is responsible for ensuring safety
}
```

In this example, the function `manipulateMemory` is marked as `@system`, indicating that it may perform unsafe operations and that the developer is responsible for ensuring its safety.

#### Responsibilities of `@system` Functions

- **Manual Safety Checks**: Developers must manually ensure the safety of `@system` functions, as the compiler does not enforce any safety checks.
- **Documentation**: It is crucial to document the assumptions and safety guarantees of `@system` functions to aid future maintenance and review.
- **Testing**: Thorough testing is essential to ensure that `@system` functions do not introduce memory safety issues.

### Enforcing Safety: Compiler Checks and Their Significance in Systems Programming

The D compiler plays a crucial role in enforcing memory safety through its checks on `@safe`, `@trusted`, and `@system` functions. These checks help prevent common programming errors and ensure that code adheres to the specified safety level.

#### Compiler Checks for `@safe` Functions

- **Bounds Checking**: The compiler performs bounds checking to ensure that array accesses are within valid bounds.
- **Null Pointer Dereference Prevention**: The compiler prevents dereferencing null pointers in `@safe` functions.
- **Type Safety**: The compiler enforces type safety, preventing operations that could lead to type-related errors.

#### Compiler Checks for `@trusted` and `@system` Functions

- **No Automatic Checks**: The compiler does not perform automatic safety checks for `@trusted` and `@system` functions. It is the developer's responsibility to ensure safety.
- **Explicit Annotations**: Developers must explicitly annotate functions with `@trusted` or `@system` to indicate their safety level.

### Visualizing Memory Safety in D

To better understand the relationship between `@safe`, `@trusted`, and `@system`, let's visualize these concepts using a diagram:

```mermaid
graph TD;
    A[@safe] -->|Compiler Enforced| B[No Unsafe Operations];
    C[@trusted] -->|Developer Ensured| D[Potentially Unsafe Operations];
    E[@system] -->|Developer Managed| F[Unsafe Operations Allowed];
```

**Diagram Description**: This diagram illustrates the relationship between `@safe`, `@trusted`, and `@system` attributes in D. `@safe` functions are enforced by the compiler to have no unsafe operations. `@trusted` functions rely on the developer to ensure safety despite potentially unsafe operations. `@system` functions allow unsafe operations, with safety managed by the developer.

### Try It Yourself: Experimenting with Memory Safety

To deepen your understanding of memory safety in D, try modifying the following code examples:

1. **Convert a `@system` function to `@safe`**: Identify unsafe operations and refactor them to comply with `@safe` restrictions.
2. **Use `@trusted` to wrap unsafe operations**: Create a `@trusted` function that safely wraps a `@system` function, ensuring that all safety checks are manually performed.
3. **Test the Compiler's Safety Checks**: Write a `@safe` function that attempts to perform an unsafe operation and observe the compiler's response.

### References and Further Reading

- [D Language Specification: Function Attributes](https://dlang.org/spec/function.html)
- [Memory Safety in D](https://dlang.org/articles/memory.html)
- [SafeD: Memory Safety in the D Programming Language](https://dlang.org/safed.html)

### Knowledge Check

To reinforce your understanding of memory safety in D, consider the following questions:

- What are the key differences between `@safe`, `@trusted`, and `@system` functions?
- How does the D compiler enforce memory safety in `@safe` functions?
- When should you use `@trusted` instead of `@safe` or `@system`?
- What responsibilities do developers have when writing `@system` functions?

### Embrace the Journey

Remember, mastering memory safety in D is a journey. As you continue to explore and experiment with these concepts, you'll gain a deeper understanding of how to write safe, efficient, and reliable systems software. Keep experimenting, stay curious, and enjoy the journey!

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of `@safe` functions in D?

- [x] To ensure that the code cannot perform unsafe operations
- [ ] To allow low-level memory manipulation
- [ ] To mark code as safe despite potential unsafe operations
- [ ] To enable inline assembly

> **Explanation:** `@safe` functions are designed to ensure that the code cannot perform unsafe operations, providing compiler-enforced memory safety.

### Which attribute allows developers to mark potentially unsafe code as safe?

- [ ] `@safe`
- [x] `@trusted`
- [ ] `@system`
- [ ] `@secure`

> **Explanation:** `@trusted` allows developers to mark potentially unsafe code as safe, relying on the developer's assurance of safety.

### What is the default safety level for functions in D?

- [ ] `@safe`
- [ ] `@trusted`
- [x] `@system`
- [ ] `@secure`

> **Explanation:** `@system` is the default safety level in D, allowing for unsafe operations without compiler checks.

### What operations are prohibited in `@safe` functions?

- [x] Pointer arithmetic
- [ ] Function calls
- [ ] Variable declarations
- [ ] Loop constructs

> **Explanation:** Pointer arithmetic is prohibited in `@safe` functions to prevent unsafe memory access.

### When should you use `@trusted` functions?

- [x] When you can ensure safety despite compiler limitations
- [ ] When you need to perform inline assembly
- [ ] When you want to bypass all safety checks
- [ ] When you write high-level code

> **Explanation:** `@trusted` functions are used when the developer can ensure safety despite the compiler's inability to verify it.

### What is the role of the D compiler in `@safe` functions?

- [x] To enforce memory safety through checks
- [ ] To allow all operations without restrictions
- [ ] To disable type safety
- [ ] To perform inline assembly

> **Explanation:** The D compiler enforces memory safety in `@safe` functions through various checks, such as bounds checking and null pointer prevention.

### Which attribute allows for full control over memory and hardware?

- [ ] `@safe`
- [ ] `@trusted`
- [x] `@system`
- [ ] `@secure`

> **Explanation:** `@system` allows for full control over memory and hardware, enabling low-level programming tasks.

### What is a key responsibility of developers when writing `@system` functions?

- [x] Ensuring manual safety checks
- [ ] Relying on compiler checks
- [ ] Avoiding all unsafe operations
- [ ] Using only high-level constructs

> **Explanation:** Developers must ensure manual safety checks when writing `@system` functions, as the compiler does not enforce safety.

### What is the benefit of using `@safe` functions?

- [x] Compiler guarantees of no unsafe operations
- [ ] Ability to perform inline assembly
- [ ] Full control over memory
- [ ] Bypassing type safety

> **Explanation:** `@safe` functions provide compiler guarantees that no unsafe operations are performed, enhancing memory safety.

### True or False: `@trusted` functions are automatically checked by the compiler for safety.

- [ ] True
- [x] False

> **Explanation:** False. `@trusted` functions are not automatically checked by the compiler for safety; the developer must ensure their safety.

{{< /quizdown >}}
