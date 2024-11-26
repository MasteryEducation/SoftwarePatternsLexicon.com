---
canonical: "https://softwarepatternslexicon.com/patterns-rust/15/2"
title: "Writing Rust Libraries for Other Languages: Expanding Rust's Reach"
description: "Learn how to create Rust libraries that can be used from other programming languages, expanding the reach of Rust code with examples for Python, Node.js, and C#."
linkTitle: "15.2. Writing Rust Libraries for Other Languages"
tags:
- "Rust"
- "Interoperability"
- "PyO3"
- "Neon"
- "CSharp"
- "FFI"
- "Cross-Language"
- "Library"
date: 2024-11-25
type: docs
nav_weight: 152000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 15.2. Writing Rust Libraries for Other Languages

Creating Rust libraries that can be used from other programming languages is a powerful way to leverage Rust's performance and safety features across different ecosystems. This section will guide you through the process of exposing Rust functions and types to other languages, with examples for Python, Node.js, and C#. We will also discuss data conversion, packaging, distribution, and the performance benefits and limitations of using Rust in this manner.

### Introduction

Rust is known for its memory safety, concurrency, and performance. These features make it an attractive choice for building libraries that can be used in other programming languages. By writing Rust libraries for other languages, you can:

- **Leverage Rust's performance** in computationally intensive tasks.
- **Ensure memory safety** in applications that require high reliability.
- **Expand Rust's reach** by integrating it into existing ecosystems.

Let's explore how to create Rust libraries for Python, Node.js, and C#.

### Exposing Rust Functions and Types

To expose Rust functions and types to other languages, we need to create bindings that allow the target language to interact with Rust code. This involves:

1. **Defining a Foreign Function Interface (FFI)**: This is a way for a program written in one language to call functions or use services written in another language.
2. **Handling Data Conversion**: Converting data types between Rust and the target language.
3. **Packaging and Distribution**: Creating a package that can be easily distributed and used in the target language.

#### Rust and Python: Using PyO3

[PyO3](https://pyo3.rs/) is a library that allows you to write Python bindings for Rust code. It provides a way to call Rust functions from Python and vice versa.

**Example: Creating a Rust Library for Python**

Let's create a simple Rust library that provides a function to add two numbers.

```rust
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn my_rust_lib(py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "add")]
    fn add_py(a: i32, b: i32) -> PyResult<i32> {
        Ok(add(a, b))
    }
    Ok(())
}

/// Add two numbers.
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

**Building and Using the Library**

1. **Add PyO3 to your `Cargo.toml`:**

   ```toml
   [dependencies]
   pyo3 = { version = "0.15", features = ["extension-module"] }
   ```

2. **Build the library:**

   Use `maturin` or `setuptools-rust` to build the library as a Python extension module.

3. **Use the library in Python:**

   ```python
   import my_rust_lib

   result = my_rust_lib.add(3, 4)
   print(f"The result is {result}")
   ```

**Data Conversion**

PyO3 handles data conversion between Python and Rust automatically for many types. However, for complex types, you may need to implement custom conversion logic.

#### Rust and Node.js: Using Neon

[Neon](https://neon-bindings.com/) is a library for writing Node.js bindings in Rust. It allows you to create high-performance Node.js modules with Rust.

**Example: Creating a Rust Library for Node.js**

Let's create a simple Rust library that provides a function to multiply two numbers.

```rust
use neon::prelude::*;

fn multiply(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let a = cx.argument::<JsNumber>(0)?.value(&mut cx);
    let b = cx.argument::<JsNumber>(1)?.value(&mut cx);
    Ok(cx.number(a * b))
}

register_module!(mut cx, {
    cx.export_function("multiply", multiply)
});
```

**Building and Using the Library**

1. **Add Neon to your `Cargo.toml`:**

   ```toml
   [dependencies]
   neon = "0.9"
   ```

2. **Build the library:**

   Use `neon build` to compile the library as a Node.js addon.

3. **Use the library in Node.js:**

   ```javascript
   const myRustLib = require('./my_rust_lib');

   const result = myRustLib.multiply(3, 4);
   console.log(`The result is ${result}`);
   ```

**Data Conversion**

Neon provides utilities for converting between JavaScript and Rust types. For complex types, you may need to implement custom conversion logic.

#### Rust and C#: Using Unmanaged Code

To use Rust libraries in C#, you can expose Rust functions as C-style functions and use P/Invoke to call them from C#.

**Example: Creating a Rust Library for C#**

Let's create a simple Rust library that provides a function to subtract two numbers.

```rust
#[no_mangle]
pub extern "C" fn subtract(a: i32, b: i32) -> i32 {
    a - b
}
```

**Building and Using the Library**

1. **Build the library:**

   Use `cargo build --release` to compile the library as a shared library (`.dll` on Windows, `.so` on Linux, `.dylib` on macOS).

2. **Use the library in C#:**

   ```csharp
   using System;
   using System.Runtime.InteropServices;

   class Program
   {
       [DllImport("my_rust_lib")]
       private static extern int subtract(int a, int b);

       static void Main()
       {
           int result = subtract(7, 4);
           Console.WriteLine($"The result is {result}");
       }
   }
   ```

**Data Conversion**

When using P/Invoke, you need to ensure that the data types match between Rust and C#. For complex types, you may need to use `struct` or `class` to represent the data in C#.

### Packaging and Distribution

Once you have created your Rust library, you need to package it for distribution. This involves:

- **Creating a package** that includes the compiled library and any necessary metadata.
- **Distributing the package** through a package manager or as a standalone download.

For Python, you can use `maturin` or `setuptools-rust` to create a Python package. For Node.js, you can publish the library to npm. For C#, you can distribute the library as a NuGet package.

### Performance Benefits and Limitations

Using Rust libraries in other languages can provide significant performance benefits, especially for computationally intensive tasks. However, there are some limitations to consider:

- **Overhead**: There is some overhead associated with calling Rust functions from other languages, especially if data conversion is required.
- **Complexity**: Writing and maintaining bindings can be complex, especially for large libraries.
- **Compatibility**: Not all Rust features are easily exposed to other languages, and some languages may have limitations that affect how Rust code can be used.

### External Frameworks

- [PyO3 (Python bindings for Rust)](https://pyo3.rs/)
- [Neon (Rust bindings for Node.js)](https://neon-bindings.com/)
- [CXX (safe interop between Rust and C++)](https://cxx.rs/)

### Try It Yourself

Experiment with the examples provided by modifying the functions to perform different operations or by adding new functions. Try creating a Rust library that provides a more complex functionality, such as a data processing algorithm, and expose it to Python, Node.js, or C#.

### Visualizing Rust's Interaction with Other Languages

```mermaid
graph TD;
    A[Rust Library] --> B[Python (PyO3)];
    A --> C[Node.js (Neon)];
    A --> D[C# (P/Invoke)];
    B --> E[Python Application];
    C --> F[Node.js Application];
    D --> G[C# Application];
```

**Diagram Description**: This diagram illustrates how a Rust library can interact with different languages like Python, Node.js, and C#, enabling applications in these languages to leverage Rust's capabilities.

### Knowledge Check

- **What are the key steps in creating a Rust library for another language?**
- **How does PyO3 facilitate Rust and Python interoperability?**
- **What are the performance considerations when using Rust libraries in other languages?**

### Summary

Writing Rust libraries for other languages allows you to leverage Rust's performance and safety features across different ecosystems. By understanding how to expose Rust functions and types, handle data conversion, and package your library for distribution, you can create powerful, cross-language applications.

Remember, this is just the beginning. As you progress, you'll be able to create more complex libraries and integrate Rust into a wider range of applications. Keep experimenting, stay curious, and enjoy the journey!

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of creating Rust libraries for other languages?

- [x] To leverage Rust's performance and safety features in other ecosystems.
- [ ] To replace the target language entirely.
- [ ] To make Rust code more complex.
- [ ] To avoid using Rust's concurrency features.

> **Explanation:** The main goal is to utilize Rust's strengths, such as performance and safety, in other programming environments.

### Which library is used for creating Python bindings for Rust?

- [x] PyO3
- [ ] Neon
- [ ] CXX
- [ ] Serde

> **Explanation:** PyO3 is specifically designed for creating Python bindings for Rust.

### What is the role of Neon in Rust interoperability?

- [x] It allows writing Node.js bindings in Rust.
- [ ] It provides Python bindings for Rust.
- [ ] It is used for C# interoperability.
- [ ] It is a Rust testing framework.

> **Explanation:** Neon is used to create high-performance Node.js modules with Rust.

### How can Rust functions be exposed to C#?

- [x] By using P/Invoke to call C-style functions.
- [ ] By using PyO3.
- [ ] By using Neon.
- [ ] By using the Rust compiler directly.

> **Explanation:** P/Invoke is used to call C-style functions from C#.

### What is a key consideration when handling data conversion between Rust and other languages?

- [x] Ensuring data types match between Rust and the target language.
- [ ] Ignoring data types.
- [ ] Using only primitive types.
- [ ] Avoiding data conversion altogether.

> **Explanation:** Matching data types is crucial to ensure correct data handling and prevent errors.

### What tool can be used to build a Rust library as a Python extension module?

- [x] Maturin
- [ ] Cargo
- [ ] Neon
- [ ] P/Invoke

> **Explanation:** Maturin is used to build Rust libraries as Python extension modules.

### What is one limitation of using Rust libraries in other languages?

- [x] There is some overhead associated with calling Rust functions.
- [ ] Rust libraries cannot be used in other languages.
- [ ] Rust libraries are slower than native libraries.
- [ ] Rust libraries do not support concurrency.

> **Explanation:** There is some overhead, especially if data conversion is required.

### Which of the following is NOT a benefit of using Rust libraries in other languages?

- [ ] Leveraging Rust's performance
- [ ] Ensuring memory safety
- [x] Making Rust code more complex
- [ ] Expanding Rust's reach

> **Explanation:** The goal is to leverage Rust's strengths, not to increase complexity.

### What is the purpose of creating a package for a Rust library?

- [x] To distribute the library easily and include necessary metadata.
- [ ] To make the library more complex.
- [ ] To hide the library's functionality.
- [ ] To convert the library to another language.

> **Explanation:** Packaging allows for easy distribution and use in the target language.

### True or False: PyO3 automatically handles data conversion between Python and Rust for many types.

- [x] True
- [ ] False

> **Explanation:** PyO3 provides automatic data conversion for many common types, simplifying interoperability.

{{< /quizdown >}}
