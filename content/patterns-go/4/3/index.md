---
linkTitle: "4.3 Error Handling Patterns"
title: "Error Handling Patterns in Go: Best Practices and Techniques"
description: "Explore error handling patterns in Go, including idiomatic practices, error wrapping, custom error types, and more for robust and maintainable code."
categories:
- Go Programming
- Software Design
- Error Handling
tags:
- Go
- Error Handling
- Best Practices
- Custom Errors
- Error Wrapping
date: 2024-10-25
type: docs
nav_weight: 430000
canonical: "https://softwarepatternslexicon.com/patterns-go/4/3"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 4.3 Error Handling Patterns

Error handling in Go is a critical aspect of writing robust and maintainable software. Unlike many other languages that use exceptions, Go employs a straightforward approach where errors are treated as values. This section explores various error handling patterns in Go, emphasizing idiomatic practices, error wrapping, custom error types, and more.

### Introduction to Error Handling in Go

Go's error handling philosophy is simple: functions return an error as the last return value, and it's the caller's responsibility to check and handle it. This approach promotes explicit error handling and makes the control flow clear.

### Idiomatic Error Handling

The idiomatic way to handle errors in Go is to return an error as the last return value of a function and immediately check it after the function call.

#### Example:

```go
package main

import (
    "fmt"
    "os"
)

func readFile(filename string) ([]byte, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        return nil, err
    }
    return data, nil
}

func main() {
    data, err := readFile("example.txt")
    if err != nil {
        fmt.Println("Error reading file:", err)
        return
    }
    fmt.Println("File content:", string(data))
}
```

In this example, the `readFile` function returns an error, which is immediately checked in the `main` function. This pattern ensures that errors are handled as close to their source as possible.

### Error Wrapping and Unwrapping

Go 1.13 introduced error wrapping, which allows adding context to errors while preserving the original error. This is done using the `fmt.Errorf` function with the `%w` verb.

#### Wrapping Errors:

```go
package main

import (
    "errors"
    "fmt"
)

func openFile(filename string) error {
    return fmt.Errorf("failed to open file %s: %w", filename, errors.New("file not found"))
}

func main() {
    err := openFile("example.txt")
    if err != nil {
        fmt.Println("Error:", err)
    }
}
```

#### Unwrapping Errors:

To retrieve the original error, use `errors.Unwrap` or `errors.Is`.

```go
package main

import (
    "errors"
    "fmt"
)

var ErrNotFound = errors.New("file not found")

func openFile(filename string) error {
    return fmt.Errorf("failed to open file %s: %w", filename, ErrNotFound)
}

func main() {
    err := openFile("example.txt")
    if errors.Is(err, ErrNotFound) {
        fmt.Println("The file was not found.")
    } else {
        fmt.Println("Error:", err)
    }
}
```

### Creating Custom Error Types

Custom error types can provide more detailed information about errors and enable type assertions.

#### Defining a Custom Error Type:

```go
package main

import (
    "fmt"
)

type MyError struct {
    Code    int
    Message string
}

func (e *MyError) Error() string {
    return fmt.Sprintf("Error %d: %s", e.Code, e.Message)
}

func doSomething() error {
    return &MyError{Code: 404, Message: "Resource not found"}
}

func main() {
    err := doSomething()
    if err != nil {
        if myErr, ok := err.(*MyError); ok {
            fmt.Printf("Custom error occurred: %s\n", myErr)
        } else {
            fmt.Println("An error occurred:", err)
        }
    }
}
```

### Sentinel Errors

Sentinel errors are predefined error values that can be compared using `errors.Is`.

#### Example:

```go
package main

import (
    "errors"
    "fmt"
)

var ErrNotFound = errors.New("not found")

func findResource(id string) error {
    return ErrNotFound
}

func main() {
    err := findResource("123")
    if errors.Is(err, ErrNotFound) {
        fmt.Println("Resource not found.")
    } else {
        fmt.Println("Error:", err)
    }
}
```

### Avoiding Panic

In Go, `panic` is reserved for unrecoverable errors, such as programmer mistakes or critical failures. For recoverable errors, prefer returning an error.

#### Example of Using Panic:

```go
package main

import (
    "fmt"
)

func divide(a, b int) int {
    if b == 0 {
        panic("division by zero")
    }
    return a / b
}

func main() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Println("Recovered from panic:", r)
        }
    }()
    fmt.Println(divide(4, 2))
    fmt.Println(divide(4, 0)) // This will cause a panic
}
```

### Best Practices for Error Handling in Go

1. **Handle Errors Immediately:** Check and handle errors right after a function call to maintain clarity.
2. **Wrap Errors for Context:** Use error wrapping to add context to errors, making them easier to debug.
3. **Use Custom Error Types:** Define custom error types for more detailed error information.
4. **Avoid Panic for Recoverable Errors:** Reserve panic for truly exceptional situations.
5. **Use Sentinel Errors Wisely:** Declare common errors as sentinel values and use `errors.Is` for comparison.

### Conclusion

Error handling in Go is straightforward yet powerful. By following idiomatic patterns, using error wrapping, and defining custom error types, developers can write robust and maintainable Go code. Avoiding panic for recoverable errors ensures that applications remain stable and predictable.

## Quiz Time!

{{< quizdown >}}

### What is the idiomatic way to handle errors in Go?

- [x] Return an error as the last return value and check it immediately.
- [ ] Use exceptions to handle errors.
- [ ] Ignore errors and proceed with execution.
- [ ] Use panic and recover for all error handling.

> **Explanation:** In Go, the idiomatic way to handle errors is to return them as the last return value of a function and check them immediately after the function call.

### How can you add context to an error in Go?

- [x] Use `fmt.Errorf` with the `%w` verb.
- [ ] Use `panic` to add context.
- [ ] Use `errors.New` to add context.
- [ ] Use `fmt.Print` to add context.

> **Explanation:** `fmt.Errorf` with the `%w` verb is used to wrap errors with additional context in Go.

### What function can you use to retrieve the original error from a wrapped error?

- [x] `errors.Unwrap`
- [ ] `errors.Wrap`
- [ ] `fmt.Errorf`
- [ ] `errors.New`

> **Explanation:** `errors.Unwrap` is used to retrieve the original error from a wrapped error.

### How do you define a custom error type in Go?

- [x] Implement the `Error()` method from the `error` interface.
- [ ] Use `errors.New` with a custom message.
- [ ] Use `fmt.Errorf` with a custom message.
- [ ] Use `panic` with a custom message.

> **Explanation:** To define a custom error type in Go, you need to implement the `Error()` method from the `error` interface.

### What is a sentinel error?

- [x] A predefined error value used for comparison.
- [ ] An error that causes a program to panic.
- [ ] An error that is ignored by the program.
- [ ] An error that is logged but not handled.

> **Explanation:** A sentinel error is a predefined error value that can be used for comparison using `errors.Is`.

### When should you use panic in Go?

- [x] For unrecoverable errors such as programmer mistakes or critical failures.
- [ ] For all types of errors.
- [ ] For errors that are expected to occur frequently.
- [ ] For errors that are handled by the caller.

> **Explanation:** Panic should be used for unrecoverable errors such as programmer mistakes or critical failures.

### How can you compare errors in Go to account for wrapped errors?

- [x] Use `errors.Is`
- [ ] Use `==` for direct equality.
- [ ] Use `errors.New`
- [ ] Use `fmt.Errorf`

> **Explanation:** `errors.Is` is used to compare errors, accounting for wrapped errors.

### What is the purpose of the `recover` function in Go?

- [x] To regain control after a panic.
- [ ] To handle errors returned by functions.
- [ ] To wrap errors with additional context.
- [ ] To compare errors for equality.

> **Explanation:** The `recover` function is used to regain control after a panic, allowing the program to continue execution.

### Why is it important to handle errors immediately in Go?

- [x] To maintain clarity and ensure errors are addressed as soon as they occur.
- [ ] To allow errors to propagate through the call stack.
- [ ] To avoid using the `recover` function.
- [ ] To prevent the use of custom error types.

> **Explanation:** Handling errors immediately maintains clarity and ensures that errors are addressed as soon as they occur.

### True or False: In Go, it is common to use exceptions for error handling.

- [ ] True
- [x] False

> **Explanation:** False. Go does not use exceptions for error handling; instead, it uses error values returned by functions.

{{< /quizdown >}}
