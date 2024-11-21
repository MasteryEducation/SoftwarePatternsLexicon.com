---
linkTitle: "2.3.2 Command"
title: "Command Design Pattern in Go: Encapsulating Requests for Flexibility and Control"
description: "Explore the Command design pattern in Go, a powerful way to encapsulate requests as objects, enabling parameterization, queuing, and undoable operations."
categories:
- Design Patterns
- Go Programming
- Software Architecture
tags:
- Command Pattern
- Behavioral Patterns
- GoF Patterns
- Go Language
- Software Design
date: 2024-10-25
type: docs
nav_weight: 232000
canonical: "https://softwarepatternslexicon.com/patterns-go/2/3/2"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.3.2 Command

The Command design pattern is a behavioral pattern from the Gang of Four (GoF) collection that encapsulates a request as an object. This encapsulation allows for parameterization, queuing of requests, and supports undoable operations. In this article, we will explore the Command pattern in the context of Go, providing detailed explanations, code examples, and best practices.

### Understand the Intent

The primary intent of the Command pattern is to decouple the sender of a request from the object that executes the request. By encapsulating requests as objects, the pattern allows for:

- **Parameterization of Objects with Operations:** Commands can be passed around as first-class objects, enabling flexible invocation of operations.
- **Queuing and Logging of Requests:** Commands can be stored in a queue or log, facilitating batch processing and auditing.
- **Support for Undoable Operations:** By maintaining a history of executed commands, the pattern can support undo functionality.

### Implementation Steps

To implement the Command pattern in Go, follow these steps:

1. **Define a Command Interface:** Create an interface with an `Execute()` method that all command objects will implement.

2. **Implement Concrete Command Structs:** For each action, implement a concrete struct that adheres to the Command interface.

3. **Create an Invoker:** Develop an invoker that holds and calls the commands.

4. **Optionally Implement Undo Functionality:** If undo functionality is required, extend the Command interface to include an `Undo()` method.

### When to Use

Consider using the Command pattern in the following scenarios:

- **Parameterizing Objects with Operations:** When you need to pass operations as arguments to other objects.
- **Queuing or Logging Requests:** When you need to queue requests for later execution or log them for auditing purposes.
- **Supporting Undoable Operations:** When your application requires undo functionality.

### Go-Specific Tips

- **Use Function Types as Commands:** In Go, you can use function types to represent commands, simplifying the implementation.
- **Store Commands in Slices:** Utilize slices to store commands for batch processing or implementing undo functionality.

### Example: Text Editor Actions

Let's illustrate the Command pattern with a practical example involving a text editor where commands represent actions like copy and paste.

```go
package main

import "fmt"

// Command interface with Execute method
type Command interface {
    Execute()
}

// Concrete Command for Copy
type CopyCommand struct {
    editor *TextEditor
}

func (c *CopyCommand) Execute() {
    c.editor.Copy()
}

// Concrete Command for Paste
type PasteCommand struct {
    editor *TextEditor
}

func (p *PasteCommand) Execute() {
    p.editor.Paste()
}

// Receiver class
type TextEditor struct {
    clipboard string
    content   string
}

func (t *TextEditor) Copy() {
    t.clipboard = t.content
    fmt.Println("Copied content to clipboard:", t.clipboard)
}

func (t *TextEditor) Paste() {
    t.content += t.clipboard
    fmt.Println("Pasted content:", t.content)
}

// Invoker class
type CommandInvoker struct {
    history []Command
}

func (i *CommandInvoker) StoreAndExecute(cmd Command) {
    i.history = append(i.history, cmd)
    cmd.Execute()
}

func main() {
    editor := &TextEditor{content: "Hello, World!"}
    copyCmd := &CopyCommand{editor: editor}
    pasteCmd := &PasteCommand{editor: editor}

    invoker := &CommandInvoker{}
    invoker.StoreAndExecute(copyCmd)
    invoker.StoreAndExecute(pasteCmd)
}
```

### Explanation of the Example

- **Command Interface:** Defines the `Execute()` method that all commands must implement.
- **Concrete Commands:** `CopyCommand` and `PasteCommand` implement the `Command` interface, encapsulating the actions of copying and pasting.
- **Receiver:** `TextEditor` is the object that performs the actual operations.
- **Invoker:** `CommandInvoker` stores and executes commands, maintaining a history for potential undo functionality.

### Advantages and Disadvantages

**Advantages:**

- **Decoupling:** Separates the object that invokes the operation from the one that knows how to perform it.
- **Flexibility:** Commands can be parameterized and passed around easily.
- **Undo/Redo Support:** Facilitates implementing undoable operations.

**Disadvantages:**

- **Complexity:** Can introduce additional complexity with numerous command classes.
- **Overhead:** May lead to increased memory usage if many commands are stored.

### Best Practices

- **Use Function Types for Simplicity:** In Go, consider using function types to represent simple commands.
- **Batch Processing:** Store commands in slices for batch execution or undo functionality.
- **SOLID Principles:** Adhere to SOLID principles to ensure maintainable and scalable command implementations.

### Comparisons with Other Patterns

- **Strategy Pattern:** While both patterns encapsulate actions, the Command pattern focuses on requests and supports undo functionality, whereas the Strategy pattern is about selecting algorithms at runtime.
- **Chain of Responsibility:** The Command pattern encapsulates requests, while the Chain of Responsibility passes requests along a chain of handlers.

### Conclusion

The Command pattern is a powerful tool in Go for encapsulating requests as objects, providing flexibility, and supporting undoable operations. By following best practices and leveraging Go's unique features, developers can implement this pattern effectively to enhance their applications.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Command pattern?

- [x] To encapsulate a request as an object, allowing for parameterization and queuing of requests.
- [ ] To provide a simplified interface to a complex subsystem.
- [ ] To define a family of algorithms and make them interchangeable.
- [ ] To ensure a class has only one instance.

> **Explanation:** The Command pattern encapsulates a request as an object, enabling parameterization, queuing, and undoable operations.

### Which method is typically defined in the Command interface?

- [x] Execute()
- [ ] Run()
- [ ] Perform()
- [ ] Action()

> **Explanation:** The `Execute()` method is the standard method defined in the Command interface for executing the command.

### What is a common use case for the Command pattern?

- [x] Supporting undoable operations.
- [ ] Simplifying complex interfaces.
- [ ] Managing object creation.
- [ ] Structuring code into layers.

> **Explanation:** The Command pattern is often used to support undoable operations by maintaining a history of executed commands.

### In Go, how can commands be represented for simplicity?

- [x] Using function types.
- [ ] Using channels.
- [ ] Using interfaces only.
- [ ] Using structs without methods.

> **Explanation:** In Go, function types can be used to represent commands, simplifying the implementation.

### What is the role of the invoker in the Command pattern?

- [x] To store and call the commands.
- [ ] To perform the actual operations.
- [ ] To define the command interface.
- [ ] To encapsulate the request.

> **Explanation:** The invoker stores and calls the commands, often maintaining a history for undo functionality.

### Which of the following is a disadvantage of the Command pattern?

- [x] It can introduce additional complexity with numerous command classes.
- [ ] It tightly couples the sender and receiver.
- [ ] It limits the flexibility of operations.
- [ ] It cannot support undo functionality.

> **Explanation:** The Command pattern can introduce complexity due to the creation of numerous command classes.

### How does the Command pattern differ from the Strategy pattern?

- [x] The Command pattern focuses on requests and supports undo functionality, while the Strategy pattern is about selecting algorithms.
- [ ] The Command pattern simplifies interfaces, while the Strategy pattern encapsulates requests.
- [ ] The Command pattern manages object creation, while the Strategy pattern provides a simplified interface.
- [ ] The Command pattern ensures a single instance, while the Strategy pattern supports undo functionality.

> **Explanation:** The Command pattern encapsulates requests and supports undo functionality, whereas the Strategy pattern is about selecting algorithms at runtime.

### What is a benefit of using the Command pattern?

- [x] It decouples the object that invokes the operation from the one that knows how to perform it.
- [ ] It simplifies the interface of a complex subsystem.
- [ ] It ensures a class has only one instance.
- [ ] It allows for the dynamic selection of algorithms.

> **Explanation:** The Command pattern decouples the invoker from the executor, providing flexibility and supporting undo operations.

### Which of the following is a Go-specific tip for implementing the Command pattern?

- [x] Store commands in slices for batch processing or undo functionality.
- [ ] Use channels to represent commands.
- [ ] Avoid using interfaces for commands.
- [ ] Implement commands as global variables.

> **Explanation:** In Go, storing commands in slices allows for batch processing and implementing undo functionality.

### The Command pattern can be used to queue requests for later execution.

- [x] True
- [ ] False

> **Explanation:** True. The Command pattern allows requests to be queued for later execution, facilitating batch processing and undo functionality.

{{< /quizdown >}}
