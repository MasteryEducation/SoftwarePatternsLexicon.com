---
canonical: "https://softwarepatternslexicon.com/patterns-erlang/9/6"

title: "Composite Pattern with Nested Data Structures in Erlang"
description: "Explore the Composite Pattern in Erlang using nested data structures like lists and tuples to represent hierarchical information."
linkTitle: "9.6 Composite Pattern with Nested Data Structures"
categories:
- Erlang Design Patterns
- Functional Programming
- Concurrent Programming
tags:
- Composite Pattern
- Nested Data Structures
- Erlang
- Functional Programming
- Hierarchical Data
date: 2024-11-23
type: docs
nav_weight: 96000
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"

---

## 9.6 Composite Pattern with Nested Data Structures

In this section, we delve into the Composite Pattern, a structural design pattern that allows you to compose objects into tree structures to represent part-whole hierarchies. Erlang, with its powerful pattern matching and functional programming capabilities, provides a unique approach to implementing this pattern using nested data structures like lists and tuples. 

### Understanding the Composite Pattern

The Composite Pattern is a design pattern used to treat individual objects and compositions of objects uniformly. This is particularly useful when you have a hierarchy of objects, and you want to perform operations on both individual objects and composites of objects in a consistent manner.

#### Key Participants

- **Component**: An interface for all objects in the composition, both leaf and composite nodes.
- **Leaf**: Represents leaf objects in the composition. A leaf has no children.
- **Composite**: Represents a composite node that can have children. It implements child-related operations.

### Intent of the Composite Pattern

The primary intent of the Composite Pattern is to allow clients to treat individual objects and compositions of objects uniformly. This pattern is particularly useful when dealing with tree structures, such as file systems, organizational hierarchies, or UI components.

### Applicability

Use the Composite Pattern when:

- You want to represent part-whole hierarchies of objects.
- You want clients to be able to ignore the difference between compositions of objects and individual objects.

### Erlang's Unique Approach

Erlang's functional nature and its support for pattern matching make it an excellent fit for implementing the Composite Pattern. Instead of using classes and interfaces, Erlang leverages its powerful data structures, such as lists and tuples, to represent hierarchical data.

### Implementing the Composite Pattern in Erlang

Let's explore how we can implement the Composite Pattern in Erlang using nested data structures. We'll use lists and tuples to represent a simple tree structure.

#### Example: Representing a File System

Consider a file system where directories can contain files and other directories. We can represent this hierarchy using tuples and lists in Erlang.

```erlang
% Define a file as a tuple with its name and size
-type file() :: {file, string(), integer()}.

% Define a directory as a tuple with its name and a list of contents
-type directory() :: {directory, string(), [file_or_directory()]}.

% Define a file or directory
-type file_or_directory() :: file() | directory().

% Example file system
FileSystem = {directory, "root", [
    {file, "file1.txt", 100},
    {directory, "subdir", [
        {file, "file2.txt", 200},
        {file, "file3.txt", 300}
    ]}
]}.
```

In this example, a file is represented as a tuple `{file, Name, Size}`, and a directory is represented as a tuple `{directory, Name, Contents}`, where `Contents` is a list of files or directories.

### Operating on the Composite Structure

One of the strengths of the Composite Pattern is the ability to perform operations on both individual elements and composites uniformly. In Erlang, we can achieve this using recursive functions.

#### Calculating Total Size

Let's write a function to calculate the total size of all files in the file system.

```erlang
% Calculate the total size of a file system
-spec total_size(file_or_directory()) -> integer().
total_size({file, _, Size}) ->
    Size;
total_size({directory, _, Contents}) ->
    lists:sum([total_size(Item) || Item <- Contents]).
```

In this function, we use pattern matching to differentiate between files and directories. For a file, we simply return its size. For a directory, we recursively calculate the total size of its contents.

### Visualizing the Composite Pattern

To better understand the structure of our file system, let's visualize it using a tree diagram.

```mermaid
graph TD;
    A[Root Directory] --> B[File: file1.txt (100)]
    A --> C[Subdirectory: subdir]
    C --> D[File: file2.txt (200)]
    C --> E[File: file3.txt (300)]
```

This diagram illustrates the hierarchical structure of our file system, with directories containing files and other directories.

### Benefits of Using the Composite Pattern in Erlang

The Composite Pattern, when implemented using Erlang's nested data structures, offers several advantages:

- **Simplicity**: Erlang's pattern matching and recursion make it easy to traverse and manipulate hierarchical data.
- **Uniformity**: Operations can be applied uniformly to both individual elements and composites.
- **Flexibility**: The pattern can be easily extended to accommodate additional types of components.

### Scenarios for Simplified Data Processing

The Composite Pattern is particularly useful in scenarios where you need to process hierarchical data structures. Some common use cases include:

- **File Systems**: Navigating and manipulating file and directory structures.
- **Organizational Hierarchies**: Representing and processing organizational charts.
- **UI Components**: Managing nested UI components in a graphical application.

### Try It Yourself

Experiment with the code examples provided by modifying the file system structure or adding new operations. For instance, try adding a function to count the number of files in the file system or to find the largest file.

### Design Considerations

When implementing the Composite Pattern in Erlang, consider the following:

- **Data Structure Choice**: Choose the appropriate data structure (lists, tuples, maps) based on your specific use case.
- **Performance**: Be mindful of the performance implications of recursive operations on large data structures.
- **Error Handling**: Implement robust error handling to manage unexpected data structures.

### Erlang Unique Features

Erlang's pattern matching and recursion are particularly well-suited for implementing the Composite Pattern. These features allow for concise and expressive code that can handle complex hierarchical data structures.

### Differences and Similarities

The Composite Pattern is often compared to other structural patterns, such as the Decorator Pattern. While both patterns deal with object composition, the Composite Pattern focuses on part-whole hierarchies, whereas the Decorator Pattern focuses on adding responsibilities to objects.

### Summary

The Composite Pattern is a powerful tool for managing hierarchical data structures in Erlang. By leveraging Erlang's functional programming features, we can implement this pattern in a way that is both simple and effective. Remember, this is just the beginning. As you progress, you'll build more complex and interactive applications. Keep experimenting, stay curious, and enjoy the journey!

## Quiz: Composite Pattern with Nested Data Structures

{{< quizdown >}}

### What is the primary intent of the Composite Pattern?

- [x] To allow clients to treat individual objects and compositions of objects uniformly.
- [ ] To add responsibilities to objects dynamically.
- [ ] To provide a way to create a family of related objects.
- [ ] To ensure a class has only one instance.

> **Explanation:** The Composite Pattern allows clients to treat individual objects and compositions of objects uniformly, which is its primary intent.

### Which Erlang feature is particularly useful for implementing the Composite Pattern?

- [x] Pattern matching
- [ ] Object-oriented inheritance
- [ ] Dynamic typing
- [ ] Global variables

> **Explanation:** Erlang's pattern matching is particularly useful for implementing the Composite Pattern, as it allows for concise and expressive code.

### In the provided file system example, how is a file represented?

- [x] As a tuple `{file, Name, Size}`
- [ ] As a list `[file, Name, Size]`
- [ ] As a map `#{file => Name, Size}`
- [ ] As a binary

> **Explanation:** In the provided example, a file is represented as a tuple `{file, Name, Size}`.

### What operation does the `total_size` function perform?

- [x] It calculates the total size of all files in the file system.
- [ ] It counts the number of files in the file system.
- [ ] It lists all directories in the file system.
- [ ] It deletes empty directories.

> **Explanation:** The `total_size` function calculates the total size of all files in the file system by recursively summing their sizes.

### Which of the following is a benefit of using the Composite Pattern in Erlang?

- [x] Simplicity
- [x] Uniformity
- [ ] Complexity
- [ ] Inflexibility

> **Explanation:** The Composite Pattern in Erlang offers simplicity and uniformity, allowing operations to be applied uniformly to both individual elements and composites.

### What is a common use case for the Composite Pattern?

- [x] File systems
- [ ] Sorting algorithms
- [ ] Network protocols
- [ ] Cryptographic operations

> **Explanation:** A common use case for the Composite Pattern is file systems, where it helps manage hierarchical structures of files and directories.

### How can you extend the Composite Pattern in Erlang?

- [x] By adding new types of components
- [ ] By using global variables
- [ ] By implementing inheritance
- [ ] By using macros

> **Explanation:** You can extend the Composite Pattern in Erlang by adding new types of components to the hierarchy.

### What should you consider when implementing the Composite Pattern in Erlang?

- [x] Data structure choice
- [x] Performance
- [ ] Global state
- [ ] Object-oriented principles

> **Explanation:** When implementing the Composite Pattern in Erlang, consider the data structure choice and performance implications.

### True or False: The Composite Pattern is often compared to the Decorator Pattern.

- [x] True
- [ ] False

> **Explanation:** True. The Composite Pattern is often compared to the Decorator Pattern, although they serve different purposes.

### What is a key feature of Erlang that aids in implementing the Composite Pattern?

- [x] Recursion
- [ ] Inheritance
- [ ] Polymorphism
- [ ] Global variables

> **Explanation:** Recursion is a key feature of Erlang that aids in implementing the Composite Pattern, allowing for operations on hierarchical data structures.

{{< /quizdown >}}

Remember, the journey of mastering Erlang and its design patterns is ongoing. Keep exploring, experimenting, and learning. The Composite Pattern is just one of many tools in your toolkit for building robust and scalable applications.
