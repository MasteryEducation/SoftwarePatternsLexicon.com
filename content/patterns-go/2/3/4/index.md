---

linkTitle: "2.3.4 Iterator"
title: "Iterator Design Pattern in Go: Accessing Collections Efficiently"
description: "Explore the Iterator design pattern in Go, enabling sequential access to elements of a collection without exposing its underlying structure. Learn implementation steps, use cases, and Go-specific tips with practical examples."
categories:
- Design Patterns
- Go Programming
- Software Architecture
tags:
- Iterator
- Design Patterns
- GoLang
- Behavioral Patterns
- Software Development
date: 2024-10-25
type: docs
nav_weight: 234000
canonical: "https://softwarepatternslexicon.com/patterns-go/2/3/4"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 2.3.4 Iterator

The Iterator pattern is a fundamental design pattern that provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation. This pattern is particularly useful in scenarios where you need to traverse complex data structures or collections in a consistent manner.

### Understand the Intent

The primary intent of the Iterator pattern is to:

- **Provide a way to access elements of an aggregate object sequentially** without exposing its underlying structure.
- **Support multiple traversal methods** over a collection, allowing flexibility in how elements are accessed.

### Implementation Steps

Implementing the Iterator pattern in Go involves several key steps:

#### 1. Define Iterator Interface

The first step is to define an interface that outlines the methods required for iteration. Common methods include:

- `Next()`: Advances the iterator to the next element.
- `HasNext()`: Checks if there are more elements to iterate over.
- `Value()`: Retrieves the current element.

```go
type Iterator interface {
    Next() bool
    Value() interface{}
}
```

#### 2. Implement Concrete Iterators

Create structs that implement the `Iterator` interface for specific collections. Each concrete iterator will manage its own state and traversal logic.

```go
type LinkedListIterator struct {
    current *Node
}

func (it *LinkedListIterator) Next() bool {
    if it.current != nil {
        it.current = it.current.next
        return it.current != nil
    }
    return false
}

func (it *LinkedListIterator) Value() interface{} {
    if it.current != nil {
        return it.current.value
    }
    return nil
}
```

#### 3. Aggregate Provides Iterator

The collection (aggregate) should have a method that returns an iterator. This method initializes the iterator with the starting state.

```go
type LinkedList struct {
    head *Node
}

func (list *LinkedList) Iterator() Iterator {
    return &LinkedListIterator{current: list.head}
}
```

#### 4. Traverse Using Iterator

Client code can use the iterator to access elements in the collection without needing to know the internal structure.

```go
func main() {
    list := &LinkedList{}
    // Assume list is populated with nodes

    iterator := list.Iterator()
    for iterator.Next() {
        fmt.Println(iterator.Value())
    }
}
```

### When to Use

Consider using the Iterator pattern in the following scenarios:

- When you need to access a collection's elements without exposing its internal representation.
- To support different traversal strategies, such as forward, backward, or custom traversal logic.

### Go-Specific Tips

- **Leverage Go's `for range` loop** for simple collections like slices and maps, which inherently support iteration.
- **Implement custom iterators** for complex traversal logic that cannot be achieved with `for range`.

### Example: Iterating Over a Custom Linked List

Let's explore a practical example of iterating over a custom linked list, including reverse iteration.

```go
type Node struct {
    value int
    next  *Node
}

type LinkedList struct {
    head *Node
}

func (list *LinkedList) Add(value int) {
    newNode := &Node{value: value}
    if list.head == nil {
        list.head = newNode
    } else {
        current := list.head
        for current.next != nil {
            current = current.next
        }
        current.next = newNode
    }
}

type LinkedListIterator struct {
    current *Node
}

func (it *LinkedListIterator) Next() bool {
    if it.current != nil {
        it.current = it.current.next
        return it.current != nil
    }
    return false
}

func (it *LinkedListIterator) Value() interface{} {
    if it.current != nil {
        return it.current.value
    }
    return nil
}

func (list *LinkedList) Iterator() Iterator {
    return &LinkedListIterator{current: list.head}
}

func main() {
    list := &LinkedList{}
    list.Add(1)
    list.Add(2)
    list.Add(3)

    iterator := list.Iterator()
    for iterator.Next() {
        fmt.Println(iterator.Value())
    }
}
```

### Implementing Reverse Iteration

To implement reverse iteration, you would need a different iterator that traverses the list from tail to head. This requires additional logic to maintain a reference to the previous node.

### Advantages and Disadvantages

**Advantages:**

- **Encapsulation:** Hides the internal structure of the collection.
- **Flexibility:** Supports different traversal strategies.
- **Reusability:** Iterators can be reused across different collections.

**Disadvantages:**

- **Overhead:** May introduce additional complexity and overhead for simple collections.
- **State Management:** Requires careful management of iterator state, especially for complex data structures.

### Best Practices

- **Use Iterators for Complex Structures:** Reserve iterators for complex data structures where direct access is not feasible.
- **Keep Iterators Simple:** Ensure that iterator logic is straightforward to avoid unnecessary complexity.
- **Combine with Other Patterns:** Consider combining iterators with other patterns like Composite for tree structures.

### Comparisons

The Iterator pattern is often compared with the `for range` loop in Go. While `for range` is suitable for built-in collections, iterators provide more control and flexibility for custom collections.

### Conclusion

The Iterator pattern is a powerful tool for accessing elements in a collection without exposing its internal structure. By implementing iterators in Go, you can enhance the flexibility and maintainability of your code, especially when dealing with complex data structures.

## Quiz Time!

{{< quizdown >}}

### What is the primary intent of the Iterator pattern?

- [x] To provide a way to access elements of an aggregate object sequentially without exposing its underlying structure.
- [ ] To modify elements of a collection directly.
- [ ] To sort elements in a collection.
- [ ] To delete elements from a collection.

> **Explanation:** The Iterator pattern is designed to allow sequential access to elements without exposing the collection's internal structure.

### Which method is NOT typically part of an Iterator interface?

- [ ] Next()
- [ ] HasNext()
- [x] Add()
- [ ] Value()

> **Explanation:** The `Add()` method is not part of the Iterator interface; it is typically used for modifying collections.

### What does the `Next()` method in an iterator do?

- [x] Advances the iterator to the next element.
- [ ] Retrieves the current element.
- [ ] Checks if there are more elements.
- [ ] Resets the iterator to the start.

> **Explanation:** The `Next()` method is responsible for moving the iterator to the next element in the collection.

### In Go, which loop can be used for simple collection iteration without a custom iterator?

- [x] for range
- [ ] while
- [ ] do-while
- [ ] foreach

> **Explanation:** Go's `for range` loop is used for iterating over slices, maps, and channels.

### When should you consider using a custom iterator in Go?

- [x] When you need to traverse complex data structures.
- [ ] When iterating over a simple slice.
- [ ] When using built-in maps.
- [ ] When iterating over strings.

> **Explanation:** Custom iterators are useful for complex data structures where built-in iteration is insufficient.

### What is a disadvantage of using the Iterator pattern?

- [x] It may introduce additional complexity and overhead.
- [ ] It makes code less readable.
- [ ] It exposes the internal structure of collections.
- [ ] It limits traversal strategies.

> **Explanation:** The Iterator pattern can add complexity, especially for simple collections where it's not necessary.

### Which of the following is a benefit of using the Iterator pattern?

- [x] It encapsulates the internal structure of the collection.
- [ ] It allows direct modification of collection elements.
- [ ] It simplifies the collection's internal logic.
- [ ] It automatically sorts the collection.

> **Explanation:** Encapsulation of the collection's structure is a key benefit of the Iterator pattern.

### How can you implement reverse iteration in a linked list?

- [x] By creating a new iterator that traverses from tail to head.
- [ ] By using the same iterator and reversing the list.
- [ ] By modifying the `Next()` method to go backward.
- [ ] By using Go's `for range` loop.

> **Explanation:** Reverse iteration requires a separate iterator that manages traversal from the tail to the head.

### What is a common method name in an Iterator interface for checking more elements?

- [ ] NextElement()
- [ ] More()
- [x] HasNext()
- [ ] IsDone()

> **Explanation:** `HasNext()` is commonly used to check if there are more elements to iterate over.

### The Iterator pattern supports multiple traversal methods over a collection.

- [x] True
- [ ] False

> **Explanation:** True, the Iterator pattern allows for different traversal strategies, such as forward and reverse iteration.

{{< /quizdown >}}
