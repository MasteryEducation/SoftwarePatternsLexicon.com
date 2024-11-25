---
linkTitle: "Test Double"
title: "Test Double: Replacing Real Objects in Tests with Simpler Objects"
description: "Exploring how Test Doubles can be used in functional programming to replace real objects in tests with simpler, behavior-mimicking objects."
categories:
- Functional Programming
- Design Patterns
tags:
- Test Double
- Mock
- Stub
- Fake
- Test Patterns
date: 2024-07-07
type: docs
series:
  - Functional Programming
canonical: "https://softwarepatternslexicon.com/functional/testing-patterns/testing-strategies/test-double"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

In the world of software development, testing is a cornerstone for ensuring code quality, correctness, and robustness. Functional programming, with its emphasis on pure functions and immutability, provides an excellent foundation for writing testable code. However, there are instances where testing certain parts of the code can be challenging due to dependencies on external systems, stateful services, or complex interactions. This is where the **Test Double** design pattern comes to the rescue.

Test Doubles are simplified versions of collaborating objects that mimic the behavior of real objects during testing. They replace real components in tests to overcome dependencies and achieve isolation in testing. The Test Double pattern encompasses various types, including Mocks, Stubs, Fakes, and Spies.

## Types of Test Doubles

Test Doubles can be classified into several categories based on their functionality and purpose:

1. **Dummy**
2. **Stub**
3. **Fake**
4. **Mock**
5. **Spy**

### Dummy

A Dummy is an object that is passed around but never actually used. Dummies are useful for filling parameter lists.

```kotlin
// Example in Kotlin
data class DummyService(val id: Int = 0)

fun process(dummy: DummyService) {
    // Do nothing with dummy
}

fun testProcess() {
    val dummyService = DummyService()
    process(dummyService) // Using dummy service to fulfill the parameter requirement
}
```

### Stub

A Stub is an object that provides predefined responses to method calls. They typically do not replace full objects but help simulate responses to certain inputs.

```kotlin
class NetworkStub : Network {
    override fun fetchData(): String {
        return "Mocked Data"
    }
}

fun getData(network: Network): String {
    return network.fetchData()
}

fun testGetData() {
    val networkStub = NetworkStub()
    val data = getData(networkStub)
    assert(data == "Mocked Data")
}
```

### Fake

A Fake is a working implementation, but it is not suitable for production (e.g., an in-memory database).

```kotlin
class InMemoryDatabase : Database {
    private val data = mutableListOf<String>()

    override fun save(value: String) {
        data.add(value)
    }

    override fun fetchAll(): List<String> {
        return data
    }
}

fun testDatabase() {
    val fakeDb = InMemoryDatabase()
    fakeDb.save("Test")
    assert(fakeDb.fetchAll() == listOf("Test"))
}
```

### Mock

Mocks are objects that not only simulate behavior but also enforce expectations in terms of method calls and their order. Mocks are generally verified post-execution.

```kotlin
class LoggingService {
    fun log(message: String) {
        println(message)
    }
}

fun processWithLogging(loggingService: LoggingService) {
    loggingService.log("Processing started")
    // Perform some task
    loggingService.log("Processing completed")
}

fun testProcessWithLogging() {
    val mockLogger = mock<LoggingService>()
    
    Mockito.`when`(mockLogger.log("Processing started")).then { println("Processing started") }
    Mockito.`when`(mockLogger.log("Processing completed")).then { println("Processing completed") }
    
    processWithLogging(mockLogger)
    
    verify(mockLogger).log("Processing started")
    verify(mockLogger).log("Processing completed")
}
```

### Spy

Spies are partial mocks that typically track interactions with real objects. They allow for behavior verification while keeping the underlying object's functionality intact.

```kotlin
class ListService {
    private val items = mutableListOf<String>()
    
    fun addItem(item: String) {
        items.add(item)
    }
    
    fun getItemsCount(): Int {
        return items.size
    }
}

fun testListServiceSpy() {
    val listService = ListService()
    val spyService = Mockito.spy(listService)
    
    spyService.addItem("Item1")
    
    Mockito.verify(spyService).addItem("Item1")
    assert(spyService.getItemsCount() == 1)
}
```

## Related Design Patterns and Principles

- **Null Object Pattern**: This pattern involves an object with defined neutral ("null") behavior. It simplifies code by avoiding null checks.
  
- **Dependency Injection**: Test Doubles are often used in systems where dependency injection is prevalent. It allows for replacing real objects with Test Doubles and facilitates testing.

- **Factory Method**: This pattern can be used to create instances of Test Doubles as part of setup procedures in tests.

## Additional Resources

- [XUnit Test Patterns](http://xunitpatterns.com/Test%20Double.html) by Gerard Meszaros
- [Testing with Mocks in Scala](https://www.baeldung.com/scala/testing-with-mockito) on Baeldung
- [Mockito-Kotlin](https://github.com/nhaarman/mockito-kotlin) for more Kotlin examples

## Summary

Test Doubles are crucial in functional programming and testing paradigms, facilitating isolation and control during tests. They help ensure that our tests are robust, maintainable, and free from external dependencies, leading to higher confidence in our code quality. Understanding and effectively using the various types of Test Doubles—Dummies, Stubs, Fakes, Mocks, and Spies—is essential for any developer involved in writing automated tests.

By incorporating these patterns and practices, developers can write more reliable and maintainable tests that accurately mimic real-world scenarios without the complications and unpredictabilities of external dependencies.
