---
linkTitle: "13.2 Mocking, Stubbing, and Spying"
title: "Mocking, Stubbing, and Spying in JavaScript and TypeScript Testing"
description: "Explore the essential testing patterns of mocking, stubbing, and spying in JavaScript and TypeScript. Learn how to simulate, replace, and monitor code behavior effectively."
categories:
- Software Development
- Testing
- JavaScript
tags:
- Mocking
- Stubbing
- Spying
- Jest
- Sinon.js
date: 2024-10-25
type: docs
nav_weight: 1320000
canonical: "https://softwarepatternslexicon.com/patterns-js/13/2"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 13.2 Mocking, Stubbing, and Spying

In the realm of software testing, particularly in JavaScript and TypeScript, the concepts of mocking, stubbing, and spying are pivotal. These techniques allow developers to simulate, replace, and monitor the behavior of code components, ensuring that tests are both effective and isolated from external dependencies. This article delves into these testing patterns, providing insights into their implementation, best practices, and tools.

### Understand the Concepts

#### Mocking
Mocking involves creating mock objects that simulate the behavior of real objects in a controlled manner. This is particularly useful when you want to test the interaction between objects without relying on the actual implementation.

- **Purpose:** To isolate the unit of code being tested by replacing dependencies with mock objects.
- **Use Case:** Testing a function that interacts with a database without making actual database calls.

#### Stubbing
Stubbing replaces parts of the system under test with stubs that return predefined responses. This technique is useful when you want to control the behavior of a function or method during testing.

- **Purpose:** To provide controlled responses from dependencies, ensuring predictable test outcomes.
- **Use Case:** Replacing a network request with a stub that returns a fixed response.

#### Spying
Spying involves monitoring how functions are called, including the arguments passed and the number of times they are invoked. Spies are useful for verifying the behavior of functions without altering their implementation.

- **Purpose:** To observe and verify the interactions with functions.
- **Use Case:** Ensuring a callback function is called with the correct arguments.

### Implementation Steps

#### 1. Identify Dependencies
Determine the external systems or modules your code interacts with. These could be databases, APIs, or other services that your code relies on.

#### 2. Create Mocks/Stubs
Use testing libraries to create mock objects or stub functions. Define the expected outputs or behaviors for these mocks/stubs.

```javascript
// Using Jest for mocking
const fetchData = jest.fn().mockResolvedValue({ data: 'mock data' });

// Using Sinon.js for stubbing
const stub = sinon.stub(api, 'getData').returns(Promise.resolve({ data: 'stubbed data' }));
```

#### 3. Inject Mocks/Stubs
Replace actual dependencies with mocks during testing. This ensures that your tests are isolated from external factors.

```javascript
// Injecting a mock into a function
function processData(fetchData) {
  return fetchData().then(data => data);
}

// Test
test('processData uses mock fetchData', async () => {
  const result = await processData(fetchData);
  expect(result).toEqual({ data: 'mock data' });
});
```

#### 4. Set Up Spies
Use spies to verify that functions are called correctly. Spies can be set up to track function calls and arguments.

```javascript
// Using Jest for spying
const callback = jest.fn();

function executeCallback(cb) {
  cb('argument');
}

// Test
test('executeCallback calls the callback with correct argument', () => {
  executeCallback(callback);
  expect(callback).toHaveBeenCalledWith('argument');
});
```

#### 5. Write Tests
Focus on the unit of code, ensuring external dependencies do not affect test results. This involves writing tests that are independent and reliable.

### Tools and Libraries

- **Jest:** A popular testing framework with built-in mocking and spying capabilities. It simplifies the process of creating mocks and spies.
- **Sinon.js:** A standalone library for creating test spies, stubs, and mocks. It provides a flexible API for controlling and verifying function behavior.

### Practice

- **Mock API Calls:** Test error handling and response processing without making real network requests. This is crucial for testing scenarios like network failures or specific API responses.
- **Use Spies for Callbacks:** Verify that callback functions are invoked as expected, with the correct arguments and the right number of times.

### Considerations

- **Avoid Overusing Mocks/Stubs:** Excessive use of mocks and stubs can lead to tests that are disconnected from reality. Ensure that your tests reflect real-world scenarios as closely as possible.
- **Accurate Mocking:** Ensure that the mocked behavior accurately reflects the real-world scenarios. This involves understanding the behavior of the actual dependencies and replicating it in your mocks.

### Best Practices

- **Maintain Test Isolation:** Ensure that each test is independent and does not rely on the state or outcome of other tests.
- **Use Descriptive Names:** Name your mocks, stubs, and spies descriptively to make your tests more readable and understandable.
- **Regularly Update Mocks:** Keep your mocks and stubs up-to-date with the actual implementation to prevent tests from becoming obsolete.

### Conclusion

Mocking, stubbing, and spying are essential techniques in the toolkit of any JavaScript or TypeScript developer. They enable the creation of robust, isolated tests that ensure code behaves as expected without relying on external systems. By understanding and implementing these patterns effectively, developers can enhance the reliability and maintainability of their codebases.

## Quiz Time!

{{< quizdown >}}

### What is the primary purpose of mocking in testing?

- [x] To isolate the unit of code being tested by replacing dependencies with mock objects.
- [ ] To monitor how functions are called.
- [ ] To replace parts of the system under test with stubs.
- [ ] To ensure that tests are disconnected from reality.

> **Explanation:** Mocking is used to isolate the unit of code being tested by replacing dependencies with mock objects, allowing for controlled testing environments.

### Which library provides built-in mocking and spying capabilities?

- [x] Jest
- [ ] Mocha
- [ ] Chai
- [ ] Jasmine

> **Explanation:** Jest is a popular testing framework that provides built-in mocking and spying capabilities, making it easy to create and manage mocks and spies.

### What is the role of stubbing in testing?

- [x] To provide controlled responses from dependencies, ensuring predictable test outcomes.
- [ ] To monitor how functions are called.
- [ ] To replace dependencies with mock objects.
- [ ] To ensure that tests are disconnected from reality.

> **Explanation:** Stubbing is used to provide controlled responses from dependencies, ensuring that tests have predictable outcomes.

### How can spies be used in testing?

- [x] To observe and verify the interactions with functions.
- [ ] To replace parts of the system under test with stubs.
- [ ] To isolate the unit of code being tested.
- [ ] To ensure that tests are disconnected from reality.

> **Explanation:** Spies are used to observe and verify interactions with functions, such as how many times they are called and with what arguments.

### What is a potential drawback of overusing mocks and stubs?

- [x] Tests might become disconnected from reality.
- [ ] Tests will always fail.
- [ ] Tests will be too slow.
- [ ] Tests will not cover enough code.

> **Explanation:** Overusing mocks and stubs can lead to tests that are disconnected from reality, as they may not accurately reflect real-world scenarios.

### Which tool is a standalone library for creating test spies, stubs, and mocks?

- [x] Sinon.js
- [ ] Jest
- [ ] Mocha
- [ ] Jasmine

> **Explanation:** Sinon.js is a standalone library specifically designed for creating test spies, stubs, and mocks, offering a flexible API for testing.

### What should be regularly updated to prevent tests from becoming obsolete?

- [x] Mocks and stubs
- [ ] Test descriptions
- [ ] Test frameworks
- [ ] Code comments

> **Explanation:** Mocks and stubs should be regularly updated to align with the actual implementation, preventing tests from becoming obsolete.

### What is a best practice when naming mocks, stubs, and spies?

- [x] Use descriptive names to make tests more readable.
- [ ] Use short names to save space.
- [ ] Use random names to avoid conflicts.
- [ ] Use numbers to differentiate them.

> **Explanation:** Using descriptive names for mocks, stubs, and spies makes tests more readable and understandable, aiding in maintenance and collaboration.

### Which of the following is a use case for mocking API calls?

- [x] Testing error handling without making real network requests.
- [ ] Monitoring how functions are called.
- [ ] Replacing parts of the system under test with stubs.
- [ ] Ensuring that tests are disconnected from reality.

> **Explanation:** Mocking API calls is useful for testing error handling and response processing without making real network requests, allowing for controlled testing scenarios.

### True or False: Spies alter the implementation of functions they monitor.

- [ ] True
- [x] False

> **Explanation:** Spies do not alter the implementation of functions they monitor; they simply observe and record how functions are called.

{{< /quizdown >}}
