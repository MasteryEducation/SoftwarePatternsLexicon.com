---
canonical: "https://softwarepatternslexicon.com/patterns-erlang/3/1/2"
title: "Erlang Records and Maps: Structured Data Storage with Named Fields"
description: "Explore the use of records and maps in Erlang for structured data storage, including syntax, compilation, and practical examples."
linkTitle: "3.1.2 Records and Maps"
categories:
- Erlang Programming
- Data Structures
- Functional Programming
tags:
- Erlang
- Records
- Maps
- Data Structures
- Functional Programming
date: 2024-11-23
type: docs
nav_weight: 31200
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## 3.1.2 Records and Maps

In Erlang, structured data storage is a fundamental aspect of building robust applications. Two primary constructs for achieving this are **records** and **maps**. Both offer unique ways to handle data with named fields, but they serve different purposes and have distinct characteristics. In this section, we'll delve into the syntax, usage, advantages, and limitations of records and maps, providing you with a comprehensive understanding of when and how to use each.

### Understanding Records in Erlang

**Records** in Erlang are a way to group related data together using named fields. They are similar to structs in C or objects in other programming languages, but with a functional twist. Records are defined at compile-time and provide a convenient way to handle fixed-format data.

#### Defining Records

To define a record, you use the `-record` directive. Here's the basic syntax:

```erlang
-record(person, {name, age, occupation}).
```

In this example, we define a record named `person` with three fields: `name`, `age`, and `occupation`.

#### Using Records

Once defined, you can create instances of a record and access its fields. Here's how you can use the `person` record:

```erlang
% Creating a record instance
Person = #person{name = "Alice", age = 30, occupation = "Engineer"}.

% Accessing record fields
Name = Person#person.name,
Age = Person#person.age.

% Updating a record field
UpdatedPerson = Person#person{age = 31}.
```

**Key Points:**
- **Record Syntax**: Use `#record_name{field1 = Value1, ...}` to create a record.
- **Field Access**: Use `Record#record_name.field` to access a field.
- **Field Update**: Use `Record#record_name{field = NewValue}` to update a field.

#### Compilation of Records

Records are a compile-time construct in Erlang. They are essentially syntactic sugar for tuples, which means that when you compile your Erlang code, the record definitions are translated into tuple operations. This makes records efficient but also means they lack some of the dynamic flexibility found in other data structures.

### Understanding Maps in Erlang

**Maps** are a more recent addition to Erlang, introduced in version 17. They provide a flexible way to store key-value pairs and are more dynamic than records. Maps are particularly useful when you need to handle data with varying structures or when the number of fields is not fixed.

#### Defining and Using Maps

Maps are defined using the `#{}` syntax. Here's an example of creating and using a map:

```erlang
% Creating a map
PersonMap = #{name => "Alice", age => 30, occupation => "Engineer"}.

% Accessing map values
Name = maps:get(name, PersonMap),
Age = maps:get(age, PersonMap).

% Updating a map
UpdatedPersonMap = maps:put(age, 31, PersonMap).
```

**Key Points:**
- **Map Syntax**: Use `#{Key1 => Value1, ...}` to create a map.
- **Value Access**: Use `maps:get(Key, Map)` to access a value.
- **Value Update**: Use `maps:put(Key, NewValue, Map)` to update a value.

#### Differences Between Records and Maps

While both records and maps allow you to store data with named fields, they have several differences:

- **Flexibility**: Maps are more flexible as they allow dynamic addition and removal of fields, whereas records have a fixed structure defined at compile-time.
- **Performance**: Records are generally faster for access and updates due to their tuple-based implementation, but maps offer more flexibility at the cost of some performance.
- **Use Cases**: Use records when you have a fixed data structure and maps when you need dynamic field management.

### Advantages and Limitations

#### Advantages of Records

- **Efficiency**: Records are efficient due to their tuple-based implementation.
- **Compile-Time Checks**: Errors related to field names can be caught at compile-time.
- **Readability**: Named fields improve code readability and maintainability.

#### Limitations of Records

- **Inflexibility**: Records cannot be modified at runtime, making them unsuitable for dynamic data structures.
- **Dependency on Compilation**: Any change in the record structure requires recompilation of the code.

#### Advantages of Maps

- **Flexibility**: Maps allow dynamic addition, removal, and modification of fields.
- **Ease of Use**: Maps provide a straightforward way to handle key-value pairs.

#### Limitations of Maps

- **Performance Overhead**: Maps may have a performance overhead compared to records, especially for large data sets.
- **Lack of Compile-Time Checks**: Errors related to keys are only caught at runtime.

### When to Use Records vs. Maps

- **Use Records** when you have a well-defined, fixed data structure and need efficient access and updates.
- **Use Maps** when you need flexibility in your data structure, such as when dealing with JSON-like data or when the number of fields can change dynamically.

### Code Examples

Let's explore some practical examples to solidify our understanding of records and maps.

#### Example 1: Using Records

```erlang
-module(record_example).
-export([create_person/0, update_person_age/1]).

-record(person, {name, age, occupation}).

create_person() ->
    #person{name = "Bob", age = 25, occupation = "Developer"}.

update_person_age(Person) ->
    Person#person{age = Person#person.age + 1}.
```

#### Example 2: Using Maps

```erlang
-module(map_example).
-export([create_person_map/0, update_person_age/1]).

create_person_map() ->
    #{name => "Bob", age => 25, occupation => "Developer"}.

update_person_age(PersonMap) ->
    maps:put(age, maps:get(age, PersonMap) + 1, PersonMap).
```

### Visualizing Records and Maps

To better understand the differences between records and maps, let's use a diagram to illustrate their structures.

```mermaid
graph TD;
    A[Record: #person{name, age, occupation}] --> B[(Tuple)]
    C[Map: #{name => value, age => value}] --> D[(Key-Value Pairs)]
```

**Diagram Description**: This diagram shows how a record is essentially a tuple with named fields, while a map is a collection of key-value pairs.

### Try It Yourself

To deepen your understanding, try modifying the code examples:

- **Add a new field** to the `person` record and update the code to handle it.
- **Remove a field** from the map and observe how the code behaves.
- **Experiment with performance** by timing access and updates for both records and maps.

### References and Further Reading

- [Erlang Documentation on Records](https://www.erlang.org/doc/reference_manual/records.html)
- [Erlang Documentation on Maps](https://www.erlang.org/doc/man/maps.html)

### Knowledge Check

- **What are the key differences between records and maps in Erlang?**
- **When would you choose to use a map over a record?**

### Embrace the Journey

Remember, mastering records and maps is just one step in your Erlang journey. As you progress, you'll discover more powerful ways to structure and manage data. Keep experimenting, stay curious, and enjoy the process!

## Quiz: Records and Maps

{{< quizdown >}}

### What is a primary advantage of using records in Erlang?

- [x] Efficiency due to tuple-based implementation
- [ ] Flexibility in adding fields at runtime
- [ ] Built-in support for JSON serialization
- [ ] Automatic field validation

> **Explanation:** Records are efficient because they are implemented as tuples, which are optimized for performance in Erlang.

### How are maps different from records in Erlang?

- [x] Maps allow dynamic addition and removal of fields
- [ ] Maps are compiled into tuples
- [ ] Maps require compile-time definition
- [ ] Maps are less flexible than records

> **Explanation:** Maps provide flexibility by allowing dynamic changes to the data structure, unlike records which are fixed at compile-time.

### Which syntax is used to define a record in Erlang?

- [x] -record(name, {field1, field2})
- [ ] #{field1 => value1, field2 => value2}
- [ ] {field1, field2}
- [ ] [field1, field2]

> **Explanation:** The `-record` directive is used to define records in Erlang.

### What is a limitation of using records in Erlang?

- [x] Inflexibility due to fixed structure
- [ ] Lack of compile-time checks
- [ ] Performance overhead for large data sets
- [ ] Difficulty in accessing fields

> **Explanation:** Records have a fixed structure defined at compile-time, making them inflexible for dynamic data.

### When should you use maps in Erlang?

- [x] When you need a flexible data structure
- [ ] When you need compile-time checks
- [ ] When performance is the primary concern
- [ ] When dealing with fixed-format data

> **Explanation:** Maps are ideal for situations where the data structure needs to be flexible and dynamic.

### How do you access a field in a record?

- [x] Record#record_name.field
- [ ] maps:get(field, Record)
- [ ] Record[field]
- [ ] Record.field

> **Explanation:** The syntax `Record#record_name.field` is used to access fields in a record.

### How do you update a value in a map?

- [x] maps:put(Key, NewValue, Map)
- [ ] Map#map_name{Key = NewValue}
- [ ] Map[Key] = NewValue
- [ ] maps:update(Key, NewValue, Map)

> **Explanation:** The `maps:put` function is used to update values in a map.

### What is a disadvantage of using maps?

- [x] Performance overhead compared to records
- [ ] Lack of flexibility
- [ ] Requirement for compile-time definition
- [ ] Inability to handle key-value pairs

> **Explanation:** Maps may have a performance overhead compared to records, especially for large data sets.

### True or False: Records in Erlang are dynamically typed.

- [ ] True
- [x] False

> **Explanation:** Records are not dynamically typed; they are defined at compile-time and translated into tuples.

### Which of the following is a correct way to create a map in Erlang?

- [x] #{key1 => value1, key2 => value2}
- [ ] -record(name, {key1, key2})
- [ ] {key1, key2}
- [ ] [key1, key2]

> **Explanation:** The `#{}` syntax is used to create maps in Erlang.

{{< /quizdown >}}
